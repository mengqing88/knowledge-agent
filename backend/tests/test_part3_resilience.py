import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest
from app.core.errors import BudgetExhausted, ProviderError
from app.integrations.providers import ModelProvider
from app.integrations.resilience import circuits


class Response:
    def __init__(self, status, headers=None):
        self.status_code, self.headers = status, headers or {}

    def json(self):
        return {"ok": True}


class Client:
    def __init__(self, responses, requests):
        self.responses, self.requests = responses, requests

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, url, headers, json):
        self.requests.append((url, headers, json))
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def provider_for(settings, monkeypatch, responses):
    settings.chat_api_key = "test-secret-value"
    settings.chat_base_url = "https://model.example/v1"
    requests = []
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: Client(responses, requests))
    with circuits.lock:
        circuits.states.clear()
    return ModelProvider(settings), requests


def invoke(provider, events=None, seconds=5):
    with provider.execution(time.monotonic() + seconds, lambda: None,
                            (events if events is not None else []).append):
        return provider._post(provider.settings.chat_base_url, provider.settings.chat_api_key,
                              "responses", {"model": provider.settings.model, "input": "test"})


def test_rate_limit_honors_retry_after_within_deadline_and_records_safe_events(settings, monkeypatch):
    settings.provider_backoff_seconds = 0
    responses = [Response(429, {"Retry-After": "0"}), Response(200)]
    provider, requests = provider_for(settings, monkeypatch, responses)
    events = []

    assert invoke(provider, events) == {"ok": True}
    assert len(requests) == 2
    assert requests[0][1]["Authorization"] == "Bearer test-secret-value"
    assert [event["phase"] for event in events] == ["started", "response", "retry", "started", "response"]
    assert "test-secret-value" not in str(events)


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_retryable_server_failures_retry_once_with_a_bounded_limit(settings, monkeypatch, status):
    settings.provider_backoff_seconds = 0
    provider, requests = provider_for(settings, monkeypatch, [Response(status), Response(200)])

    assert invoke(provider) == {"ok": True}
    assert len(requests) == 2


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_http_errors_are_not_replayed(settings, monkeypatch, status):
    provider, requests = provider_for(settings, monkeypatch, [Response(status), Response(200)])
    with pytest.raises(ProviderError) as exc:
        invoke(provider)
    assert exc.value.code == f"provider_http_{status}"
    assert not exc.value.retryable
    assert len(requests) == 1


def test_retry_after_is_a_minimum_and_backoff_jitter_is_bounded(settings, monkeypatch):
    settings.provider_backoff_seconds = 2
    settings.provider_backoff_jitter = .25
    provider = ModelProvider(settings)
    import app.integrations.providers as providers
    monkeypatch.setattr(providers.random, "random", lambda: 0)
    assert provider.retry_delay("3", 0) == 3
    assert provider.retry_delay(None, 1) == 3
    monkeypatch.setattr(providers.random, "random", lambda: 1)
    assert provider.retry_delay("3", 0) == 3.75
    assert provider.retry_delay(None, 1) == 5


def test_retry_after_http_date_and_invalid_value_are_handled(settings, monkeypatch):
    settings.provider_backoff_seconds = 2
    settings.provider_backoff_jitter = 0
    provider = ModelProvider(settings)
    future = datetime.now(timezone.utc) + timedelta(minutes=10)
    assert provider.retry_delay(format_datetime(future, usegmt=True), 0) >= 599
    assert provider.retry_delay("not-a-date", 0) == 2


def test_repeated_retryable_failure_stops_at_configured_attempt_limit(settings, monkeypatch):
    settings.provider_max_retries = 2
    settings.provider_backoff_seconds = 0
    provider, requests = provider_for(settings, monkeypatch,
                                      [Response(503), Response(503), Response(503), Response(200)])
    with pytest.raises(ProviderError) as exc:
        invoke(provider)
    assert exc.value.code == "provider_http_503"
    assert len(requests) == settings.provider_max_retries + 1


def test_cancelled_during_backoff_does_not_start_next_request(settings, monkeypatch):
    settings.provider_backoff_seconds = 1
    provider, requests = provider_for(settings, monkeypatch, [Response(503), Response(200)])
    cancelled = [False]

    def guard():
        if cancelled[0]:
            raise BudgetExhausted("cancelled")

    def cancel_during_sleep(_seconds):
        cancelled[0] = True

    import app.integrations.providers as providers
    monkeypatch.setattr(providers.time, "sleep", cancel_during_sleep)

    with pytest.raises(BudgetExhausted):
        with provider.execution(time.monotonic() + 5, guard, lambda _event: None):
            provider._post(provider.settings.chat_base_url, provider.settings.chat_api_key,
                           "responses", {"model": provider.settings.model, "input": "test"})
    assert len(requests) == 1


def test_read_timeout_is_not_replayed_because_completion_is_unknown(settings, monkeypatch):
    provider, requests = provider_for(settings, monkeypatch, [httpx.ReadTimeout("late response")])
    with pytest.raises(ProviderError) as exc:
        invoke(provider)
    assert exc.value.code == "provider_timeout"
    assert exc.value.retryable
    assert len(requests) == 1


def test_repeated_server_failures_open_a_circuit_until_cooldown(settings, monkeypatch):
    settings.provider_max_retries = 0
    settings.circuit_failure_threshold = 2
    provider, requests = provider_for(settings, monkeypatch, [Response(503), Response(503)])
    for _ in range(2):
        with pytest.raises(ProviderError, match="HTTP 503"):
            invoke(provider)
    with pytest.raises(ProviderError) as exc:
        invoke(provider)
    assert exc.value.code == "circuit_open"
    assert len(requests) == 2


def test_expired_task_deadline_blocks_provider_request(settings, monkeypatch):
    provider, requests = provider_for(settings, monkeypatch, [Response(200)])
    with pytest.raises(BudgetExhausted):
        invoke(provider, seconds=-1)
    assert requests == []


@pytest.mark.parametrize("stage", ["quota", "retry", "progress"])
@pytest.mark.parametrize("deadline_kind", ["run", "provider"])
def test_preflight_wait_cannot_dispatch_after_deadline(settings, monkeypatch, stage, deadline_kind):
    import app.integrations.providers as providers

    settings.provider_backoff_seconds = 0
    settings.model_timeout_seconds = 5 if deadline_kind == "provider" else 45
    responses = ([Response(503)] if stage == "retry" else []) + [Response(200)]
    provider, requests = provider_for(settings, monkeypatch, responses)
    stamp = [100.]
    monkeypatch.setattr(providers, "time", SimpleNamespace(monotonic=lambda: stamp[0]))

    def wait(*_args):
        stamp[0] += 6

    if stage == "quota":
        monkeypatch.setattr(providers, "reserve", wait)
    elif stage == "retry":
        provider.before_retry = wait

    def observe(event):
        if stage == "progress" and event["phase"] == "started":
            wait()

    expected = BudgetExhausted if deadline_kind == "run" else ProviderError
    with provider.execution(105. if deadline_kind == "run" else 130., lambda: None, observe):
        with pytest.raises(expected) as exc:
            provider._post(settings.chat_base_url, settings.chat_api_key, "responses", {"model": settings.model})
    assert getattr(exc.value, "category", getattr(exc.value, "code", None)) == (
        "time" if deadline_kind == "run" else "provider_timeout")
    assert len(requests) == (1 if stage == "retry" else 0)


def test_http_timeout_uses_remaining_budget_after_preflight_wait(settings, monkeypatch):
    import app.integrations.providers as providers

    provider, requests = provider_for(settings, monkeypatch, [Response(200)])
    stamp, timeouts = [100.], []
    monkeypatch.setattr(providers, "time", SimpleNamespace(monotonic=lambda: stamp[0]))

    def wait(*_args):
        stamp[0] += 3

    def client(**kwargs):
        timeouts.append(kwargs["timeout"])
        return Client([Response(200)], requests)

    monkeypatch.setattr(providers, "reserve", wait)
    monkeypatch.setattr(httpx, "Client", client)
    with provider.execution(105., lambda: None, lambda _event: None):
        assert provider._post(settings.chat_base_url, settings.chat_api_key, "responses", {}) == {"ok": True}
    assert timeouts == [2.]
    assert len(requests) == 1


def test_revocation_during_quota_wait_blocks_dispatch(settings, monkeypatch):
    import app.integrations.providers as providers
    from app.core.errors import InterruptedRun

    provider, requests = provider_for(settings, monkeypatch, [Response(200)])
    cancelled = [False]

    def reserve(*_args):
        cancelled[0] = True

    def guard():
        if cancelled[0]:
            raise InterruptedRun()

    monkeypatch.setattr(providers, "reserve", reserve)
    with provider.execution(time.monotonic() + 5, guard, lambda _event: None):
        with pytest.raises(InterruptedRun):
            provider._post(settings.chat_base_url, settings.chat_api_key, "responses", {})
    assert requests == []


def test_retry_after_beyond_task_deadline_returns_without_replaying(settings, monkeypatch):
    provider, requests = provider_for(settings, monkeypatch, [Response(429, {"Retry-After": "20"}), Response(200)])
    with pytest.raises(ProviderError) as exc:
        invoke(provider, seconds=0.05)
    assert exc.value.code == "provider_http_429"
    assert len(requests) == 1


def test_known_secret_is_redacted_before_provider_dispatch(settings, monkeypatch):
    provider, requests = provider_for(settings, monkeypatch, [Response(200)])
    with provider.execution(time.monotonic() + 5, lambda: None, lambda _event: None):
        provider._post(provider.settings.chat_base_url, provider.settings.chat_api_key, "responses", {
            "model": provider.settings.model, "input": "credential test-secret-value",
        })
    assert requests[0][2]["input"] == "credential [REDACTED]"
