from concurrent.futures import ThreadPoolExecutor

import pytest
from app.core.errors import ProviderError
from app.integrations.resilience import Circuits


def test_circuit_opens_after_threshold_allows_one_probe_and_recovers(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("app.integrations.resilience.time.monotonic", lambda: now[0])
    circuit = Circuits()

    for _ in range(3):
        circuit.enter("chat-provider", cooldown=30)
        circuit.finish("chat-provider", failed=True, threshold=3)

    with pytest.raises(ProviderError, match="熔断"):
        circuit.enter("chat-provider", cooldown=30)

    now[0] += 30
    circuit.enter("chat-provider", cooldown=30)
    with pytest.raises(ProviderError, match="熔断"):
        circuit.enter("chat-provider", cooldown=30)

    circuit.finish("chat-provider", failed=False, threshold=3)
    circuit.enter("chat-provider", cooldown=30)


def test_circuit_allows_only_one_concurrent_half_open_probe(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("app.integrations.resilience.time.monotonic", lambda: now[0])
    circuit = Circuits()
    circuit.enter("chat-provider", cooldown=30)
    circuit.finish("chat-provider", failed=True, threshold=1)
    now[0] += 30

    def probe(_):
        try:
            circuit.enter("chat-provider", cooldown=30)
            return "probe"
        except Exception as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(probe, range(8)))
    assert outcomes.count("probe") == 1
    assert outcomes.count("circuit_open") == 7
    circuit.finish("chat-provider", failed=False, threshold=1, healthy=True)
