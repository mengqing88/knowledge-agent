"""Durable, single-process experiment budget shared by every charged request.

Reservations use UTF-8 bytes plus protocol headroom as a conservative token
estimate. Unconfirmed requests retain their entire reservation and stop the run.
This ledger estimates charges; it is not a provider billing statement.
"""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from app.core.errors import BudgetExhausted
from app.integrations.providers import ModelProvider


def decimal(value):
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError("Invalid nonnegative price/budget")
    return result


class BudgetLedger:
    def __init__(self, path, prices, hard_limit="3", stop_at="2.5"):
        self.path = Path(path)
        self.prices = prices
        self.hard_limit, self.stop_at = decimal(hard_limit), decimal(stop_at)
        if not 0 < self.stop_at < self.hard_limit:
            raise ValueError("Budget must leave a margin")
        for price in prices.values():
            if not price.get("source") or not price.get("model") or not price.get("base_url"):
                raise ValueError("Verified price/model/endpoint source required")
            decimal(price["input"])
            decimal(price["output"])
        self.data = {"currency": "CNY", "hard_limit": str(self.hard_limit), "stop_at": str(self.stop_at),
                     "prices": prices, "calls": [], "adjustments": [], "halt_reason": None}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if (self.data["prices"] != prices or self.data["hard_limit"] != str(self.hard_limit)
                    or self.data["stop_at"] != str(self.stop_at)):
                raise ValueError("Existing ledger identity/limits differ")
            self.data.setdefault("adjustments", [])
            if any(c["status"] == "reserved" for c in self.data["calls"]):
                self.data["halt_reason"] = "interrupted_request_usage_unknown"
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".writing")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def total(self):
        calls = sum((decimal(c.get("actual_cost", c["reserved_cost"])) for c in self.data["calls"]), Decimal(0))
        adjustments = sum((decimal(item["amount"]) for item in self.data.get("adjustments", [])), Decimal(0))
        return calls + adjustments

    def guard(self):
        if self.data["halt_reason"] or self.total() >= self.stop_at:
            raise BudgetExhausted("evaluation_global_budget_or_unknown_usage")

    def reserve(self, kind, base, model, input_bound, output_bound):
        self.guard()
        price = self.prices[kind]
        if base.rstrip("/") != price["base_url"].rstrip("/") or model != price["model"]:
            raise ValueError("Unpriced model or endpoint; request blocked")
        amount = (decimal(input_bound) * decimal(price["input"]) +
                  decimal(output_bound) * decimal(price["output"])) / 1_000_000
        if self.total() + amount > self.stop_at:
            self.data["halt_reason"] = "next_maximum_reservation_exceeds_stop_at"
            self.save()
            raise BudgetExhausted("evaluation_global_budget")
        call = {"kind": kind, "model": model, "status": "reserved", "reserved_cost": str(amount),
                "input_bound": input_bound, "output_bound": output_bound,
                "started_at": datetime.now(timezone.utc).isoformat()}
        self.data["calls"].append(call)
        self.save()  # Must reach disk before sending.
        return len(self.data["calls"]) - 1

    def settle(self, index, result=None):
        call = self.data["calls"][index]
        usage = result.get("usage") if isinstance(result, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        inp = usage.get("input_tokens", usage.get("prompt_tokens",
                        usage.get("total_tokens") if call["kind"] == "embedding" else None))
        out = 0 if call["kind"] == "embedding" else usage.get("output_tokens", usage.get("completion_tokens"))
        if not all(type(v) is int and v >= 0 for v in (inp, out)):
            call.update(status="usage_unknown")
            self.data["halt_reason"] = "request_usage_unknown"
        else:
            price = self.prices[call["kind"]]
            cost = (decimal(inp) * decimal(price["input"]) + decimal(out) * decimal(price["output"])) / 1_000_000
            call.update(status="settled", input_tokens=inp, output_tokens=out, actual_cost=str(cost))
            if inp > call["input_bound"] or out > call["output_bound"]:
                self.data["halt_reason"] = "usage_exceeded_conservative_bound"
        call["finished_at"] = datetime.now(timezone.utc).isoformat()
        self.save()


class BudgetProvider(ModelProvider):
    def __init__(self, settings, ledger, embedding_cache):
        if settings.provider_max_retries:
            raise ValueError("Paid evaluation requires retries disabled")
        super().__init__(settings)
        self.ledger, self.embedding_cache = ledger, embedding_cache

    def _post(self, base, key, endpoint, payload):
        kind = "embedding" if endpoint == "embeddings" else "chat"
        output = payload.get("max_output_tokens", payload.get("max_tokens", 0))
        if kind == "chat" and (type(output) is not int or output <= 0):
            raise ValueError("Unbounded charged output is forbidden")
        bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024
        ticket = self.ledger.reserve(kind, base, payload.get("model"), bound, output)
        try:
            response = super()._post(base, key, endpoint, payload)
        except BaseException:
            self.ledger.settle(ticket)
            raise
        self.ledger.settle(ticket, response)
        # Allow this response to be recorded even if its usage is unknown;
        # every subsequent charged request will be blocked by the ledger.
        return response

    def embed(self, texts, expected_dim=0):
        self.remaining()
        missing = list(dict.fromkeys(t for t in texts if t not in self.embedding_cache))
        if missing:
            vectors = super().embed(missing, expected_dim)
            self.embedding_cache.update(zip(missing, vectors))
        else:
            self.last_embedding_usage = {"usage_known": True, "input_tokens": 0, "output_tokens": 0,
                                         "cache_hit": True}
        vectors = [self.embedding_cache[t] for t in texts]
        if expected_dim and any(len(v) != expected_dim for v in vectors):
            raise ValueError("Shared embedding cache dimension mismatch")
        return vectors
