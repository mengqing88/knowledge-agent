"""Process-local circuits shared across provider instances; not a distributed service limiter."""
import threading
import time

from .errors import ProviderError


class Circuits:
    def __init__(self):
        self.lock = threading.Lock()
        self.states = {}

    def enter(self, key, cooldown):
        with self.lock:
            state = self.states.setdefault(key, {"failures": 0, "opened": None, "probe": False})
            if state["opened"] is not None:
                if time.monotonic() - state["opened"] < cooldown or state["probe"]:
                    raise ProviderError("模型服务暂时熔断，请稍后重试", "circuit_open", True)
                state["probe"] = True

    def finish(self, key, failed, threshold, healthy=False):
        with self.lock:
            state = self.states[key]
            state["probe"] = False
            if failed:
                state["failures"] += 1
                if state["failures"] >= threshold:
                    state["opened"] = time.monotonic()
            elif healthy:
                state.update(failures=0, opened=None)


circuits = Circuits()
