"""Verify real Redis admission across independent processes without model requests."""
import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import Settings  # noqa: E402
from app.errors import ProviderError  # noqa: E402
from app.provider_quota import client_for, reserve  # noqa: E402


def attempt(arguments):
    namespace, url = arguments
    settings = Settings(_env_file=None, provider_quota_enabled=True, provider_quota_namespace=namespace,
                        provider_quota_redis_url=url, provider_quota_rpm=3, provider_quota_tpm=300)
    try:
        reserve(settings, "https://quota-control.invalid", "not-a-real-key", 100)
        return "admitted"
    except ProviderError as exc:
        return exc.code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://127.0.0.1:56379/0")
    args = parser.parse_args()
    namespace = "quota_check_" + uuid4().hex[:12]
    client = client_for(args.redis_url)
    client.ping()
    report = {"scope": "real Redis, separate Windows spawn processes; no model HTTP requests"}
    try:
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(attempt, [(namespace, args.redis_url)] * 12))
        report.update(attempts=len(results), admitted=results.count("admitted"),
                      denied=results.count("provider_quota_exceeded"), unexpected=[r for r in results if r not in {
                          "admitted", "provider_quota_exceeded"}])
        assert report["admitted"] == 3 and report["denied"] == 9 and not report["unexpected"]
        # Expired reservations must be cleaned atomically by Redis server time.
        keys = list(client.scan_iter(match=namespace + ":*"))
        zkey = next(k for k in keys if k.endswith(":requests"))
        for member in client.zrange(zkey, 0, -1):
            client.zadd(zkey, {member: 0})
        assert attempt((namespace, args.redis_url)) == "admitted"
        report["expired_window_released"] = True
    finally:
        keys = list(client.scan_iter(match=namespace + ":*"))
        if keys:
            client.delete(*keys)
        report["remaining_test_keys"] = len(list(client.scan_iter(match=namespace + ":*")))
    path = ROOT / "docs" / "provider-quota-live-check.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
