"""Atomic, conservative 60-second RPM/estimated-TPM reservations in Redis."""
import hashlib
from functools import lru_cache
from uuid import uuid4

import redis

from app.core.errors import ProviderError

# Redis TIME is shared by workers; admission and expiry are one atomic script.
# Reservations are not refunded: a lost response may still have consumed tokens.
ADMIT = """
local stamp = redis.call('TIME')
local now = tonumber(stamp[1]) * 1000 + math.floor(tonumber(stamp[2]) / 1000)
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', now - 60000)
for _, id in ipairs(expired) do redis.call('HDEL', KEYS[2], id) end
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - 60000)
local count = redis.call('ZCARD', KEYS[1])
local used = 0
for _, value in ipairs(redis.call('HVALS', KEYS[2])) do used = used + tonumber(value) end
if count + 1 > tonumber(ARGV[1]) or used + tonumber(ARGV[3]) > tonumber(ARGV[2]) then
  return {0, count, used}
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('HSET', KEYS[2], ARGV[4], ARGV[3])
redis.call('PEXPIRE', KEYS[1], 61000)
redis.call('PEXPIRE', KEYS[2], 61000)
return {1, count + 1, used + tonumber(ARGV[3])}
"""


@lru_cache(maxsize=8)
def client_for(url):
    return redis.Redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)


def reserve(settings, base_url, api_key, estimated_tokens):
    if not settings.provider_quota_enabled:
        return None
    if estimated_tokens > settings.provider_quota_tpm:
        raise ProviderError("请求估算 Token 超过共享配额上限，未发送请求", "provider_quota_request")
    # Same account and endpoint share a pool across roles/models/chat/embedding.
    identity = hashlib.sha256(f"{base_url.rstrip('/')}:{api_key}".encode()).hexdigest()
    prefix = f"{settings.provider_quota_namespace}:{{{identity}}}"
    try:
        admitted, requests, tokens = client_for(settings.provider_quota_redis_url).eval(
            ADMIT, 2, f"{prefix}:requests", f"{prefix}:tokens",
            settings.provider_quota_rpm, settings.provider_quota_tpm, estimated_tokens, uuid4().hex)
    except redis.RedisError:
        raise ProviderError("共享配额服务不可用，未发送模型请求", "provider_quota_unavailable") from None
    if not admitted:
        raise ProviderError("模型共享配额已满，请稍后重试", "provider_quota_exceeded", True)
    return {"pool": identity, "requests_in_window": requests, "reserved_tokens_in_window": tokens}
