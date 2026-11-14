"""Redact known credentials and common credential forms at model/log boundaries."""
import os
import re


def redact(value, settings):
    if isinstance(value, dict):
        return {k: redact(v, settings) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, settings) for v in value]
    if not isinstance(value, str):
        return value
    route_secrets = [os.environ.get(route.api_key_env, "") for route in settings.role_model_routes.values()
                     if route.api_key_env]
    for secret in (settings.chat_api_key, settings.embedding_api_key, settings.milvus_token, *route_secrets):
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED]", value)
    return re.sub(r"(?i)(\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;\"']+",
                  r"\1[REDACTED]", value)
