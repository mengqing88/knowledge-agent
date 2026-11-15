import re

from app.core.security import redact

SENSITIVE = re.compile(r"(?i)(?:password|secret|api[_-]?key|access[_-]?token|authorization)")


def protect(value, settings, secrets=()):
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if SENSITIVE.search(str(k)) else protect(v, settings, secrets)
                for k, v in value.items()}
    if isinstance(value, list):
        return [protect(v, settings, secrets) for v in value]
    if isinstance(value, str):
        value = redact(value, settings)
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value
