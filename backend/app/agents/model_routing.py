"""Explicit role routes; credentials are environment references, never frozen secrets."""
import os
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import ProviderError

ROLES = {"Planner", "Reranker", "Analyst", "Verifier", "ToolRouter", "EvidenceGate"}


class ModelRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=200)
    base_url: str | None = None
    api_key_env: str | None = Field(None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    api_style: Literal["responses", "chat_completions"] | None = None
    input_price_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)
    output_price_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def endpoint(self):
        if self.base_url:
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username or
                    url.password or url.query or url.fragment):
                raise ValueError("角色模型地址不合法")
        return self


@dataclass(frozen=True)
class ResolvedRoute:
    model: str
    base_url: str
    api_key: str
    api_style: str


def resolve_route(settings, role):
    route = settings.role_model_routes.get(role)
    if route is None:
        return ResolvedRoute(settings.model, settings.chat_base_url, settings.chat_api_key, settings.chat_api_style)
    standard_keys = {"CHAT_API_KEY": settings.chat_api_key, "EMBEDDING_API_KEY": settings.embedding_api_key}
    key = os.environ.get(route.api_key_env, standard_keys.get(route.api_key_env, "")) if route.api_key_env else settings.chat_api_key
    if route.api_key_env and not key:
        raise ProviderError("角色模型凭据环境变量未配置", "provider_config")
    return ResolvedRoute(route.model, route.base_url or settings.chat_base_url,
                         key, route.api_style or settings.chat_api_style)


def route_manifest(settings):
    return {role: route.model_dump() for role, route in sorted(settings.role_model_routes.items())}


def role_model(config, role):
    return config.get("role_model_routes", {}).get(role, {}).get("model", config["model"])


def role_rates(config, role):
    if role == "Embedding":
        return config.get("embedding_price_per_million"), 0.
    route = config.get("role_model_routes", {}).get(role)
    if route:
        # Different aliases/endpoints cannot inherit an unrelated model's prices.
        return route.get("input_price_per_million"), route.get("output_price_per_million")
    return config.get("chat_input_price_per_million"), config.get("chat_output_price_per_million")
