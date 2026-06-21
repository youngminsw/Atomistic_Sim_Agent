from __future__ import annotations

from typing import TextIO

from sim_agent.llm_endpoints import ModelProviderConfig, ModelUseCase, find_model_catalog_entry
from sim_agent.runtime_config import ModelEndpointRuntimeConfig

from .tui_state import ModelSettings


def reasoning_effort_from_options(
    options: dict[str, str],
    fallback: str,
    *,
    catalog_default: str | None = None,
) -> str:
    return options.get("thinking_level", options.get("reasoning_effort", catalog_default or fallback))


def endpoint_from_command(
    remainder: tuple[str, ...],
    options: dict[str, str],
    fallback: ModelEndpointRuntimeConfig,
) -> ModelEndpointRuntimeConfig:
    entry = find_model_catalog_entry(remainder[0]) if remainder else None
    if remainder and entry is None:
        provider, model = split_reference(remainder[0])
    elif entry is not None:
        provider, model = entry.provider, entry.model
    else:
        provider, model = fallback.provider, fallback.model
    return ModelEndpointRuntimeConfig(
        provider=options.get("provider", provider),
        model=options.get("model", model),
        reasoning_effort=reasoning_effort_from_options(
            options,
            fallback.reasoning_effort,
            catalog_default=entry.reasoning_effort if entry is not None else None,
        ),
        base_url=options.get("base_url", entry.base_url if entry is not None else fallback.base_url),
        auth_mode=options.get("auth_mode", entry.auth_mode if entry is not None else fallback.auth_mode),
        api_key_env=options.get("api_key_env", entry.api_key_env if entry is not None else fallback.api_key_env),
    )


def split_reference(reference: str) -> tuple[str, str]:
    if "/" not in reference:
        return reference, ""
    provider, model = reference.split("/", 1)
    return provider, model


def validated_endpoint(
    endpoint: ModelEndpointRuntimeConfig,
    use_case: ModelUseCase,
) -> ModelEndpointRuntimeConfig:
    normalized = ModelProviderConfig.from_mapping(
        {
            "provider": endpoint.provider,
            "model": endpoint.model,
            "reasoning_effort": endpoint.reasoning_effort,
            "base_url": endpoint.base_url,
            "auth_mode": endpoint.auth_mode,
            "api_key_env": endpoint.api_key_env,
            "use_case": use_case.value,
        }
    )
    return ModelEndpointRuntimeConfig(
        provider=normalized.provider,
        model=normalized.model,
        reasoning_effort=normalized.reasoning_effort,
        base_url=normalized.base_url,
        auth_mode=normalized.auth_mode,
        api_key_env=normalized.api_key_env,
    )


def endpoint_from_model_settings(model: ModelSettings) -> ModelEndpointRuntimeConfig:
    return ModelEndpointRuntimeConfig(
        provider=model.provider,
        model=model.name,
        reasoning_effort=model.reasoning_effort,
        base_url=model.base_url,
        auth_mode=model.auth_mode,
        api_key_env=model.api_key_env,
    )


def model_settings_from_endpoint(endpoint: ModelEndpointRuntimeConfig) -> ModelSettings:
    return ModelSettings(
        provider=endpoint.provider,
        name=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )


def write_endpoint_config(endpoint: ModelEndpointRuntimeConfig, output_stream: TextIO) -> None:
    output_stream.write(
        f"provider={endpoint.provider} model={endpoint.model} "
        f"reasoning_effort={endpoint.reasoning_effort}\n"
    )
    output_stream.write(f"base_url={endpoint.base_url} auth_mode={endpoint.auth_mode}\n")
    output_stream.write(f"api_key_env={endpoint.api_key_env}\n")
