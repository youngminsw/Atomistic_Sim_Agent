from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TextIO

from sim_agent.llm_endpoints import (
    ModelCatalogEntry,
    ModelPolicyError,
    ModelUseCase,
    ProviderConfigPolicyError,
    find_model_catalog_entry,
    list_model_catalog,
    model_catalog_references,
)
from sim_agent.llm_endpoints.model_profiles import list_model_profiles
from sim_agent.runtime_config import load_runtime_config, mark_active_profile_customized, save_runtime_config

from .tui_model_endpoint import (
    endpoint_from_command,
    endpoint_from_model_settings,
    model_settings_from_endpoint,
    validated_endpoint,
    write_endpoint_config,
)
from .tui_model_profiles import save_model_profile
from .tui_parse import parse_options
from .tui_select import MenuOption, choose_option
from .tui_state import TuiState, append_event, replace_model
from .tui_thinking import choose_thinking_level


def write_model_catalog(args: Sequence[str], output_stream: TextIO) -> None:
    parsed = parse_options(args)
    entries = list_model_catalog(parsed.options.get("provider"))
    if output_stream.isatty():
        _write_human_model_catalog(entries, output_stream)
        return
    output_stream.write("model_catalog=true\n")
    current_company = ""
    current_provider = ""
    for entry in entries:
        if entry.company != current_company:
            current_company = entry.company
            output_stream.write(f"model_group={current_company}\n")
            current_provider = ""
        if entry.source_provider != current_provider:
            current_provider = entry.source_provider
            output_stream.write(f"model_provider_group={entry.source_provider}\n")
        output_stream.write(
            f"source_provider={entry.source_provider} model={entry.model} "
            f"provider={entry.provider}\n"
        )
        output_stream.write(
            f"model_runtime reasoning_effort={entry.reasoning_effort} "
            f"role_hint={entry.role_hint}\n"
        )
        output_stream.write(
            f"model_endpoint auth_mode={entry.auth_mode} "
            f"base_url={entry.base_url} api_key_env={entry.api_key_env}\n"
        )


def _write_human_model_catalog(entries: Sequence[ModelCatalogEntry], output_stream: TextIO) -> None:
    output_stream.write("\nModel Catalog\n")
    current_company = ""
    current_provider = ""
    for entry in entries:
        if entry.company != current_company:
            current_company = entry.company
            output_stream.write(f"\n{current_company}\n")
            current_provider = ""
        if entry.source_provider != current_provider:
            current_provider = entry.source_provider
            output_stream.write(f"  {entry.source_provider}\n")
        output_stream.write(f"    {entry.model:<28} thinking={entry.reasoning_effort:<6} {entry.role_hint}\n")
    output_stream.write("\nUse /model set to choose, or /model use <provider/model> --thinking-level high.\n")


def choose_model(state: TuiState, output_stream: TextIO, input_stream: TextIO | None) -> TuiState:
    if input_stream is None or not input_stream.isatty():
        output_stream.write("model_error=model_reference_required\n")
        output_stream.write("model_hint=/model set <provider/model> or /model list\n")
        output_stream.write(f"model_catalog_refs={model_catalog_references()}\n")
        return state
    profile_options = tuple(
        MenuOption(
            value=f"profile:{profile.name}",
            label=f"Profile / {profile.label}",
            summary=f"{profile.default.reference} · {profile.default.reasoning_effort} · {profile.summary}",
        )
        for profile in list_model_profiles()
    )
    model_options = tuple(
        MenuOption(
            value=entry.reference,
            label=f"{entry.company} / {entry.source_provider}",
            summary=f"{entry.model} · {entry.reasoning_effort} · {entry.role_hint}",
        )
        for entry in list_model_catalog()
    )
    options = (*profile_options, *model_options)
    selected = choose_option("Model Selection", options, input_stream, output_stream)
    if selected is None:
        output_stream.write("model_selection_cancelled=true\n")
        return state
    if selected.startswith("profile:"):
        return save_model_profile((selected.removeprefix("profile:"),), state, output_stream)
    selected_entry = find_model_catalog_entry(selected)
    default_level = selected_entry.reasoning_effort if selected_entry is not None else "high"
    thinking = choose_thinking_level("Model Thinking Level", default_level, input_stream, output_stream)
    if thinking is None:
        output_stream.write("model_thinking_selection_cancelled=true\n")
        return state
    return use_model((selected, "--thinking-level", thinking), state, output_stream)


def use_model(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    endpoint = endpoint_from_command(
        parsed.remainder,
        parsed.options,
        endpoint_from_model_settings(state.model),
    )
    if not endpoint.model:
        output_stream.write("model_error=model_reference_required\n")
        output_stream.write(f"model_catalog_refs={model_catalog_references()}\n")
        return state
    try:
        normalized = validated_endpoint(endpoint, ModelUseCase.PRIMARY_CONTROL)
    except (ModelPolicyError, ProviderConfigPolicyError) as exc:
        output_stream.write(f"model_error={exc}\n")
        append_event(state, "model_use_blocked", str(exc))
        return state

    config = load_runtime_config()
    path = save_runtime_config(mark_active_profile_customized(replace(config, model_endpoint=normalized)))
    next_state = replace_model(state, model_settings_from_endpoint(normalized))
    append_event(next_state, "model_saved", f"{normalized.provider}/{normalized.model}")
    output_stream.write("model_saved=true\n")
    write_endpoint_config(normalized, output_stream)
    output_stream.write(f"runtime_config_path={path}\n")
    return next_state
