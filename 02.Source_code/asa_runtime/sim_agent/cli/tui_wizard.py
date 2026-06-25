from __future__ import annotations

from collections.abc import Sequence
from typing import TextIO

from sim_agent.knowledge.memory_seed import MemorySeedError, build_memory_seed_bundle, read_memory_seed_sources_from_neo4j
from sim_agent.provider_registry import OPENAI_CODEX_BASE_URL, OPENAI_CODEX_TOKEN_ENV
from sim_agent.runtime_config import load_runtime_config

from .tui_login import TerminalLoginSelector, handle_login
from .tui_select import MenuOption, choose_option, prompt_visible
from .tui_setup import handle_setup
from .tui_state import TuiState, default_output_dir
from .tui_thinking import choose_thinking_level
from .tui_wizard_graphdb import graphdb_wizard
from .tui_wizard_run import interview_run_wizard


def handle_wizard(
    args: Sequence[str],
    state: TuiState,
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    interactive: bool,
) -> TuiState:
    if not interactive:
        _write_wizard_help(output_stream)
        return state
    if args:
        match args[0]:  # noqa: MATCH_OK - direct wizard targets are open command strings.
            case "endpoint":
                return _endpoint_wizard(state, input_stream, output_stream)
            case "graphdb":
                return graphdb_wizard(state, input_stream, output_stream)
            case "login":
                return handle_login((), state, output_stream, TerminalLoginSelector(input_stream, output_stream))
            case "interview_run":
                return interview_run_wizard(state, input_stream, output_stream)
            case _:
                pass
    choice = choose_option(
        "ASA Wizard",
        (
            MenuOption("endpoint", "Model endpoint", "provider/model/auth setup"),
            MenuOption("login", "OAuth/API login", "store token without echoing it"),
            MenuOption("graphdb", "GraphDB brain", "Neo4j URI/env/database setup"),
            MenuOption("memory_seed", "Memory seed", "build seed bundle from personal memory DB"),
            MenuOption("interview_run", "Interview → run", "ask key simulation questions then prepare run"),
            MenuOption("cancel", "Cancel", "return to shell"),
        ),
        input_stream,
        output_stream,
    )
    match choice:  # noqa: MATCH_OK - menu values are open strings from terminal input.
        case "endpoint":
            return _endpoint_wizard(state, input_stream, output_stream)
        case "login":
            return handle_login(args, state, output_stream, TerminalLoginSelector(input_stream, output_stream))
        case "graphdb":
            return graphdb_wizard(state, input_stream, output_stream)
        case "memory_seed":
            _memory_seed_wizard(output_stream)
            return state
        case "interview_run":
            return interview_run_wizard(state, input_stream, output_stream)
        case "cancel" | None:
            output_stream.write("wizard_cancelled=true\n")
            return state
        case unreachable:
            output_stream.write(f"wizard_error=unexpected_choice:{unreachable}\n")
            return state


def _endpoint_wizard(state: TuiState, input_stream: TextIO, output_stream: TextIO) -> TuiState:
    preset = choose_option(
        "Model Endpoint",
        (
            MenuOption("codex", "OpenAI Codex subscription", "browser OAuth provider endpoint"),
            MenuOption("openai", "OpenAI API key", "direct API key env"),
            MenuOption("anthropic", "Anthropic API key", "direct Claude API key env"),
            MenuOption("local", "Local gateway", "no-auth smoke gateway"),
            MenuOption("custom", "Custom", "type fields"),
        ),
        input_stream,
        output_stream,
    )
    if preset is None:
        output_stream.write("wizard_cancelled=true\n")
        return state
    if preset == "custom":
        provider = prompt_visible("Provider", state.model.provider, input_stream, output_stream)
        model = prompt_visible("Model", state.model.name, input_stream, output_stream)
        base_url = prompt_visible("Base URL", state.model.base_url, input_stream, output_stream)
        auth_mode = prompt_visible("Auth mode", state.model.auth_mode, input_stream, output_stream)
        api_key_env = prompt_visible("Token env", state.model.api_key_env, input_stream, output_stream)
    else:
        provider, model, base_url, auth_mode, api_key_env = _endpoint_preset(preset)
    thinking_level = choose_thinking_level("Model Thinking Level", "high", input_stream, output_stream) or "high"
    return handle_setup(
        (
            "endpoint",
            "--provider",
            provider,
            "--model",
            model,
            "--thinking-level",
            thinking_level,
            "--base-url",
            base_url,
            "--auth-mode",
            auth_mode,
            "--api-key-env",
            api_key_env,
        ),
        state,
        output_stream,
    )


def _memory_seed_wizard(output_stream: TextIO) -> None:
    config = load_runtime_config()
    output_dir = default_output_dir() / "graphdb-memory-seed"
    try:
        sources = read_memory_seed_sources_from_neo4j()
        bundle = build_memory_seed_bundle(
            output_dir,
            database_name=config.graphdb.database,
            sync_run_id="tui-personal-memory-seed",
            memory_sources=sources,
        )
    except MemorySeedError as exc:
        output_stream.write("memory_seed_status=blocked\n")
        output_stream.write(f"memory_seed_blocker={exc}\n")
        return
    output_stream.write("memory_seed_status=ready\n")
    output_stream.write(f"memory_seed_source_count={len(sources)}\n")
    output_stream.write(f"memory_seed_bundle_dir={bundle.output_dir}\n")
    output_stream.write(f"memory_seed_ingest_report={bundle.ingest_report_path}\n")
    output_stream.write("memory_seed_write=false\n")


def _endpoint_preset(preset: str) -> tuple[str, str, str, str, str]:
    match preset:  # noqa: MATCH_OK - endpoint preset input has a safe fallback.
        case "codex":
            return ("openai-codex", "gpt-5-codex", OPENAI_CODEX_BASE_URL, "oauth", OPENAI_CODEX_TOKEN_ENV)
        case "openai":
            return ("openai", "gpt-5.5", "https://api.openai.com/v1", "api_key", "OPENAI_API_KEY")
        case "anthropic":
            return ("anthropic", "claude-sonnet-4.5", "https://api.anthropic.com/v1", "api_key", "ANTHROPIC_API_KEY")
        case "local":
            return ("local_gateway", "gpt-5.5", "http://localhost:8787/v1", "none", "RUNTIME_GATEWAY_TOKEN")
        case _:
            return ("local_gateway", "gpt-5.5", "http://localhost:8787/v1", "none", "RUNTIME_GATEWAY_TOKEN")


def _write_wizard_help(output_stream: TextIO) -> None:
    output_stream.write("wizard_available=true\n")
    output_stream.write("wizard_requires_interactive_tty=true\n")
    output_stream.write("wizard_option=endpoint\n")
    output_stream.write("wizard_option=login\n")
    output_stream.write("wizard_option=graphdb\n")
    output_stream.write("wizard_option=memory_seed\n")
    output_stream.write("wizard_option=interview_run\n")
