from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime import AGENT_TEAM_SESSION_LEDGER_NAME, run_agent_team_session_runtime
from sim_agent.cli.orchestrator import (
    OrchestratorChatConfig,
    OrchestratorChatError,
    prepare_orchestrator_chat,
)
from sim_agent.compute import default_compute_resource
from sim_agent.llm_endpoints import ModelPolicyError, ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.runtime_config import load_runtime_config

from .tui_parse import parse_options
from .tui_state import (
    DEFAULT_OUTPUT_DIR,
    SOURCE_ROOT,
    ModelSettings,
    TuiState,
    append_event,
    replace_run_ledger,
    replace_team_ledger,
)


def handle_run(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    goal = " ".join(parsed.remainder).strip()
    if not goal:
        output_stream.write("run_error=missing_goal\n")
        return state
    config = _chat_config(goal, parsed.options, state.model)
    runtime_config = load_runtime_config()
    next_state = state
    if runtime_config.team_mode_default and "no_team" not in parsed.flags:
        next_state = _run_team_default(config, state, output_stream)
    try:
        report = prepare_orchestrator_chat(config)
    except OrchestratorChatError as exc:
        output_stream.write(f"run_error={exc}\n")
        append_event(state, "run_failed", str(exc))
        return state
    next_state = replace_run_ledger(next_state, report.ledger_path)
    append_event(next_state, "run_prepared", report.run_id)
    output_stream.write("run_prepared=true\n")
    output_stream.write(f"run_id={report.run_id}\n")
    output_stream.write(f"artifact_dir={report.artifact_dir}\n")
    output_stream.write(f"agent_run_ledger_path={report.ledger_path}\n")
    return next_state


def _chat_config(goal: str, options: Mapping[str, str], model: ModelSettings) -> OrchestratorChatConfig:
    compute = default_compute_resource()
    return OrchestratorChatConfig(
        message=goal,
        output_dir=Path(options.get("output_dir", str(DEFAULT_OUTPUT_DIR))),
        source_root=Path(options.get("source_root", str(SOURCE_ROOT))),
        material=options.get("material", "Si"),
        phase=options.get("phase", "amorphous"),
        ion=options.get("ion", "Ar"),
        feature_type=options.get("feature_type", "hole"),
        mode=options.get("mode", "3d"),
        energy_range_ev=options.get("energy_range_ev", "30:150"),
        polar_range_deg=options.get("polar_range_deg", "0:55"),
        azimuth_range_deg=options.get("azimuth_range_deg", "0:360"),
        host=options.get("host", compute.host_alias),
        environment_name=options.get("environment_name", compute.environment_name),
        model_provider=model.provider,
        model_name=model.name,
        model_base_url=model.base_url,
        reasoning_effort=model.reasoning_effort,
        model_auth_mode=model.auth_mode,
        model_api_key_env=model.api_key_env,
    )


def _run_team_default(config: OrchestratorChatConfig, state: TuiState, output_stream: TextIO) -> TuiState:
    try:
        endpoint = ModelProviderConfig.from_mapping(
            {
                "provider": config.model_provider,
                "model": config.model_name,
                "reasoning_effort": config.reasoning_effort,
                "base_url": config.model_base_url,
                "auth_mode": config.model_auth_mode,
                "api_key_env": config.model_api_key_env,
            }
        )
    except (ModelPolicyError, ProviderConfigPolicyError) as exc:
        output_stream.write(f"team_error={exc}\n")
        append_event(state, "team_blocked", str(exc))
        return state
    output_dir = config.output_dir / "team"
    result = run_agent_team_session_runtime(
        {
            "request_id": state.session_id,
            "user_goal": config.message,
            "host": config.host,
            "material": config.material,
            "phase": config.phase,
            "ion": config.ion,
        },
        endpoint,
        output_dir,
    )
    ledger = output_dir / AGENT_TEAM_SESSION_LEDGER_NAME
    next_state = replace_team_ledger(state, ledger)
    append_event(next_state, "team_runtime_primary", result.session_id)
    output_stream.write("team_runtime_primary=true\n")
    output_stream.write(f"team_session_id={result.session_id}\n")
    output_stream.write(f"team_ledger_path={ledger}\n")
    return next_state
