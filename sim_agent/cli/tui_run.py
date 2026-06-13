from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.cli.orchestrator import (
    OrchestratorChatConfig,
    OrchestratorChatError,
    prepare_orchestrator_chat,
)

from .tui_parse import parse_options
from .tui_state import DEFAULT_OUTPUT_DIR, SOURCE_ROOT, ModelSettings, TuiState, append_event, replace_run_ledger


def handle_run(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    goal = " ".join(parsed.remainder).strip()
    if not goal:
        output_stream.write("run_error=missing_goal\n")
        return state
    config = _chat_config(goal, parsed.options, state.model)
    try:
        report = prepare_orchestrator_chat(config)
    except OrchestratorChatError as exc:
        output_stream.write(f"run_error={exc}\n")
        append_event(state, "run_failed", str(exc))
        return state
    next_state = replace_run_ledger(state, report.ledger_path)
    append_event(next_state, "run_prepared", report.run_id)
    output_stream.write("run_prepared=true\n")
    output_stream.write(f"run_id={report.run_id}\n")
    output_stream.write(f"artifact_dir={report.artifact_dir}\n")
    output_stream.write(f"agent_run_ledger_path={report.ledger_path}\n")
    return next_state


def _chat_config(goal: str, options: Mapping[str, str], model: ModelSettings) -> OrchestratorChatConfig:
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
        host=options.get("host", "gpu-5090"),
        environment_name=options.get("environment_name", "atomistic-sim-gpu"),
        model_provider=model.provider,
        model_name=model.name,
        model_base_url=model.base_url,
        model_auth_mode=model.auth_mode,
        model_api_key_env=model.api_key_env,
    )
