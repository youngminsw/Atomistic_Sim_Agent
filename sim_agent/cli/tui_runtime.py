from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run, write_agents_sdk_runtime_ledger
from sim_agent.llm_endpoints import ModelProviderConfig

from .tui_parse import parse_options
from .tui_state import TuiState, append_event, replace_runtime_ledger


def handle_runtime(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": state.model.provider,
            "model": state.model.name,
            "reasoning_effort": "high",
            "base_url": state.model.base_url,
            "auth_mode": state.model.auth_mode,
            "api_key_env": state.model.api_key_env,
        }
    )
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "runtime")))
    result = run_agents_sdk_runtime_dry_run(
        {
            "request_id": state.session_id,
            "user_goal": "Interactive ASA runtime session",
            "host": parsed.options.get("host", "local"),
            "estimated_runtime_s": _runtime_seconds(parsed.options.get("estimated_runtime_s")),
            "graphdb": {"mode": parsed.options.get("graphdb_mode", "dry_run")},
        },
        endpoint,
        run_sdk_smoke="smoke" in parsed.flags,
        output_dir=output_dir,
    )
    ledger = write_agents_sdk_runtime_ledger(output_dir, result)
    next_state = replace_runtime_ledger(state, ledger)
    append_event(next_state, "runtime_dry_run", result.run_id)
    output_stream.write("runtime_dry_run=true\n")
    output_stream.write(f"runtime_run_id={result.run_id}\n")
    output_stream.write(f"sdk_available={str(result.sdk_available).lower()}\n")
    output_stream.write(f"sdk_run_completed={str(result.sdk_run_completed).lower()}\n")
    for role_id in result.handoff_sequence:
        output_stream.write(f"handoff={role_id}\n")
    output_stream.write(f"runtime_ledger_path={ledger}\n")
    return next_state


def _runtime_seconds(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0
