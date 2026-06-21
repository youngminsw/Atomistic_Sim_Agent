from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from sim_agent.agents_sdk_runtime import run_workflow_harness_smoke, workflow_harness_catalog

from .tui_parse import parse_options
from .tui_state import TuiState, append_event


WORKFLOW_ALIASES: tuple[str, ...] = ("deep-interview", "ralplan", "ultrawork", "ultraqa", "ultragoal")


def handle_workflow(args: Sequence[str], state: TuiState, output_stream: TextIO) -> TuiState:
    parsed = parse_options(args)
    workflow_id = _workflow_id(parsed.remainder)
    output_dir = Path(parsed.options.get("output_dir", str(state.session_dir / "workflows")))
    result = run_workflow_harness_smoke(
        workflow_id,
        {"request_id": state.session_id, "user_goal": "Interactive ASA workflow harness"},
        output_dir,
    )
    append_event(state, "workflow_harness", f"{result.workflow_id}:{result.status}")
    output_stream.write("workflow_harness_ready=true\n")
    output_stream.write(f"workflow={result.workflow_id}\n")
    output_stream.write(f"workflow_status={result.status}\n")
    output_stream.write(f"current_state={result.current_state}\n")
    output_stream.write(f"verification_gate={result.verification_gate}\n")
    output_stream.write(f"workflow_ledger_path={output_dir / result.ledger_ref}\n")
    for blocker in result.blockers:
        output_stream.write(f"workflow_blocker={blocker}\n")
    return state


def write_workflow_catalog(output_stream: TextIO) -> None:
    output_stream.write("workflow_catalog=true\n")
    for workflow in workflow_harness_catalog():
        output_stream.write(
            f"workflow={workflow.workflow_id} current_state={workflow.current_state} "
            f"verification_gate={workflow.verification_gate}\n"
        )


def _workflow_id(remainder: tuple[str, ...]) -> str:
    if remainder:
        return remainder[0]
    return "deep-interview"
