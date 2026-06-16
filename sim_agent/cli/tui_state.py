from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR: Final = SOURCE_ROOT / "evidence" / "asa-tui"
SESSION_LEDGER_NAME: Final = "asa_session.json"
SESSION_EVENTS_NAME: Final = "asa_session_events.jsonl"
SESSION_DIR_ENV: Final = "ASA_SESSION_DIR"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    provider: str = "openclaw"
    name: str = "gpt-5.5"
    base_url: str = "https://openclaw.local/v1"
    auth_mode: str = "oauth"
    api_key_env: str = "OPENCLAW_OAUTH_TOKEN"


@dataclass(frozen=True, slots=True)
class TuiState:
    session_id: str
    session_dir: Path
    model: ModelSettings
    last_run_ledger: Path | None = None
    team_ledger: Path | None = None
    runtime_ledger: Path | None = None


@dataclass(frozen=True, slots=True)
class TuiStep:
    state: TuiState
    exit_requested: bool


def initial_state(session_dir: Path | None = None) -> TuiState:
    env_dir = os.environ.get(SESSION_DIR_ENV)
    resolved = session_dir or (Path(env_dir) if env_dir else DEFAULT_OUTPUT_DIR)
    endpoint = load_runtime_config().model_endpoint
    state = TuiState(
        session_id=f"asa-{int(time.time())}",
        session_dir=resolved,
        model=ModelSettings(
            provider=endpoint.provider,
            name=endpoint.model,
            base_url=endpoint.base_url,
            auth_mode=endpoint.auth_mode,
            api_key_env=endpoint.api_key_env,
        ),
    )
    persist_state(state)
    append_event(state, "session_created", "Interactive ASA session opened")
    return state


def replace_model(state: TuiState, model: ModelSettings) -> TuiState:
    next_state = TuiState(
        session_id=state.session_id,
        session_dir=state.session_dir,
        model=model,
        last_run_ledger=state.last_run_ledger,
        team_ledger=state.team_ledger,
        runtime_ledger=state.runtime_ledger,
    )
    persist_state(next_state)
    return next_state


def replace_run_ledger(state: TuiState, ledger: Path) -> TuiState:
    next_state = TuiState(
        session_id=state.session_id,
        session_dir=state.session_dir,
        model=state.model,
        last_run_ledger=ledger,
        team_ledger=state.team_ledger,
        runtime_ledger=state.runtime_ledger,
    )
    persist_state(next_state)
    return next_state


def replace_team_ledger(state: TuiState, ledger: Path) -> TuiState:
    next_state = TuiState(
        session_id=state.session_id,
        session_dir=state.session_dir,
        model=state.model,
        last_run_ledger=state.last_run_ledger,
        team_ledger=ledger,
        runtime_ledger=state.runtime_ledger,
    )
    persist_state(next_state)
    return next_state


def replace_runtime_ledger(state: TuiState, ledger: Path) -> TuiState:
    next_state = TuiState(
        session_id=state.session_id,
        session_dir=state.session_dir,
        model=state.model,
        last_run_ledger=state.last_run_ledger,
        team_ledger=state.team_ledger,
        runtime_ledger=ledger,
    )
    persist_state(next_state)
    return next_state


def persist_state(state: TuiState) -> Path:
    state.session_dir.mkdir(parents=True, exist_ok=True)
    path = state.session_dir / SESSION_LEDGER_NAME
    path.write_text(json.dumps(_state_payload(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def append_event(state: TuiState, event_type: str, summary: str) -> Path:
    state.session_dir.mkdir(parents=True, exist_ok=True)
    path = state.session_dir / SESSION_EVENTS_NAME
    payload: JsonMap = {
        "at": time.time(),
        "session_id": state.session_id,
        "event_type": event_type,
        "summary": summary,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def recent_events(state: TuiState, limit: int) -> tuple[JsonMap, ...]:
    path = state.session_dir / SESSION_EVENTS_NAME
    if not path.is_file():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(json.loads(line) for line in lines[-limit:])


def _state_payload(state: TuiState) -> JsonMap:
    return {
        "session_id": state.session_id,
        "session_dir": str(state.session_dir),
        "model": asdict(state.model),
        "last_run_ledger": str(state.last_run_ledger) if state.last_run_ledger else "",
        "team_ledger": str(state.team_ledger) if state.team_ledger else "",
        "runtime_ledger": str(state.runtime_ledger) if state.runtime_ledger else "",
    }
