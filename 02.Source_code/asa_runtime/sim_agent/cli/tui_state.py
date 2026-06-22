from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from sim_agent.agent_runtime import (
    GlobalSessionModel,
    GlobalSessionOpenRequest,
    append_global_session_event,
    open_global_session,
)
from sim_agent.runtime_config import load_runtime_config
from sim_agent.schemas._parse import JsonMap


SOURCE_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR: Final = SOURCE_ROOT / "evidence" / "asa-tui"
SESSION_LEDGER_NAME: Final = "asa_session.json"
SESSION_EVENTS_NAME: Final = "asa_session_events.jsonl"
SESSION_DIR_ENV: Final = "ASA_SESSION_DIR"


@dataclass(frozen=True, slots=True)
class ModelSettings:
    provider: str = "openai-codex"
    name: str = "gpt-5-codex"
    reasoning_effort: str = "high"
    base_url: str = "https://model-gateway.local/v1"
    auth_mode: str = "gateway"
    api_key_env: str = "MODEL_GATEWAY_TOKEN"


@dataclass(frozen=True, slots=True)
class TuiState:
    session_id: str
    session_dir: Path
    model: ModelSettings
    last_run_ledger: Path | None = None
    team_ledger: Path | None = None
    runtime_ledger: Path | None = None
    global_session_id: str = ""
    global_session_path: Path | None = None


@dataclass(frozen=True, slots=True)
class TuiStep:
    state: TuiState
    exit_requested: bool


def initial_state(session_dir: Path | None = None, *, resume: str | None = None) -> TuiState:
    env_dir = os.environ.get(SESSION_DIR_ENV)
    resolved = session_dir or (Path(env_dir) if env_dir else DEFAULT_OUTPUT_DIR)
    endpoint = load_runtime_config().model_endpoint
    model = ModelSettings(
        provider=endpoint.provider,
        name=endpoint.model,
        reasoning_effort=endpoint.reasoning_effort,
        base_url=endpoint.base_url,
        auth_mode=endpoint.auth_mode,
        api_key_env=endpoint.api_key_env,
    )
    session_result = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=resolved if session_dir or env_dir else None,
            default_root=DEFAULT_OUTPUT_DIR,
            model=_global_model(model),
            resume=resume,
        )
    )
    state = TuiState(
        session_id=session_result.record.session_id,
        session_dir=session_result.record.session_dir,
        model=model,
        global_session_id=session_result.record.session_id,
        global_session_path=session_result.record.paths.global_session,
    )
    persist_state(state)
    match session_result.opened_as:
        case "created":
            append_event(state, "session_created", "Interactive ASA session opened")
        case "resumed":
            append_event(state, "session_resumed", "Interactive ASA session resumed")
        case "migrated":
            append_event(state, "session_resumed", "Legacy ASA session migrated and resumed")
    return state


def resume_state(state: TuiState, target: str = "latest") -> TuiState:
    target_value = target.strip() or "latest"
    model = _model_settings_from_global(state.model)
    requested_dir: Path | None = None
    resume = target_value
    if target_value in {"latest", ""}:
        requested_dir = state.session_dir
        resume = "latest"
    elif _looks_like_path(target_value):
        requested_dir = Path(target_value).expanduser()
        resume = "latest"
    session_result = open_global_session(
        GlobalSessionOpenRequest(
            requested_dir=requested_dir,
            default_root=DEFAULT_OUTPUT_DIR,
            model=_global_model(model),
            resume=resume,
        )
    )
    record_model = session_result.record.model
    next_state = TuiState(
        session_id=session_result.record.session_id,
        session_dir=session_result.record.session_dir,
        model=_model_settings_from_global(record_model),
        global_session_id=session_result.record.session_id,
        global_session_path=session_result.record.paths.global_session,
    )
    persist_state(next_state)
    append_event(next_state, "session_resumed", f"Interactive ASA session resumed via /resume {target_value}")
    return next_state


def replace_model(state: TuiState, model: ModelSettings) -> TuiState:
    next_state = TuiState(
        session_id=state.session_id,
        session_dir=state.session_dir,
        model=model,
        last_run_ledger=state.last_run_ledger,
        team_ledger=state.team_ledger,
        runtime_ledger=state.runtime_ledger,
        global_session_id=state.global_session_id,
        global_session_path=state.global_session_path,
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
        global_session_id=state.global_session_id,
        global_session_path=state.global_session_path,
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
        global_session_id=state.global_session_id,
        global_session_path=state.global_session_path,
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
        global_session_id=state.global_session_id,
        global_session_path=state.global_session_path,
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
    append_global_session_event(state.session_dir, event_type, summary)
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
        "global_session_id": state.global_session_id or state.session_id,
        "global_session_path": str(state.global_session_path) if state.global_session_path else "",
    }


def _global_model(model: ModelSettings) -> GlobalSessionModel:
    return GlobalSessionModel(
        provider=model.provider,
        name=model.name,
        reasoning_effort=model.reasoning_effort,
        base_url=model.base_url,
        auth_mode=model.auth_mode,
        api_key_env=model.api_key_env,
    )


def _model_settings_from_global(model: GlobalSessionModel | ModelSettings) -> ModelSettings:
    return ModelSettings(
        provider=model.provider,
        name=model.name,
        reasoning_effort=model.reasoning_effort,
        base_url=model.base_url,
        auth_mode=model.auth_mode,
        api_key_env=model.api_key_env,
    )


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", "~", ".")) or "\\" in value
