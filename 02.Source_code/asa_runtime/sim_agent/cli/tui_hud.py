from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim_agent.runtime_config import active_profile_status, load_runtime_config
from sim_agent.ui.model_connection import model_connection_status

from .tui_state import TuiState


@dataclass(frozen=True, slots=True)
class TuiHudStatus:
    provider: str
    model: str
    auth_mode: str
    connected: bool
    connection_label: str
    friendly_message: str
    action_hint: str
    active_profile: str
    profile_customized: bool
    session_id: str
    session_dir: Path
    last_run_ledger: Path | None
    team_ledger: Path | None
    runtime_ledger: Path | None


def build_hud_status(state: TuiState) -> TuiHudStatus:
    runtime_config = load_runtime_config()
    active = active_profile_status(runtime_config)
    connection = model_connection_status(
        state.model.provider,
        state.model.name,
        state.model.auth_mode,
        state.model.api_key_env,
    )
    return TuiHudStatus(
        provider=connection.provider,
        model=connection.model,
        auth_mode=connection.auth_mode,
        connected=connection.connected,
        connection_label=connection.connection_label,
        friendly_message=connection.friendly_message,
        action_hint=connection.action_hint,
        active_profile=active.name or "none",
        profile_customized=active.customized,
        session_id=state.session_id,
        session_dir=state.session_dir,
        last_run_ledger=state.last_run_ledger,
        team_ledger=state.team_ledger,
        runtime_ledger=state.runtime_ledger,
    )


def ledger_label(path: Path | None) -> str:
    return str(path) if path is not None else "-"
