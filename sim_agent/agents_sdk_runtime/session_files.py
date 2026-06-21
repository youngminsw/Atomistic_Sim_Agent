from __future__ import annotations

from pathlib import Path

from sim_agent.runtime_config import load_runtime_config


def ensure_runtime_session_path(session_id: str, output_dir: Path | None = None) -> Path:
    base_dir = output_dir or Path(load_runtime_config().evidence_root)
    session_dir = base_dir / "agents_sdk_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / f"{session_id}.sqlite"
    path.touch(exist_ok=True)
    return path
