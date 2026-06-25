from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

from sim_agent.agents_sdk_runtime.roles import AGENT_ROLES
from sim_agent.agents_sdk_runtime.prompt_assets import load_domain_role_prompt
from sim_agent.schemas._parse import JsonMap, as_mapping, as_str
from sim_agent.schemas.errors import SchemaValidationError

from .global_session_store import backup_legacy_json, write_global_session
from .global_session_types import (
    ORCHESTRATOR_AGENT_ID,
    GlobalSessionModel,
    GlobalSessionRecord,
)


AGENT_REGISTRY_SCHEMA_VERSION: Final = "asa_agent_registry_v2"
AGENT_REGISTRY_LEDGER_NAME: Final = "agent_registry.json"
AGENT_SESSION_SCHEMA_VERSION: Final = "asa_agent_session_v2"
AGENT_SESSION_EVENT_SCHEMA_VERSION: Final = "asa_agent_session_event_v2"


@dataclass(frozen=True, slots=True)
class AgentRoleSeed:
    agent_id: str
    display_name: str
    boundary: str
    role_prompt: str


@dataclass(frozen=True, slots=True)
class AgentSessionHandle:
    agent_id: str
    display_name: str
    boundary: str
    role_prompt: str
    agent_session_id: str
    session_dir: Path
    messages_path: Path
    events_path: Path
    model: GlobalSessionModel
    created_at: float


@dataclass(frozen=True, slots=True)
class AgentRegistry:
    schema_version: str
    global_session_id: str
    registry_path: Path
    handles: dict[str, AgentSessionHandle]


def ensure_agent_registry(record: GlobalSessionRecord) -> AgentRegistry:
    registry_path = record.session_dir / AGENT_REGISTRY_LEDGER_NAME
    if registry_path.is_file():
        registry = load_agent_registry(record.session_dir)
        _ensure_handle_sessions(registry)
        _append_resume_events(registry)
        return registry
    registry = _create_agent_registry(record, registry_path)
    _write_agent_registry(registry)
    return registry


def load_agent_registry(session_dir: Path) -> AgentRegistry:
    path = session_dir / AGENT_REGISTRY_LEDGER_NAME
    payload = as_mapping(json.loads(path.read_text(encoding="utf-8")), "agent_registry")
    handles_value = payload.get("handles")
    if not isinstance(handles_value, list):
        raise SchemaValidationError("agent_registry.handles must be a list")
    handles = [_handle_from_payload(as_mapping(item, "agent_registry.handle")) for item in handles_value]
    return AgentRegistry(
        schema_version=as_str(payload.get("schema_version"), "agent_registry.schema_version"),
        global_session_id=as_str(payload.get("global_session_id"), "agent_registry.global_session_id"),
        registry_path=path,
        handles={handle.agent_id: handle for handle in handles},
    )


def _create_agent_registry(record: GlobalSessionRecord, registry_path: Path) -> AgentRegistry:
    handles = tuple(_create_handle(seed, record) for seed in _role_seeds())
    for handle in handles:
        _write_handle_session(handle)
        _append_agent_event(handle, "agent_session_registered", "Persistent agent session handle registered")
    refreshed = replace(record, agent_ids=tuple(handle.agent_id for handle in handles))
    write_global_session(refreshed)
    return AgentRegistry(
        schema_version=AGENT_REGISTRY_SCHEMA_VERSION,
        global_session_id=record.session_id,
        registry_path=registry_path,
        handles={handle.agent_id: handle for handle in handles},
    )


def _write_agent_registry(registry: AgentRegistry) -> None:
    payload: JsonMap = {
        "schema_version": registry.schema_version,
        "global_session_id": registry.global_session_id,
        "handles": [_handle_payload(handle) for handle in registry.handles.values()],
    }
    registry.registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry.registry_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_handle_session(handle: AgentSessionHandle) -> None:
    handle.session_dir.mkdir(parents=True, exist_ok=True)
    payload = _handle_payload(handle)
    with (handle.session_dir / "session.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ensure_handle_sessions(registry: AgentRegistry) -> None:
    for handle in registry.handles.values():
        path = handle.session_dir / "session.json"
        if path.is_file():
            backup = backup_legacy_json(path, AGENT_SESSION_SCHEMA_VERSION)
            if backup is None:
                continue
        _write_handle_session_with_replace(handle)


def _write_handle_session_with_replace(handle: AgentSessionHandle) -> None:
    handle.session_dir.mkdir(parents=True, exist_ok=True)
    path = handle.session_dir / "session.json"
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp_path.write_text(json.dumps(_handle_payload(handle), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _append_resume_events(registry: AgentRegistry) -> None:
    for handle in registry.handles.values():
        _append_agent_event(handle, "agent_session_resumed", "Persistent agent session handle resumed")


def _append_agent_event(handle: AgentSessionHandle, event_type: str, summary: str) -> None:
    sequence = _next_sequence(handle.events_path)
    payload: JsonMap = {
        "schema_version": AGENT_SESSION_EVENT_SCHEMA_VERSION,
        "at": time.time(),
        "sequence": sequence,
        "agent_id": handle.agent_id,
        "agent_session_id": handle.agent_session_id,
        "event_type": event_type,
        "summary": summary,
    }
    handle.events_path.parent.mkdir(parents=True, exist_ok=True)
    with handle.events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _create_handle(seed: AgentRoleSeed, record: GlobalSessionRecord) -> AgentSessionHandle:
    session_dir = record.paths.agent_sessions / seed.agent_id
    return AgentSessionHandle(
        agent_id=seed.agent_id,
        display_name=seed.display_name,
        boundary=seed.boundary,
        role_prompt=seed.role_prompt,
        agent_session_id=f"{record.session_id}:{seed.agent_id}",
        session_dir=session_dir,
        messages_path=session_dir / "messages.jsonl",
        events_path=session_dir / "events.jsonl",
        model=record.model,
        created_at=record.created_at,
    )


def _role_seeds() -> tuple[AgentRoleSeed, ...]:
    orchestrator = AgentRoleSeed(
        ORCHESTRATOR_AGENT_ID,
        "Orchestrator",
        "routes work, approvals, and final run assembly",
        load_domain_role_prompt(ORCHESTRATOR_AGENT_ID),
    )
    specialists = tuple(
        AgentRoleSeed(role.role_id, role.display_name, role.boundary, role.instructions)
        for role in AGENT_ROLES
    )
    return (orchestrator, *specialists)


def _handle_payload(handle: AgentSessionHandle) -> JsonMap:
    return {
        "schema_version": AGENT_SESSION_SCHEMA_VERSION,
        "agent_id": handle.agent_id,
        "display_name": handle.display_name,
        "boundary": handle.boundary,
        "role_prompt": handle.role_prompt,
        "agent_session_id": handle.agent_session_id,
        "session_dir": str(handle.session_dir),
        "messages_path": str(handle.messages_path),
        "events_path": str(handle.events_path),
        "model": asdict(handle.model),
        "created_at": handle.created_at,
    }


def _handle_from_payload(payload: JsonMap) -> AgentSessionHandle:
    model_payload = as_mapping(payload.get("model"), "agent_registry.handle.model")
    return AgentSessionHandle(
        agent_id=as_str(payload.get("agent_id"), "agent_registry.handle.agent_id"),
        display_name=as_str(payload.get("display_name"), "agent_registry.handle.display_name"),
        boundary=as_str(payload.get("boundary"), "agent_registry.handle.boundary"),
        role_prompt=as_str(payload.get("role_prompt"), "agent_registry.handle.role_prompt"),
        agent_session_id=as_str(payload.get("agent_session_id"), "agent_registry.handle.agent_session_id"),
        session_dir=Path(as_str(payload.get("session_dir"), "agent_registry.handle.session_dir")),
        messages_path=Path(as_str(payload.get("messages_path"), "agent_registry.handle.messages_path")),
        events_path=Path(as_str(payload.get("events_path"), "agent_registry.handle.events_path")),
        model=GlobalSessionModel(
            provider=as_str(model_payload.get("provider"), "agent_registry.handle.model.provider"),
            name=as_str(model_payload.get("name"), "agent_registry.handle.model.name"),
            reasoning_effort=as_str(model_payload.get("reasoning_effort"), "agent_registry.handle.model.reasoning_effort"),
            base_url=as_str(model_payload.get("base_url"), "agent_registry.handle.model.base_url"),
            auth_mode=as_str(model_payload.get("auth_mode"), "agent_registry.handle.model.auth_mode"),
            api_key_env=as_str(model_payload.get("api_key_env"), "agent_registry.handle.model.api_key_env"),
        ),
        created_at=_as_float(payload.get("created_at"), "agent_registry.handle.created_at"),
    )


def _next_sequence(path: Path) -> int:
    if not path.is_file():
        return 1
    return len(path.read_text(encoding="utf-8").splitlines()) + 1


def _as_float(value: object, field: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise SchemaValidationError(f"{field} must be a number")
