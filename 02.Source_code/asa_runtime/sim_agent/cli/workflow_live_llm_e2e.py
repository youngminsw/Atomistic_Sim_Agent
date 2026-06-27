from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.agent_harness.tools import tool_registry_for_agent
from sim_agent.agent_harness.tool_types import ToolRegistry
from sim_agent.agents_sdk_runtime.agent_loop import AgentLoop, AsaAgentSession
from sim_agent.agents_sdk_runtime.prompt_assets import load_domain_role_prompt
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.agents_sdk_runtime.workflow_harness_types import normalize_workflow_id
from sim_agent.llm_endpoints import ModelProviderConfig
from sim_agent.provider_registry import OPENAI_CODEX_BASE_URL, OPENAI_CODEX_TOKEN_ENV
from sim_agent.schemas._parse import JsonMap
from sim_agent.ui.model_auth import access_token_for_provider, model_credential_status


LIVE_LLM_BLOCKER: Final = "live_llm_provider_unavailable"
LIVE_LLM_OPT_IN_BLOCKER: Final = "live_llm_opt_in_required"
LIVE_LLM_TOOL_BLOCKER: Final = "live_llm_workflow_tool_not_selected"
LIVE_LLM_RUNTIME_BLOCKER: Final = "live_llm_workflow_runtime_blocked"
LIVE_WORKFLOW_IDS: Final = ("deep-interview", "ralplan", "ultragoal", "visual-qa", "ultraresearch")
WORKFLOW_ARGS_BEGIN: Final = "ASA_WORKFLOW_START_ARGUMENTS_JSON_BEGIN"
WORKFLOW_ARGS_END: Final = "ASA_WORKFLOW_START_ARGUMENTS_JSON_END"


@dataclass(frozen=True, slots=True)
class WorkflowLiveLlmE2ERequest:
    output_dir: Path
    scenario: str


@dataclass(frozen=True, slots=True)
class WorkflowLiveLlmE2EResult:
    status: str
    output_json: Path
    provider_events_path: Path
    blockers: tuple[str, ...]


def run_workflow_live_llm_e2e(request: WorkflowLiveLlmE2ERequest) -> WorkflowLiveLlmE2EResult:
    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(request.scenario)
    provider_events_path = output_dir / "provider-events.jsonl"
    endpoint = _default_endpoint()
    credential = model_credential_status(endpoint.provider)
    provider_context = _provider_context(endpoint, credential)
    live_opt_in = os.environ.get("ASA_LIVE_LLM_E2E") == "1"
    if not live_opt_in:
        payload = _blocked_payload(
            request,
            run_id,
            provider_context,
            provider_events_path,
            LIVE_LLM_OPT_IN_BLOCKER,
            live_opt_in=live_opt_in,
        )
        output_json = output_dir / "workflow-live-llm.json"
        _write_json(output_json, payload)
        return WorkflowLiveLlmE2EResult("blocked", output_json, provider_events_path, (LIVE_LLM_OPT_IN_BLOCKER,))
    if _token_source(endpoint) == "missing":
        payload = _blocked_payload(
            request,
            run_id,
            provider_context,
            provider_events_path,
            LIVE_LLM_BLOCKER,
            live_opt_in=live_opt_in,
        )
        output_json = output_dir / "workflow-live-llm.json"
        _write_json(output_json, payload)
        return WorkflowLiveLlmE2EResult("blocked", output_json, provider_events_path, (LIVE_LLM_BLOCKER,))

    rows = tuple(_run_live_workflow(output_dir, run_id, endpoint, workflow_id) for workflow_id in LIVE_WORKFLOW_IDS)
    _write_jsonl(provider_events_path, rows)
    blockers = tuple(
        blocker
        for row in rows
        for blocker in _row_blockers(row)
    )
    status = "succeeded" if not blockers else "blocked"
    payload: JsonMap = {
        "schema_version": "asa_workflow_live_llm_e2e_v1",
        "scenario": request.scenario,
        "run_id": run_id,
        "status": status,
        "blockers": list(blockers),
        "workflow_ids": [f"/{workflow_id}" for workflow_id in LIVE_WORKFLOW_IDS],
        "live_llm": list(rows),
        "provider_events": provider_events_path.name,
        "requires_env": {"ASA_LIVE_LLM_E2E": "1"},
        "live_opt_in": live_opt_in,
        "provider": provider_context,
    }
    output_json = output_dir / "workflow-live-llm.json"
    _write_json(output_json, payload)
    return WorkflowLiveLlmE2EResult(status, output_json, provider_events_path, blockers)


def _run_live_workflow(output_dir: Path, run_id: str, endpoint: ModelProviderConfig, workflow_id: str) -> JsonMap:
    normalized = normalize_workflow_id(workflow_id)
    session_dir = output_dir / "sessions" / normalized
    arguments = _workflow_start_arguments(normalized, run_id, session_dir)
    session = AsaAgentSession(
        run_id=f"{run_id}-{normalized}",
        session_id=f"live-e2e-{normalized}",
        agent_id="orchestrator",
        user_goal=_workflow_user_goal(normalized, arguments),
        endpoint=endpoint,
        output_dir=session_dir,
        registry=_workflow_start_registry(),
        role_prompt=load_domain_role_prompt("orchestrator"),
        workflow_state={"active_workflow_id": normalized, "live_e2e": True},
        skills=("/deep-interview", "/ralplan", "/ultragoal", "/visual-qa", "/ultraresearch", "insane-search"),
    )
    result = AgentLoop(
        session,
        ProviderToolChoiceModel(timeout_s=_timeout_s(), retry_count=0),
        max_steps=1,
    ).run()
    tool_rows = [
        {
            "tool_name": tool_result.tool_name,
            "status": tool_result.status,
            "blocker": tool_result.blocker or "",
            "artifact_ref": tool_result.artifact_ref,
            "output": tool_result.output,
        }
        for tool_result in result.tool_results
    ]
    expected = _expected_runtime_observed(normalized, tool_rows)
    blockers = _live_workflow_blockers(result, normalized, tool_rows, expected)
    row: JsonMap = {
        "workflow": f"/{normalized}",
        "run_id": run_id,
        "session_id": session.session_id,
        "provider": endpoint.provider,
        "model": endpoint.model,
        "auth_mode": endpoint.auth_mode,
        "api_protocol": endpoint.api_protocol,
        "model_id": result.model_id,
        "status": "passed" if not blockers else "blocked",
        "agent_loop_status": result.status,
        "blockers": list(blockers),
        "selected_tools": [selected.tool_name for selected in result.selected_tools],
        "tool_results": tool_rows,
        "final_output_present": bool(result.final_output),
        "prompt_manifest": str(session_dir / "prompt_assembly_manifest.json"),
    }
    return row


def _default_endpoint() -> ModelProviderConfig:
    return ModelProviderConfig.from_mapping(
        {
            "provider": "openai-codex",
            "model": os.environ.get("ASA_LIVE_LLM_E2E_MODEL", "gpt-5.5"),
            "reasoning_effort": "high",
            "base_url": os.environ.get("ASA_LIVE_LLM_E2E_BASE_URL", OPENAI_CODEX_BASE_URL),
            "auth_mode": "oauth",
            "api_protocol": "openai_codex_responses",
            "api_key_env": OPENAI_CODEX_TOKEN_ENV,
        }
    )


def _workflow_start_registry() -> ToolRegistry:
    registry = tool_registry_for_agent("orchestrator")
    return ToolRegistry(tuple(tool for tool in registry.tools if tool.name == "workflow_start"))


def _workflow_start_arguments(workflow_id: str, run_id: str, session_dir: Path) -> JsonMap:
    payload: JsonMap = {
        "request_id": f"{run_id}-{workflow_id}",
        "user_goal": f"Live LLM e2e proof for ASA /{workflow_id}.",
        "evidence": _workflow_evidence(workflow_id, session_dir),
    }
    gate = _workflow_gate(workflow_id)
    if gate is not None:
        payload["gate"] = gate
    return {
        "workflow_id": workflow_id,
        "owner_agent_id": "orchestrator",
        "target_agent_id": "orchestrator",
        "goal_id": f"live-e2e-{workflow_id}",
        "payload": payload,
    }


def _workflow_evidence(workflow_id: str, session_dir: Path) -> JsonMap:
    match workflow_id:
        case "deep-interview":
            return {
                "question_answer": "Live provider reached the deep-interview command and created the ambiguity gate.",
                "ambiguity_score": "bounded",
            }
        case "ralplan":
            return {
                "prd_path": "ralplan/prd.md",
                "test_spec_path": "ralplan/test-spec.md",
            }
        case "ultragoal":
            return {
                "codex_goal_snapshot": {
                    "active": True,
                    "goal_id": f"live-e2e-{workflow_id}",
                    "status": "in_progress",
                }
            }
        case "visual-qa":
            screenshot_ref = "visual-qa-live-screenshot.txt"
            screenshot_path = session_dir / "workflows" / screenshot_ref
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            screenshot_path.write_bytes(b"ASA live visual QA fixture is intentionally non-empty.\n")
            return {
                "surface_ref": "asa://live-e2e/visual-qa",
                "screenshot_ref": screenshot_ref,
                "oracle_verdict": {
                    "passed": True,
                    "summary": "Live e2e visual QA fixture is non-empty and hashable.",
                    "checks": ["non_empty_fixture", "hash_recorded"],
                },
            }
        case "ultraresearch":
            return {
                "research_question": "Does the ASA ultraresearch command preserve public-source boundaries?",
                "source_journal": "live-e2e public source acquisition journal",
                "insane_search_trace": {
                    "skill_id": "insane_search",
                    "surface": "skill",
                    "ok": True,
                    "public_only": True,
                    "ssrf_safe": True,
                    "auth_required": False,
                    "grid_exhausted": True,
                    "untried_routes": [],
                    "must_invoke_playwright_mcp": False,
                    "stop_reason": "public_sources_ready",
                    "routes": ["search"],
                    "sources": [
                        {
                            "url": "https://example.com/asa-live-e2e",
                            "route": "search",
                            "title": "ASA live e2e public fixture",
                            "evidence_ref": "insane_search_trace",
                        }
                    ],
                },
            }
        case _:
            return {}


def _workflow_gate(workflow_id: str) -> JsonMap | None:
    match workflow_id:
        case "ralplan":
            return {
                "gate_id": "approval",
                "gate_kind": "response_schema",
                "response_schema": {
                    "type": "object",
                    "required": ["decision"],
                    "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["approve", "request-changes", "reject"]},
                        "comments": {"type": "string"},
                    },
                },
            }
        case "ultragoal":
            return {
                "gate_id": "signoff",
                "gate_kind": "response_schema",
                "response_schema": {
                    "type": "object",
                    "required": ["decision"],
                    "additionalProperties": False,
                    "properties": {
                        "decision": {"type": "string", "enum": ["approve", "decline"]},
                        "reason": {"type": "string"},
                    },
                },
            }
        case _:
            return None


def _workflow_user_goal(workflow_id: str, arguments: JsonMap) -> str:
    return "\n".join(
        (
            f"Execute the canonical ASA workflow command /{workflow_id}.",
            "Call the workflow_start tool exactly once and pass this exact JSON object as the tool arguments.",
            "Do not call any other tool. Do not answer in prose before the tool call.",
            WORKFLOW_ARGS_BEGIN,
            json.dumps(arguments, indent=2, sort_keys=True),
            WORKFLOW_ARGS_END,
        )
    )


def _expected_runtime_observed(workflow_id: str, tool_rows: list[JsonMap]) -> bool:
    if not tool_rows:
        return False
    row = tool_rows[0]
    if row.get("tool_name") != "workflow_start":
        return False
    output = row.get("output")
    if not isinstance(output, dict):
        return False
    if output.get("workflow_id") != workflow_id:
        return False
    if workflow_id == "deep-interview":
        return row.get("status") == "blocked" and row.get("blocker") == "workflow_gate_response_required"
    if workflow_id in {"ralplan", "ultragoal"}:
        return row.get("status") == "blocked" and row.get("blocker") == "workflow_gate_response_required"
    return row.get("status") == "ready" and not row.get("blocker")


def _live_workflow_blockers(
    result,
    workflow_id: str,
    tool_rows: list[JsonMap],
    expected_runtime_observed: bool,
) -> tuple[str, ...]:
    if not result.selected_tools:
        return tuple(result.blockers or (LIVE_LLM_TOOL_BLOCKER,))
    if tuple(selected.tool_name for selected in result.selected_tools) != ("workflow_start",):
        return (LIVE_LLM_TOOL_BLOCKER,)
    if not expected_runtime_observed:
        tool_blockers = tuple(
            blocker for row in tool_rows if isinstance((blocker := row.get("blocker")), str) and blocker
        )
        return tool_blockers or (LIVE_LLM_RUNTIME_BLOCKER,)
    return ()


def _provider_context(endpoint: ModelProviderConfig, credential) -> JsonMap:
    return {
        "provider": endpoint.provider,
        "model": endpoint.model,
        "auth_mode": endpoint.auth_mode,
        "api_protocol": endpoint.api_protocol,
        "base_url": endpoint.base_url,
        "api_key_env": endpoint.api_key_env,
        "credential_logged_in": credential.logged_in,
        "credential_expires": credential.expires,
        "provider_credential_store": str(credential.provider_credential_store),
        "token_source": _token_source(endpoint),
    }


def _token_source(endpoint: ModelProviderConfig) -> str:
    if os.environ.get(endpoint.api_key_env):
        return "env"
    if access_token_for_provider(endpoint.provider):
        return "credential_store"
    return "missing"


def _blocked_payload(
    request: WorkflowLiveLlmE2ERequest,
    run_id: str,
    provider_context: JsonMap,
    provider_events_path: Path,
    blocker: str,
    *,
    live_opt_in: bool,
) -> JsonMap:
    live_row: JsonMap = {
        "workflow": "/deep-interview",
        "status": "blocked",
        "blockers": [blocker],
        "run_id": run_id,
        "provider": provider_context.get("provider"),
        "model": provider_context.get("model"),
    }
    _write_jsonl(provider_events_path, (live_row,))
    return {
        "schema_version": "asa_workflow_live_llm_e2e_v1",
        "scenario": request.scenario,
        "run_id": run_id,
        "status": "blocked",
        "blockers": [blocker],
        "live_llm": [live_row],
        "provider_events": provider_events_path.name,
        "requires_env": {"ASA_LIVE_LLM_E2E": "1"},
        "live_opt_in": live_opt_in,
        "provider": provider_context,
    }


def _row_blockers(row: JsonMap) -> tuple[str, ...]:
    blockers = row.get("blockers")
    if not isinstance(blockers, list | tuple):
        return ()
    return tuple(item for item in blockers if isinstance(item, str) and item)


def _timeout_s() -> float:
    raw = os.environ.get("ASA_LIVE_LLM_E2E_TIMEOUT_S", "60")
    try:
        value = float(raw)
    except ValueError:
        return 60.0
    return max(1.0, value)


def _run_id(scenario: str) -> str:
    safe_scenario = scenario.replace("_", "-").replace("/", "-")
    return f"{safe_scenario}-{int(time.time() * 1000)}"


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: tuple[JsonMap, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
