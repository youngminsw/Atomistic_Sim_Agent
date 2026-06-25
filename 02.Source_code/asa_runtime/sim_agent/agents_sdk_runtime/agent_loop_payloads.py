from __future__ import annotations

from sim_agent.agent_harness.tools import RuntimeToolResult
from sim_agent.schemas._parse import JsonMap

from .agent_loop_contract import AsaAgentSession, ToolChoiceModel


def loop_status(tool_results: tuple[RuntimeToolResult, ...], blockers: tuple[str, ...]) -> str:
    if blockers:
        return "blocked"
    if all(result.status == "succeeded" for result in tool_results):
        return "succeeded"
    return "failed"


def model_id(model: ToolChoiceModel, session: AsaAgentSession) -> str:
    model_id_for_session = getattr(model, "model_id_for_session", None)
    if callable(model_id_for_session):
        value = model_id_for_session(session)
        if isinstance(value, str) and value:
            return value
    return model.model_id


def tool_end_payload(call_id: str, result: RuntimeToolResult) -> JsonMap:
    status = "ok" if result.status == "succeeded" else result.status
    payload: JsonMap = {
        "tool_call_id": call_id,
        "tool_name": result.tool_name,
        "status": status,
        "summary": f"{result.tool_name}:{result.status}",
    }
    if result.artifact_ref:
        payload["artifact_ref"] = result.artifact_ref
    if result.blocker:
        payload["blocker"] = result.blocker
    return payload


def supports_tool_result_continuation(model: ToolChoiceModel) -> bool:
    return bool(getattr(model, "supports_tool_result_continuation", False))


def final_output_for_session(
    model: ToolChoiceModel,
    session: AsaAgentSession,
    tool_results: tuple[RuntimeToolResult, ...],
) -> str:
    final_output = getattr(model, "final_output_for_session", None)
    if callable(final_output):
        value = final_output(session, tool_results)
        if isinstance(value, str):
            return value
    return ""
