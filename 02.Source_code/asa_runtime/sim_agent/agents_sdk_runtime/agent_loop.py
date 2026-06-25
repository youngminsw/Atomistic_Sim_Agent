from __future__ import annotations

from dataclasses import dataclass

from sim_agent.agent_harness.tools import RuntimeToolResult, execute_runtime_tool
from sim_agent.schemas._parse import JsonMap

from .agent_loop_contract import (
    AgentLoopResult,
    AsaAgentSession,
    ModelSelectedToolCall,
    ModelToolChoiceBlocked,
    ModelTurnResult,
    StaticToolChoiceModel,
    ToolChoiceModel,
)
from .agent_loop_events import AgentLoopEventStream, loop_turn_id, model_turn_id, tool_call_id
from .agent_loop_payloads import (
    final_output_for_session,
    loop_status,
    model_id,
    supports_tool_result_continuation,
    tool_end_payload,
)
from .types import RuntimeEventType, RuntimeTraceEvent

__all__ = [
    "AgentLoop",
    "AgentLoopResult",
    "AsaAgentSession",
    "ModelSelectedToolCall",
    "ModelTurnResult",
    "ModelToolChoiceBlocked",
    "StaticToolChoiceModel",
    "ToolChoiceModel",
]


@dataclass(frozen=True, slots=True)
class AgentLoop:
    session: AsaAgentSession
    model: ToolChoiceModel
    max_steps: int = 4
    cancel_requested: bool = False

    def run(self) -> AgentLoopResult:
        tool_schemas = self.session.model_visible_tool_schemas()
        active_model_id = model_id(self.model, self.session)
        event_stream = AgentLoopEventStream(session_id=self.session.session_id, agent_id=self.session.agent_id)
        trace: list[RuntimeTraceEvent] = [
            RuntimeTraceEvent("asa_agent_session_created", self.session.agent_id, self.session.session_id),
            RuntimeTraceEvent("model_visible_tools_registered", self.session.agent_id, str(len(tool_schemas))),
        ]
        if self.cancel_requested:
            event_stream.append(
                RuntimeEventType.CANCELLATION,
                loop_turn_id(self.session.session_id),
                {"status": "cancelled", "summary": "cancelled"},
            )
            trace.append(RuntimeTraceEvent("agent_loop_cancelled", self.session.agent_id, "cancelled"))
            trace.append(RuntimeTraceEvent("agent_loop_completed", self.session.agent_id, "cancelled"))
            return AgentLoopResult(
                session=self.session,
                model_id=active_model_id,
                tool_schemas=tool_schemas,
                selected_tools=(),
                tool_results=(),
                trace=tuple(trace),
                status="cancelled",
                blockers=("cancelled",),
                runtime_events=event_stream.snapshot(),
            )
        if not self.session.messages:
            self.session.append_message("user", self.session.user_goal)
            event_stream.append(
                RuntimeEventType.MESSAGE_APPEND,
                loop_turn_id(self.session.session_id),
                {"role": "user", "text": self.session.user_goal, "status": "ok"},
            )
        else:
            event_stream.append(
                RuntimeEventType.RESUME,
                loop_turn_id(self.session.session_id),
                {"status": "resumed", "summary": f"{len(self.session.messages)} messages loaded"},
            )
        selected_history: list[ModelSelectedToolCall] = []
        tool_results: list[RuntimeToolResult] = []
        blockers: tuple[str, ...] = ()
        final_output = ""
        for step in range(1, self.max_steps + 1):
            turn_id = model_turn_id(self.session.session_id, step)
            trace.append(RuntimeTraceEvent("agent_loop_model_step_started", self.session.agent_id, str(step)))
            event_stream.append(
                RuntimeEventType.MODEL_START,
                turn_id,
                {"model": active_model_id, "step": step, "status": "running"},
            )
            try:
                model_turn = _model_turn_for_session(self.model, self.session, tool_schemas)
            except ModelToolChoiceBlocked as exc:
                blockers = (exc.blocker,)
                trace.append(RuntimeTraceEvent("agent_loop_model_selected_tools", self.session.agent_id, "0"))
                trace.append(RuntimeTraceEvent("model_tool_selection_blocked", self.session.agent_id, exc.blocker))
                trace.append(RuntimeTraceEvent("agent_loop_completed", self.session.agent_id, "blocked"))
                event_stream.append(
                    RuntimeEventType.MODEL_END,
                    turn_id,
                    {"status": "blocked", "summary": exc.blocker},
                )
                event_stream.append(
                    RuntimeEventType.BLOCKER,
                    turn_id,
                    {"status": "blocked", "summary": exc.blocker},
                )
                return AgentLoopResult(
                    session=self.session,
                    model_id=active_model_id,
                    tool_schemas=tool_schemas,
                    selected_tools=tuple(selected_history),
                    tool_results=tuple(tool_results),
                    trace=tuple(trace),
                    status="blocked",
                    blockers=blockers,
                    runtime_events=event_stream.snapshot(),
                    final_output=final_output,
                )
            selected_tools = model_turn.selected_tools
            trace.append(RuntimeTraceEvent("agent_loop_model_selected_tools", self.session.agent_id, str(len(selected_tools))))
            event_stream.append(
                RuntimeEventType.MODEL_DELTA,
                turn_id,
                {"status": "streaming", "text": f"selected_tools={len(selected_tools)}"},
            )
            event_stream.append(
                RuntimeEventType.MODEL_END,
                turn_id,
                {"status": "ok", "summary": f"selected_tools={len(selected_tools)}"},
            )
            if not selected_tools:
                if model_turn.final_output:
                    final_output = model_turn.final_output
                    self.session.append_message("assistant", final_output)
                    trace.append(RuntimeTraceEvent("agent_loop_final_output", self.session.agent_id, final_output))
                    event_stream.append(
                        RuntimeEventType.MESSAGE_APPEND,
                        turn_id,
                        {"role": "assistant", "text": final_output, "status": "ok"},
                    )
                elif not selected_history:
                    blockers = ("no_model_tool_selected",)
                    event_stream.append(
                        RuntimeEventType.BLOCKER,
                        turn_id,
                        {"status": "blocked", "summary": "no_model_tool_selected"},
                    )
                else:
                    final_output = final_output_for_session(self.model, self.session, tuple(tool_results))
                    if final_output:
                        self.session.append_message("assistant", final_output)
                        trace.append(RuntimeTraceEvent("agent_loop_final_output", self.session.agent_id, final_output))
                        event_stream.append(
                            RuntimeEventType.MESSAGE_APPEND,
                            turn_id,
                            {"role": "assistant", "text": final_output, "status": "ok"},
                        )
                break
            selected_history.extend(selected_tools)
            for index, selected in enumerate(selected_tools, start=1):
                call_id = tool_call_id(turn_id, index, selected.tool_name)
                trace.append(RuntimeTraceEvent("model_tool_selected", self.session.agent_id, selected.tool_name))
                event_stream.append(
                    RuntimeEventType.TOOL_START,
                    turn_id,
                    {"tool_call_id": call_id, "tool_name": selected.tool_name, "status": "running"},
                    correlation_id=call_id,
                    parent_id=turn_id,
                )
                result = execute_runtime_tool(self.session.runtime_tool_call(selected), self.session.registry, self.session.output_dir)
                tool_results.append(result)
                self.session.append_tool_result(result)
                trace.append(RuntimeTraceEvent("tool_executed", self.session.agent_id, f"{result.tool_name}:{result.status}"))
                trace.append(RuntimeTraceEvent("tool_result_appended", self.session.agent_id, result.tool_name))
                event_stream.append(
                    RuntimeEventType.TOOL_END,
                    turn_id,
                    tool_end_payload(call_id, result),
                    correlation_id=call_id,
                    parent_id=turn_id,
                )
                if result.blocker:
                    trace.append(RuntimeTraceEvent("tool_blocked", self.session.agent_id, f"{result.tool_name}:{result.blocker}"))
                    event_stream.append(
                        RuntimeEventType.BLOCKER,
                        turn_id,
                        {
                            "status": "blocked",
                            "summary": result.blocker,
                            "tool_name": result.tool_name,
                        },
                        correlation_id=call_id,
                        parent_id=turn_id,
                    )
            blockers = tuple(result.blocker for result in tool_results if result.blocker is not None)
            if blockers or not supports_tool_result_continuation(self.model):
                final_output = final_output_for_session(self.model, self.session, tuple(tool_results))
                if final_output:
                    self.session.append_message("assistant", final_output)
                    trace.append(RuntimeTraceEvent("agent_loop_final_output", self.session.agent_id, final_output))
                    event_stream.append(
                        RuntimeEventType.MESSAGE_APPEND,
                        turn_id,
                        {"role": "assistant", "text": final_output, "status": "ok"},
                    )
                break
        else:
            blockers = ("max_agent_loop_steps_exceeded",)
            event_stream.append(
                RuntimeEventType.BLOCKER,
                loop_turn_id(self.session.session_id),
                {"status": "blocked", "summary": "max_agent_loop_steps_exceeded"},
            )
        if not selected_history and not blockers and not final_output:
            blockers = ("no_model_tool_selected",)
            event_stream.append(
                RuntimeEventType.BLOCKER,
                loop_turn_id(self.session.session_id),
                {"status": "blocked", "summary": "no_model_tool_selected"},
            )
        status = loop_status(tuple(tool_results), blockers)
        trace.append(RuntimeTraceEvent("agent_loop_completed", self.session.agent_id, status))
        return AgentLoopResult(
            session=self.session,
            model_id=active_model_id,
            tool_schemas=tool_schemas,
            selected_tools=tuple(selected_history),
            tool_results=tuple(tool_results),
            trace=tuple(trace),
            status=status,
            blockers=blockers,
            runtime_events=event_stream.snapshot(),
            final_output=final_output,
        )


def _model_turn_for_session(
    model: ToolChoiceModel,
    session: AsaAgentSession,
    tool_schemas: tuple[JsonMap, ...],
) -> ModelTurnResult:
    complete_turn = getattr(model, "complete_turn", None)
    if callable(complete_turn):
        value = complete_turn(session, tool_schemas)
        if isinstance(value, ModelTurnResult):
            return value
    return ModelTurnResult(selected_tools=model.choose_tools(session, tool_schemas))
