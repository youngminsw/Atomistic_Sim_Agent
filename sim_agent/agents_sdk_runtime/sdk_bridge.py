from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from sim_agent.llm_endpoints import ModelProviderConfig

from .roles import AGENT_ROLES
from .session_files import ensure_runtime_session_path
from .types import AgentsSdkTeam


class AgentsSdkRuntimeError(RuntimeError):
    pass


def agents_sdk_available() -> bool:
    return importlib.util.find_spec("agents") is not None


def build_agents_sdk_team(endpoint: ModelProviderConfig, session_id: str, session_path: Path | None = None) -> AgentsSdkTeam:
    try:
        agents_module = importlib.import_module("agents")
    except ImportError as exc:
        raise AgentsSdkRuntimeError("openai_agents_sdk_missing") from exc
    agent_class = getattr(agents_module, "Agent")
    sqlite_session_class = getattr(agents_module, "SQLiteSession")
    handoff_fn = getattr(agents_module, "handoff")
    specialists = {
        role.role_id: agent_class(
            name=role.display_name,
            handoff_description=role.boundary,
            instructions=role.instructions,
            model=endpoint.model,
        )
        for role in AGENT_ROLES
    }
    handoffs = [
        handoff_fn(
            specialists[role.role_id],
            tool_name_override=role.handoff_tool_name,
            tool_description_override=role.boundary,
        )
        for role in AGENT_ROLES
    ]
    orchestrator = agent_class(
        name="Orchestrator",
        handoff_description="Owns the simulation run and routes work to specialist agents.",
        instructions=(
            "Clarify missing inputs, route work to specialists, preserve approval boundaries, "
            "and stop on hard physics/data blockers."
        ),
        handoffs=handoffs,
        model=endpoint.model,
    )
    return AgentsSdkTeam(
        orchestrator=orchestrator,
        specialists=specialists,
        session=sqlite_session_class(session_id, str(session_path or ensure_runtime_session_path(session_id))),
        handoff_tool_names=tuple(role.handoff_tool_name for role in AGENT_ROLES),
    )


def run_agents_sdk_fake_gateway_smoke(
    endpoint: ModelProviderConfig,
    session_id: str,
    user_goal: str,
    session_path: Path | None = None,
) -> str:
    try:
        agents_module = importlib.import_module("agents")
        items_module = importlib.import_module("agents.items")
        interface_module = importlib.import_module("agents.models.interface")
        response_message_module = importlib.import_module("openai.types.responses.response_output_message")
        response_text_module = importlib.import_module("openai.types.responses.response_output_text")
    except ImportError as exc:
        raise AgentsSdkRuntimeError("openai_agents_sdk_missing") from exc
    model_class = getattr(interface_module, "Model")
    model_provider_class = getattr(interface_module, "ModelProvider")
    model_response_class = getattr(items_module, "ModelResponse")
    response_output_message_class = getattr(response_message_module, "ResponseOutputMessage")
    response_output_text_class = getattr(response_text_module, "ResponseOutputText")
    run_config_class = getattr(agents_module, "RunConfig")
    runner_class = getattr(agents_module, "Runner")
    usage_class = getattr(agents_module, "Usage")

    class FakeGatewayModel(model_class):
        async def get_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ):
            del system_instructions, input, model_settings, tools, output_schema, handoffs, tracing, prompt
            response_context = (previous_response_id, conversation_id)
            output_text = response_output_text_class(type="output_text", text="agents_sdk_runtime_ready", annotations=[])
            message = response_output_message_class(
                id=f"msg_agents_sdk_runtime_ready_{len(response_context)}",
                type="message",
                role="assistant",
                content=[output_text],
                status="completed",
            )
            return model_response_class(
                output=[message],
                usage=usage_class(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
                response_id="resp_agents_sdk_runtime_ready",
            )

        def stream_response(
            self,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            *,
            previous_response_id,
            conversation_id,
            prompt,
        ):
            del (
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id,
                conversation_id,
                prompt,
            )

            async def _empty_stream():
                for item in ():
                    yield item

            return _empty_stream()

    class FakeGatewayModelProvider(model_provider_class):
        def get_model(self, _model_name):
            return FakeGatewayModel()

    team = build_agents_sdk_team(endpoint, session_id, session_path)
    result = runner_class.run_sync(
        team.orchestrator,
        user_goal,
        max_turns=1,
        session=team.session,
        run_config=run_config_class(
            model_provider=FakeGatewayModelProvider(),
            tracing_disabled=True,
            workflow_name="Atomistic Simulation Agent SDK smoke",
        ),
    )
    return str(result.final_output)
