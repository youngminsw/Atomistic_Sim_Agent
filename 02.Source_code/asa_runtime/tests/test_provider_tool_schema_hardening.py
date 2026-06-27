from __future__ import annotations

from pathlib import Path

import pytest

from sim_agent.agent_harness.tools import ToolRegistry, default_tool_registry, tool_registry_for_agent
from sim_agent.agents_sdk_runtime import AsaAgentSession, ModelToolChoiceBlocked
from sim_agent.agents_sdk_runtime.provider_tool_choice_model import ProviderToolChoiceModel
from sim_agent.agents_sdk_runtime.provider_transport import ProviderTransportPolicyError, provider_transport_request
from sim_agent.llm_endpoints import ModelProviderConfig


VALID_PARAMETERS = {
    "type": "object",
    "properties": {
        "relative_path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["relative_path", "content"],
    "additionalProperties": False,
}


def test_malformed_tool_schema_is_blocker_not_unknown_tool(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ProviderTransportPolicyError, match="malformed_tool_schema:name"):
        provider_transport_request(session, ({"parameters": VALID_PARAMETERS},))


def test_missing_parameters_do_not_become_permissive_object(tmp_path: Path) -> None:
    session = _session(tmp_path)

    with pytest.raises(ProviderTransportPolicyError, match="malformed_tool_schema:parameters"):
        provider_transport_request(session, (_tool_schema(parameters=None),))


def test_valid_tool_schema_converts_without_extra_properties(tmp_path: Path) -> None:
    session = _session(tmp_path)

    request = provider_transport_request(session, (_tool_schema(parameters=VALID_PARAMETERS),))

    tool = request.payload["tools"][0]
    assert tool["name"] == "artifact_write"
    assert tool["parameters"] == VALID_PARAMETERS


def test_registered_model_visible_tools_have_explicit_object_parameters(tmp_path: Path) -> None:
    agent_ids = ("orchestrator", "md_agent", "ml_agent", "feature_scale_agent", "research_agent", "qa_agent")
    missing: list[str] = []

    for agent_id in agent_ids:
        session = _session(tmp_path / agent_id, registry=tool_registry_for_agent(agent_id), agent_id=agent_id)
        for schema in session.model_visible_tool_schemas():
            parameters = schema.get("parameters")
            input_schema = schema.get("inputSchema")
            if not _is_object_schema(parameters) and not _is_object_schema(input_schema):
                missing.append(f"{agent_id}:{schema.get('name')}")

    assert missing == []


def test_external_schema_without_parameters_blocks_before_gateway_post(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.agents_sdk_runtime import provider_tool_choice_model

    gateway_calls: list[dict[str, object]] = []

    def gateway_post_json(*args: object, **kwargs: object) -> tuple[int, dict[str, object]]:
        del kwargs
        payload = args[1]
        assert isinstance(payload, dict)
        gateway_calls.append(payload)
        return 200, {"output_text": "posted"}

    monkeypatch.setattr(provider_tool_choice_model, "gateway_post_json", gateway_post_json)
    session = _session(tmp_path)
    model = ProviderToolChoiceModel(api_key="test-token", retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="malformed_tool_schema:parameters"):
        model.complete_turn(session, (_external_tool_schema_without_parameters(),))

    assert gateway_calls == []
    assert not (tmp_path / "prompt_assembly_manifest.json").exists()


@pytest.mark.parametrize(
    "schema",
    (
        {
            "name": "bad tool",
            "parameters": VALID_PARAMETERS,
            "family": "artifact",
            "approval_required": False,
            "executable": True,
        },
        {
            "name": "tool",
            "parameters": [],
            "family": "artifact",
            "approval_required": False,
            "executable": True,
        },
        {
            "name": "tool",
            "parameters": {"type": "array", "items": {"type": "string"}},
            "family": "artifact",
            "approval_required": False,
            "executable": True,
        },
        {
            "name": "mcp_tool",
            "inputSchema": {"type": "object", "additionalProperties": True},
            "family": "mcp",
            "approval_required": False,
            "executable": True,
        },
        {
            "name": "mcp_tool",
            "inputSchema": {"type": "array", "items": {"type": "string"}},
            "family": "mcp",
            "approval_required": False,
            "executable": True,
        },
    ),
)
def test_malformed_names_and_params_do_not_reach_provider_body(tmp_path: Path, schema: dict[str, object]) -> None:
    session = _session(tmp_path)
    model = ProviderToolChoiceModel(api_key="test-token", retry_count=0)

    with pytest.raises(ModelToolChoiceBlocked, match="malformed_tool_schema"):
        model.complete_turn(session, (schema,))


def _session(
    tmp_path: Path,
    *,
    registry: ToolRegistry | None = None,
    agent_id: str = "orchestrator",
) -> AsaAgentSession:
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://provider.invalid/v1",
            "auth_mode": "api_key",
            "api_key_env": "MODEL_TOKEN",
        }
    )
    return AsaAgentSession(
        run_id="provider-schema-run",
        session_id="provider-schema-session",
        agent_id=agent_id,
        user_goal="select a safe tool",
        endpoint=endpoint,
        output_dir=tmp_path,
        registry=registry or default_tool_registry(),
    )


def _is_object_schema(value: object) -> bool:
    return isinstance(value, dict) and value.get("type") == "object" and isinstance(value.get("properties"), dict)


def _tool_schema(*, parameters: dict[str, object] | None) -> dict[str, object]:
    return {
        "name": "artifact_write",
        "boundary": "evidence",
        "family": "artifact",
        "safety": "workspace_write",
        "approval_required": False,
        "executable": True,
        "parameters": parameters,
    }


def _external_tool_schema_without_parameters() -> dict[str, object]:
    return {
        "name": "external_tool",
        "boundary": "external",
        "family": "artifact",
        "safety": "workspace_write",
        "approval_required": False,
        "executable": True,
    }
