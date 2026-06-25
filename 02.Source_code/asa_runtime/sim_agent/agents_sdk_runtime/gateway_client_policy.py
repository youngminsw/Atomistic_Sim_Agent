from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap

from .gateway_client_types import GatewayClientSmokeError
from .roles import AGENT_ROLES


@dataclass(frozen=True, slots=True)
class GatewayAgentPlanPolicy:
    policy_id: str
    default_specialist: str
    default_second_call: str
    allowed_agent_ids: frozenset[str]
    required_call_graph: tuple[str, ...]
    response_contract: JsonMap


DEFAULT_GATEWAY_AGENT_PLAN_POLICY: Final = GatewayAgentPlanPolicy(
    policy_id="production-gateway-smoke-plan-v1",
    default_specialist="research_agent",
    default_second_call="qa_agent",
    allowed_agent_ids=frozenset(("orchestrator", *(role.role_id for role in AGENT_ROLES))),
    required_call_graph=(
        "orchestrator->research_agent",
        "research_agent->qa_agent",
    ),
    response_contract={
        "agent_plan.specialist": "research_agent",
        "agent_plan.second_call": "qa_agent",
    },
)


def gateway_prompt_metadata(policy: GatewayAgentPlanPolicy, request_id: str) -> JsonMap:
    return {
        "simulation_request_id": request_id,
        "gateway_policy_id": policy.policy_id,
        "required_call_graph": list(policy.required_call_graph),
        "response_contract": policy.response_contract,
    }


def gateway_agent_plan(policy: GatewayAgentPlanPolicy, response: JsonMap) -> tuple[str, str]:
    plan = response.get("agent_plan")
    if plan is None:
        return policy.default_specialist, policy.default_second_call
    if isinstance(plan, dict):
        specialist = plan.get("specialist")
        second_call = plan.get("second_call")
        if isinstance(specialist, str) and isinstance(second_call, str):
            if (
                gateway_agent_allowed(policy, specialist)
                and gateway_agent_allowed(policy, second_call)
                and _matches_response_contract(policy, specialist, second_call)
            ):
                return specialist, second_call
    raise GatewayClientSmokeError("gateway_agent_plan_invalid")


def gateway_agent_allowed(policy: GatewayAgentPlanPolicy, agent_id: str) -> bool:
    return agent_id in policy.allowed_agent_ids


def _matches_response_contract(
    policy: GatewayAgentPlanPolicy,
    specialist: str,
    second_call: str,
) -> bool:
    return (
        specialist == _contract_value(policy, "agent_plan.specialist", policy.default_specialist)
        and second_call == _contract_value(policy, "agent_plan.second_call", policy.default_second_call)
    )


def _contract_value(policy: GatewayAgentPlanPolicy, field: str, fallback: str) -> str:
    value = policy.response_contract.get(field)
    return value if isinstance(value, str) and value else fallback
