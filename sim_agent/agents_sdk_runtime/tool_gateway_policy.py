from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class ToolGatewayPolicy:
    policy_id: str
    mode: str
    provider: str
    gateway_request_id: str
    plan: tuple[tuple[str, JsonMap], ...]


DEFAULT_TOOL_GATEWAY_POLICY: Final = ToolGatewayPolicy(
    policy_id="local-tool-gateway-smoke-v1",
    mode="local_smoke",
    provider="local_gateway",
    gateway_request_id="fake-local-tool-gateway",
    plan=(
        ("bash_process", {"argv": ("python3", "-c", "print('gateway-tool-ok')")}),
        ("graphdb_dry_run", {"database_name": "atomistic_sim_agent_knowledge"}),
    ),
)
