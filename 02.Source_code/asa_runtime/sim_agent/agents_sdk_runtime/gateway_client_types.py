from __future__ import annotations

from dataclasses import dataclass


PRODUCTION_GATEWAY_SMOKE_LEDGER_NAME = "production_gateway_smoke_ledger.json"


@dataclass(frozen=True, slots=True)
class GatewaySessionEvent:
    at: float
    event_type: str
    agent_id: str
    summary: str
    task_id: str
    peer: str | None = None
    artifact_ref: str | None = None


@dataclass(frozen=True, slots=True)
class GatewayClientSmokeResult:
    production_smoke: bool
    offline: bool
    fake_gateway_model: bool
    provider: str
    model: str
    auth_mode: str
    base_url: str
    gateway_policy_id: str
    gateway_health_ok: bool
    models_count: int
    endpoint_status: int | None
    gateway_request_id: str | None
    blockers: tuple[str, ...]
    session_files: tuple[str, ...]
    final_output: str

    @property
    def ok(self) -> bool:
        return (
            not self.blockers
            and self.gateway_health_ok
            and self.models_count > 0
            and self.endpoint_status == 200
            and bool(self.gateway_request_id)
        )


class GatewayClientSmokeError(RuntimeError):
    pass
