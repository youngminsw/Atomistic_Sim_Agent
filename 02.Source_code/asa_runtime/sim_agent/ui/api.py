from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from sim_agent.provider_registry import provider_ids
from sim_agent.knowledge import (
    GraphDBGateRequest,
    GraphDBMode,
    agent_graph_context_payload,
    build_agent_graph_context,
    build_graphdb_gate_plan,
)
from sim_agent.schemas._parse import JsonMap, as_str

from .controller import (
    ControllerRunRequest,
    ControllerValidation,
    build_offline_runner_command,
    controller_compute_targets,
    validate_controller_request,
)


FixtureName = Literal["pr_hole_3d", "pr_trench_2d"]


@dataclass(frozen=True, slots=True)
class UiApiRoute:
    method: str
    path: str
    description: str


@dataclass(frozen=True, slots=True)
class UiApiStatus:
    static_root: Path
    route_paths: tuple[str, ...]
    offline_fixtures: tuple[FixtureName, ...]
    routes: tuple[UiApiRoute, ...]
    model_providers: tuple[str, ...]
    model_options: tuple[str, ...]
    auth_modes: tuple[str, ...]
    agent_roles: tuple[str, ...]
    compute_targets: tuple[str, ...]
    graphdb_database_name: str
    graphdb_write_requires_approval: bool


@dataclass(frozen=True, slots=True)
class UiApiValidation:
    can_run: bool
    missing_fields: tuple[str, ...]
    compute_target: str
    request: ControllerRunRequest
    runner_command: tuple[str, ...]


def build_ui_api_status() -> UiApiStatus:
    routes = _routes()
    graph_context = build_ui_agent_graph_context()
    return UiApiStatus(
        static_root=_source_root() / "ui",
        route_paths=tuple(route.path for route in routes),
        offline_fixtures=("pr_hole_3d", "pr_trench_2d"),
        routes=routes,
        model_providers=provider_ids(),
        model_options=("gpt-5-codex", "gpt-5.5", "gpt-5.3-codex-spark", "claude-sonnet-4.5", "gemini-3-pro-preview"),
        auth_modes=("api_key", "oauth", "gateway", "none"),
        compute_targets=controller_compute_targets(),
        agent_roles=(
            "orchestrator",
            "research_agent",
            "md_agent",
            "ml_agent",
            "feature_scale_agent",
            "qa_agent",
            "production_gate",
        ),
        graphdb_database_name=as_str(graph_context["database_name"], "graph_context.database_name"),
        graphdb_write_requires_approval=bool(graph_context["write_requires_approval"]),
    )


def build_ui_agent_graph_context() -> JsonMap:
    gate_plan = build_graphdb_gate_plan(
        GraphDBGateRequest(
            mode=GraphDBMode.DRY_RUN,
            user_db_approval=False,
            existing_database_names=(),
        )
    )
    return agent_graph_context_payload(build_agent_graph_context(gate_plan))


def build_offline_fixture_request(
    fixture: FixtureName,
    iedf_ready: bool = True,
    iadf_ready: bool = True,
    compute_target: str = "gpu-5090",
    output_dir: str | None = None,
) -> ControllerRunRequest:
    source_root = _source_root()
    kernel_path = source_root / "tests" / "fixtures" / "kernels" / "offline_ar_si_kernel.json"
    events_path = source_root / "tests" / "fixtures" / "md_events" / "md_events_small.jsonl"
    match fixture:
        case "pr_hole_3d":
            return ControllerRunRequest(
                mode="3d",
                geometry_path=str(source_root / "tests" / "fixtures" / "scenes" / "pr_hole_scene.json"),
                kernel_path=str(kernel_path),
                events_path=str(events_path),
                steps=5,
                ions=8,
                run_id="ui-pr-hole-3d",
                compute_target=compute_target,
                iedf_ready=iedf_ready,
                iadf_ready=iadf_ready,
                output_dir=output_dir,
            )
        case "pr_trench_2d":
            return ControllerRunRequest(
                mode="2d",
                geometry_path=str(source_root / "tests" / "fixtures" / "geometry" / "pr_trench.png"),
                kernel_path=str(kernel_path),
                events_path=str(events_path),
                steps=4,
                ions=8,
                run_id="ui-pr-trench-2d",
                compute_target=compute_target,
                iedf_ready=iedf_ready,
                iadf_ready=iadf_ready,
                output_dir=output_dir,
            )
        case unreachable:
            assert_never(unreachable)


def validate_ui_api_request(request: ControllerRunRequest) -> UiApiValidation:
    validation = validate_controller_request(request)
    command = build_offline_runner_command(validation.request) if validation.can_run else ()
    return _api_validation(validation, command)


def _api_validation(validation: ControllerValidation, command: tuple[str, ...]) -> UiApiValidation:
    return UiApiValidation(
        can_run=validation.can_run,
        missing_fields=validation.missing_fields,
        compute_target=validation.compute_target,
        request=validation.request,
        runner_command=command,
    )


def _routes() -> tuple[UiApiRoute, ...]:
    return (
        UiApiRoute(method="GET", path="/", description="static controller and run-bundle viewer"),
        UiApiRoute(method="GET", path="/api/status", description="controller routes and offline fixtures"),
        UiApiRoute(method="GET", path="/api/runtime/config", description="editable runtime config and compute resources"),
        UiApiRoute(method="POST", path="/api/runtime/config", description="save editable runtime config and compute resources"),
        UiApiRoute(
            method="GET",
            path="/api/knowledge/agent-context",
            description="agent-facing GraphDB retrieval context and write approval boundary",
        ),
        UiApiRoute(method="POST", path="/api/agent/plan", description="plan request and ask for missing inputs"),
        UiApiRoute(method="GET", path="/api/model/auth/status", description="redacted model provider auth status"),
        UiApiRoute(method="POST", path="/api/model/auth/login", description="store model provider credentials"),
        UiApiRoute(method="POST", path="/api/model/gateway/smoke", description="call configured model gateway API"),
        UiApiRoute(
            method="POST",
            path="/api/agent/prepare-md-campaign-worker-bundle",
            description="prepare agent plan artifacts and remote worker bundle",
        ),
        UiApiRoute(method="POST", path="/api/run/offline", description="validate offline run request"),
        UiApiRoute(method="GET", path="/api/click-diagnostics", description="click diagnostic artifact contract"),
    )


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]
