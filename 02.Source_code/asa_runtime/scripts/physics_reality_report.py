from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import allowed_compute_hosts
from sim_agent.knowledge import GraphDBGateError, GraphDBGateRequest, GraphDBMode, build_graphdb_gate_plan, seeded_provenance_registry
from sim_agent.llm_endpoints import ModelProviderConfig, ProviderConfigPolicyError
from sim_agent.llm_endpoints.config import PRIMARY_MODEL, PRIMARY_REASONING
from sim_agent.md_campaign import plan_active_learning_run
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str, require


@dataclass(frozen=True, slots=True)
class RealityReport:
    run_dir: Path
    manifest: JsonMap
    transport_field: JsonMap
    hit_history: JsonMap
    click_index: JsonMap

    @property
    def run_status(self) -> str:
        return as_str(require(self.manifest, "run_status"), "run_status")

    @property
    def mode(self) -> str:
        return as_str(require(self.manifest, "mode"), "mode")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    report = _load_report(Path(args.run))
    for line in _summary_lines(report):
        print(line)
    return 0


def _load_report(run_dir: Path) -> RealityReport:
    return RealityReport(
        run_dir=run_dir,
        manifest=_json_map(run_dir / "manifest.json", "manifest"),
        transport_field=_json_map(run_dir / "transport_field.json", "transport_field"),
        hit_history=_json_map(run_dir / "hit_history.json", "hit_history"),
        click_index=_json_map(run_dir / "click_index.json", "click_index"),
    )


def _json_map(path: Path, field: str) -> JsonMap:
    return as_mapping(json.loads(path.read_text(encoding="utf-8")), field)


def _summary_lines(report: RealityReport) -> tuple[str, ...]:
    return (
        f"run_status={report.run_status}",
        f"mode={report.mode}",
        f"artifact_complete={str(_artifact_complete(report)).lower()}",
        "md_derived=fixture_md_events",
        f"surrogate_derived={_surrogate_backend(report)}",
        f"surrogate_gate_accepted={str(_surrogate_gate_accepted(report)).lower()}",
        f"surrogate_registered_for_feature_scale={str(_surrogate_registered(report)).lower()}",
        f"literature_derived={_literature_source_label()}",
        "semi_empirical=transport_sampling_and_level_set_fixture",
        "assumptions=offline_fixture_not_physical_validation",
        "physical_validation_claimed=false",
        "first_scope_chemistry=false",
        "missing_md_coverage=true",
        f"model_provider_config={str(_model_provider_config_ok()).lower()}",
        f"gpu_host_allowlist={str(_gpu_host_allowlist_ok()).lower()}",
        f"neo4j_write_gate={str(_neo4j_write_gate_ok()).lower()}",
        "continuous_bombardment_default=true",
        f"controlled_event_probe_active_learning_only={str(_controlled_probe_policy_ok()).lower()}",
    )


def _artifact_complete(report: RealityReport) -> bool:
    artifacts = as_mapping(require(report.manifest, "artifacts"), "artifacts")
    required_keys = (
        "profile_timeline",
        "transport_field",
        "hit_history",
        "click_index",
        "surrogate_model",
        "surrogate_training_gate",
        "surrogate_model_manifest",
    )
    for key in required_keys:
        artifact_name = as_str(require(artifacts, key), f"artifacts.{key}")
        if not (report.run_dir / artifact_name).exists():
            return False
    click_items = as_sequence(require(report.click_index, "clicks"), "clicks")
    hit_items = as_sequence(require(report.hit_history, "hits"), "hits")
    return bool(click_items) and bool(hit_items)


def _surrogate_backend(report: RealityReport) -> str:
    surrogate = _surrogate(report)
    return as_str(surrogate.get("training_backend", "missing_surrogate"), "surrogate.training_backend")


def _surrogate_gate_accepted(report: RealityReport) -> bool:
    return _bool_field(_surrogate(report), "quality_gate_accepted")


def _surrogate_registered(report: RealityReport) -> bool:
    return _bool_field(_surrogate(report), "registered_for_feature_scale")


def _surrogate(report: RealityReport) -> JsonMap:
    return as_mapping(require(report.manifest, "surrogate"), "surrogate")


def _bool_field(payload: JsonMap, field: str) -> bool:
    value = payload.get(field)
    if isinstance(value, bool):
        return value
    return False


def _literature_source_label() -> str:
    registry = seeded_provenance_registry()
    records = registry.list_by_tag("level_set") + registry.list_by_tag("feature_scale")
    if records:
        return "seeded_provenance_registry"
    return "missing_seeded_provenance"


def _model_provider_config_ok() -> bool:
    config = ModelProviderConfig.from_mapping(
        _json_map(SOURCE_ROOT / "tests" / "fixtures" / "config" / "openclaw_valid.json", "model_provider")
    )
    direct_openai = ModelProviderConfig.from_mapping(
        _json_map(SOURCE_ROOT / "tests" / "fixtures" / "config" / "direct_openai_valid.json", "model_provider")
    )
    rejected_bad_openclaw_base_url = False
    try:
        ModelProviderConfig.from_mapping(
            {
                "provider": "openclaw",
                "model": PRIMARY_MODEL,
                "reasoning_effort": PRIMARY_REASONING,
                "base_url": "https://api.openai.com/v1",
            }
        )
    except ProviderConfigPolicyError:
        rejected_bad_openclaw_base_url = True
    return (
        config.provider == "openclaw"
        and config.model == PRIMARY_MODEL
        and config.reasoning_effort == PRIMARY_REASONING
        and direct_openai.provider == "openai"
        and rejected_bad_openclaw_base_url
    )


def _gpu_host_allowlist_ok() -> bool:
    required_gpu_hosts = frozenset(("gpu-5090", "blackwell-rtxpro", "gpu-ada", "4090-gpu-ws", "ws-gpu"))
    hosts = allowed_compute_hosts()
    return required_gpu_hosts <= frozenset(hosts) and "unconfigured-gpu" not in hosts


def _neo4j_write_gate_ok() -> bool:
    dry_run = build_graphdb_gate_plan(
        GraphDBGateRequest(mode=GraphDBMode.DRY_RUN, user_db_approval=False, existing_database_names=())
    )
    blocked_without_approval = False
    try:
        build_graphdb_gate_plan(
            GraphDBGateRequest(mode=GraphDBMode.ATTEMPT_WRITE, user_db_approval=False, existing_database_names=())
        )
    except GraphDBGateError:
        blocked_without_approval = True
    return not dry_run.neo4j_write_enabled and blocked_without_approval


def _controlled_probe_policy_ok() -> bool:
    plan = plan_active_learning_run(SOURCE_ROOT / "tests" / "fixtures" / "runs" / "high_uncertainty_ar_si")
    return plan.controlled_event_probe_allowed and all(request.protocol == "controlled_event_probe" for request in plan.requests)


if __name__ == "__main__":
    raise SystemExit(main())
