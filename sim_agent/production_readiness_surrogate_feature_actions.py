from __future__ import annotations

from pathlib import Path

from sim_agent.production_readiness_contract import action_actor
from sim_agent.production_readiness_ledger import (
    SCRIPT_ROOT,
    artifact_dir,
    artifact_path,
    int_text,
    mapping,
    metric_text,
    missing_fields,
    range_text,
    text,
)
from sim_agent.production_readiness_missing_action import missing_action
from sim_agent.schemas._parse import JsonMap


def surrogate_training_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    log_path = artifact_path(ledger, "md_log_path")
    events_path = artifact_path(ledger, "md_events_path")
    kernel_path = artifact_path(ledger, "interaction_kernel_path")
    surrogate = mapping(ledger, "surrogate")
    metrics = mapping(surrogate, "validation_metrics")
    coverage = mapping(surrogate, "required_coverage")
    missing = missing_fields(
        (
            ("md_log_path", log_path),
            ("md_events_path", events_path),
            ("interaction_kernel_path", kernel_path),
        )
    )
    if _missing_surrogate_metrics(metrics):
        missing.append("surrogate_validation_metrics")
    if _missing_surrogate_coverage(coverage):
        missing.append("surrogate_required_coverage")
    if missing:
        return missing_action(action, missing)
    output_dir = str(Path(artifact_dir(ledger)) / "surrogate_training")
    command = [
        "python3",
        f"{SCRIPT_ROOT}/train_empirical_mdn_surrogate.py",
        "--log",
        log_path,
        "--events",
        events_path,
        "--kernel",
        kernel_path,
    ]
    expected_events = int_text(mapping(ledger, "md"), "expected_incident_count")
    if expected_events:
        command.extend(["--expected-events", expected_events])
    command.extend(_surrogate_metric_args(metrics, coverage, output_dir))
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "ready",
        "command": command,
        "expected_artifacts": [
            str(Path(output_dir) / "surrogate_training_gate_report.json"),
            str(Path(output_dir) / "empirical_mdn_model.json"),
            str(Path(output_dir) / "surrogate_model_manifest.json"),
        ],
    }


def feature_scale_action_entry(
    ledger: JsonMap,
    action: str,
    user_actions: list[str],
) -> JsonMap:
    scene_path = artifact_path(ledger, "feature_scene_path")
    image_path = artifact_path(ledger, "feature_image_path")
    kernel_path = artifact_path(ledger, "interaction_kernel_path")
    events_path = artifact_path(ledger, "md_events_path")
    surrogate_manifest_path = artifact_path(ledger, "surrogate_model_manifest_path")
    settings = mapping(ledger, "feature_scale_settings")
    steps = int_text(settings, "steps")
    ions = int_text(settings, "ions")
    missing: list[str] = []
    if not scene_path and not image_path:
        missing.append("feature_scene_or_image_path")
    missing.extend(
        missing_fields(
            (
                ("interaction_kernel_path", kernel_path),
                ("md_events_path", events_path),
                ("surrogate_model_manifest_path", surrogate_manifest_path),
            )
        )
    )
    if not steps or not ions:
        missing.append("feature_scale_settings")
    if missing:
        return missing_action(action, missing)
    output_dir = str(Path(artifact_dir(ledger)) / "feature_scale_production")
    run_id = text(ledger, "run_id") or "feature-scale-production"
    return {
        "action": action,
        "actor": action_actor(action),
        "status": "ready",
        "command": [
            "python3",
            f"{SCRIPT_ROOT}/run_offline_simulation.py",
            "--scene" if scene_path else "--image",
            scene_path or image_path,
            "--kernel",
            kernel_path,
            "--events",
            events_path,
            "--steps",
            steps,
            "--ions",
            ions,
            "--out",
            output_dir,
            "--run-id",
            f"{run_id}-feature-scale",
        ],
        "uses_artifacts": [surrogate_manifest_path],
        "expected_artifacts": [
            str(Path(output_dir) / "profile_timeline.json"),
            str(Path(output_dir) / "qa_report.json"),
        ],
    }


def _surrogate_metric_args(
    metrics: JsonMap,
    coverage: JsonMap,
    output_dir: str,
) -> list[str]:
    return [
        "--validation-event-count",
        metric_text(metrics, "validation_event_count"),
        "--validation-nll",
        metric_text(metrics, "validation_nll"),
        "--deposited-energy-mae-eV",
        metric_text(metrics, "deposited_energy_mae_eV"),
        "--sputter-yield-mae",
        metric_text(metrics, "sputter_yield_mae"),
        "--reflection-brier-score",
        metric_text(metrics, "reflection_brier_score"),
        "--calibration-error",
        metric_text(metrics, "calibration_error"),
        "--high-uncertainty-fraction",
        metric_text(metrics, "high_uncertainty_fraction"),
        "--required-energy-range-eV",
        range_text(coverage, "energy_range_eV"),
        "--required-polar-range-deg",
        range_text(coverage, "polar_range_deg"),
        "--required-azimuth-range-deg",
        range_text(coverage, "azimuth_range_deg"),
        "--out-dir",
        output_dir,
    ]


def _missing_surrogate_metrics(metrics: JsonMap) -> bool:
    return any(
        not metric_text(metrics, field)
        for field in (
            "validation_event_count",
            "validation_nll",
            "deposited_energy_mae_eV",
            "sputter_yield_mae",
            "reflection_brier_score",
            "calibration_error",
            "high_uncertainty_fraction",
        )
    )


def _missing_surrogate_coverage(coverage: JsonMap) -> bool:
    return any(
        not range_text(coverage, field)
        for field in ("energy_range_eV", "polar_range_deg", "azimuth_range_deg")
    )
