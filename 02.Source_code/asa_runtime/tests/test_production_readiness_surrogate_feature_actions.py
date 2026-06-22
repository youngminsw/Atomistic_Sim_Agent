from __future__ import annotations

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from production_readiness_fixtures import actions_by_id, amorphous_blocked_ledger, string_list
from sim_agent.production_readiness import assess_production_readiness_from_payloads


def test_production_readiness_structures_surrogate_and_feature_actions() -> None:
    report = assess_production_readiness_from_payloads(amorphous_blocked_ledger())
    action_plan = actions_by_id(report.payload)

    surrogate_action = action_plan["train_or_active_learn_surrogate"]
    assert surrogate_action["status"] == "blocked_on_missing_artifacts"
    assert string_list(surrogate_action, "missing_artifacts") == [
        "md_log_path",
        "md_events_path",
        "interaction_kernel_path",
        "surrogate_validation_metrics",
        "surrogate_required_coverage",
    ]
    assert string_list(surrogate_action, "next_actions") == [
        "complete_production_md_chain",
        "postprocess_lammps_execution",
        "assess_surrogate_training_gate",
    ]

    feature_action = action_plan["run_feature_scale_from_accepted_production_surrogate"]
    assert feature_action["status"] == "blocked_on_missing_artifacts"
    assert string_list(feature_action, "missing_artifacts") == [
        "feature_scene_or_image_path",
        "interaction_kernel_path",
        "md_events_path",
        "surrogate_model_manifest_path",
        "feature_scale_settings",
    ]
    assert string_list(feature_action, "next_actions") == [
        "train_or_active_learn_surrogate",
        "prepare_feature_scale_settings",
        "run_feature_scale_from_accepted_production_surrogate",
    ]


def test_production_readiness_builds_surrogate_and_feature_commands_when_ready() -> None:
    ledger = amorphous_blocked_ledger()
    ledger["artifact_paths"] = {
        "md_log_path": "/tmp/run/md/log.lammps",
        "md_events_path": "/tmp/run/md/md_events.jsonl",
        "interaction_kernel_path": "/tmp/run/kernel.json",
        "feature_scene_path": "/tmp/run/scene.json",
        "surrogate_model_manifest_path": "/tmp/run/surrogate_training/surrogate_model_manifest.json",
    }
    ledger["md"] = {
        "production_ready": True,
        "hard_blockers": [],
        "expected_incident_count": 500,
    }
    ledger["surrogate"] = {
        "training_gate_accepted": False,
        "validation_metrics": {
            "validation_event_count": 100,
            "validation_nll": 0.08,
            "deposited_energy_mae_eV": 0.8,
            "sputter_yield_mae": 0.03,
            "reflection_brier_score": 0.015,
            "calibration_error": 0.02,
            "high_uncertainty_fraction": 0.005,
        },
        "required_coverage": {
            "energy_range_eV": "30:150",
            "polar_range_deg": "0:55",
            "azimuth_range_deg": "0:360",
        },
    }
    ledger["feature_scale_settings"] = {"steps": 100, "ions": 50000}

    report = assess_production_readiness_from_payloads(ledger)
    action_plan = actions_by_id(report.payload)

    surrogate_action = action_plan["train_or_active_learn_surrogate"]
    assert surrogate_action["status"] == "ready"
    surrogate_command = string_list(surrogate_action, "command")
    assert surrogate_command[:8] == [
        "python3",
        "02.Source_code/asa_runtime/scripts/train_empirical_mdn_surrogate.py",
        "--log",
        "/tmp/run/md/log.lammps",
        "--events",
        "/tmp/run/md/md_events.jsonl",
        "--kernel",
        "/tmp/run/kernel.json",
    ]
    assert "--expected-events" in surrogate_command
    assert "/tmp/run/surrogate_training/surrogate_model_manifest.json" in string_list(
        surrogate_action, "expected_artifacts"
    )

    feature_action = action_plan["run_feature_scale_from_accepted_production_surrogate"]
    assert feature_action["status"] == "ready"
    assert string_list(feature_action, "command") == [
        "python3",
        "02.Source_code/asa_runtime/scripts/run_offline_simulation.py",
        "--scene",
        "/tmp/run/scene.json",
        "--kernel",
        "/tmp/run/kernel.json",
        "--events",
        "/tmp/run/md/md_events.jsonl",
        "--steps",
        "100",
        "--ions",
        "50000",
        "--out",
        "/tmp/run/feature_scale_production",
        "--run-id",
        "blocked-amorphous-run-feature-scale",
    ]
