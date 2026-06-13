from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_agent_cli_writes_planned_run_ledger(tmp_path: Path) -> None:
    output_dir = tmp_path / "agent-ledger"

    result = subprocess.run(
        _agent_cli_args(output_dir),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_run_ledger_path=" in result.stdout
    assert ledger["request_id"] == "cli_ar_si_amorphous_hole"
    assert ledger["overall_status"] == "md_action_required"
    assert ledger["model_provider"]["provider"] == "openclaw"
    assert ledger["model_provider"]["model"] == "gpt-5.5"
    assert "agent_plan" in ledger["pipeline_stages"]
    assert ledger["artifact_paths"]["md_campaign_plan_path"].endswith("md_campaign_plan.json")
    assert ledger["artifact_paths"]["amorphous_structure_prep_manifest_path"].endswith(
        "amorphous_structure_prep_manifest.json"
    )
    assert ledger["artifact_paths"]["amorphous_structure_source_path"].endswith(
        "amorphous_structure_source.json"
    )
    assert ledger["artifact_paths"]["amorphous_structure_prep_job_path"].endswith(
        "amorphous_structure_prep_job.json"
    )
    assert ledger["artifact_paths"]["amorphous_structure_prep_worker_path"].endswith(
        "amorphous_structure_prep_worker_bundle.json"
    )
    assert ledger["artifact_paths"]["amorphous_structure_prep_remote_plan_path"].endswith(
        "amorphous_structure_prep_remote_plan.json"
    )
    assert ledger["artifact_paths"]["graphdb_agent_report_path"].endswith(
        "research_graphdb_agent.json"
    )
    assert ledger["artifact_paths"]["graphdb_import_bundle_dir"].endswith(
        "research_graphdb"
    )
    assert ledger["artifact_paths"]["graphdb_ingest_report_path"].endswith(
        "ingest_report.json"
    )
    assert ledger["graphdb"]["status"] == "ready"
    assert ledger["graphdb"]["ingest_accepted"] is True
    assert ledger["surrogate"]["training_gate_present"] is False
    assert ledger["surrogate"]["training_gate_accepted"] is None
    assert ledger["qa"]["agent_id"] == "qa_agent"
    assert ledger["qa"]["evidence_scope"] == "planning_ledger_gate"
    assert ledger["qa"]["status"] == "blocked"
    assert "amorphous_lammps_structure_source_required" in ledger["qa"]["hard_blockers"]
    assert "level_set_profile_timeline" in ledger["qa"]["required_evidence"]
    assert "amorphous_structure_prep_worker_bundle_written" in ledger["evidence"]
    assert "graphdb_import_bundle_written" in ledger["evidence"]


def test_agent_cli_uses_5090_inventory_default_for_remote_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "agent-ledger-inventory"

    result = subprocess.run(
        _agent_cli_args(output_dir),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "remote_chain_manifest.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger["artifact_paths"]["remote_execution_manifest_path"].endswith(
        "remote_chain_manifest.json"
    )
    assert manifest["ssh_target"] == "swym@10.24.12.85"
    assert manifest["ssh_port"] == 55555


def test_agent_cli_records_user_configured_model_endpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "agent-ledger-model"

    result = subprocess.run(
        _agent_cli_args(output_dir)
        + [
            "--model-provider",
            "openai",
            "--model-name",
            "gpt-5.5",
            "--model-base-url",
            "https://api.openai.com/v1",
            "--model-auth-mode",
            "api_key",
            "--model-api-key-env",
            "OPENAI_API_KEY",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    model = ledger["model_provider"]
    assert result.returncode == 0, result.stdout + result.stderr
    assert model["provider"] == "openai"
    assert model["base_url"] == "https://api.openai.com/v1"
    assert model["auth_mode"] == "api_key"
    assert model["api_key_env"] == "OPENAI_API_KEY"


def test_agent_cli_records_canonical_gateway_auth_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "agent-ledger-gateway"

    result = subprocess.run(
        _agent_cli_args(output_dir)
        + [
            "--model-provider",
            "oauth_gateway",
            "--model-name",
            "gpt-5.5",
            "--model-base-url",
            "http://127.0.0.1:8787/v1",
            "--model-auth-mode",
            "gateway",
            "--model-api-key-env",
            "MODEL_GATEWAY_TOKEN",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    model = ledger["model_provider"]
    assert result.returncode == 0, result.stdout + result.stderr
    assert model["provider"] == "oauth_gateway"
    assert model["base_url"] == "http://127.0.0.1:8787/v1"
    assert model["auth_mode"] == "gateway"
    assert model["api_key_env"] == "MODEL_GATEWAY_TOKEN"


def test_agent_cli_run_ledger_records_remote_chain_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "agent-ledger-remote"

    result = subprocess.run(
        _agent_cli_args(output_dir)
        + [
            "--ssh-target",
            "swym@10.24.12.85",
            "--ssh-port",
            "55555",
            "--run-remote-chain",
            "--remote-run-timeout-s",
            "10",
        ],
        cwd=PROJECT_ROOT,
        env=_env_with_failing_ssh(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    remote = ledger["remote"]
    assert result.returncode == 1
    assert ledger["overall_status"] == "remote_failed"
    assert remote["chain_status"] == "remote_chain_failed"
    assert "remote_chain_command_failed" in remote["chain_blockers"]
    assert ledger["qa"]["status"] == "blocked"
    assert "remote_chain_command_failed" in ledger["qa"]["hard_blockers"]
    assert "remote_chain_result_path" in ledger["artifact_paths"]


def test_agent_cli_run_ledger_records_surrogate_training_gate_action(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "agent-ledger-surrogate"
    gate_report = tmp_path / "surrogate_training_gate.json"
    gate_report.write_text(
        json.dumps(
            {
                "accepted": False,
                "decision": "active_learning_required",
                "blockers": ["high_uncertainty_fraction_too_high"],
                "evidence": ["feature_space_coverage_sufficient"],
                "next_actions": ["plan_active_learning_md", "rerun_surrogate_training_gate"],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        _agent_cli_args(output_dir)
        + [
            "--surrogate-training-gate-report",
            str(gate_report),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    ledger = json.loads((output_dir / "agent_run_ledger.json").read_text(encoding="utf-8"))
    surrogate = ledger["surrogate"]
    assert result.returncode == 1
    assert ledger["overall_status"] == "surrogate_action_required"
    assert surrogate["training_gate_decision"] == "active_learning_required"
    assert surrogate["training_gate_present"] is True
    assert surrogate["training_gate_accepted"] is False
    assert "high_uncertainty_fraction_too_high" in surrogate["training_gate_blockers"]
    assert "plan_active_learning_md" in surrogate["next_actions"]
    assert ledger["qa"]["status"] == "blocked"
    assert "high_uncertainty_fraction_too_high" in ledger["qa"]["hard_blockers"]
    assert "surrogate_training_gate_result_path" in ledger["artifact_paths"]


def _agent_cli_args(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SOURCE_ROOT / "scripts" / "run_agent_cli.py"),
        "--offline",
        "--goal",
        "Plan Ar etching on amorphous Si through the production MD pipeline",
        "--material",
        "Si",
        "--phase",
        "amorphous",
        "--ion",
        "Ar",
        "--feature-type",
        "hole",
        "--energy-range-eV",
        "30:150",
        "--polar-range-deg",
        "0:55",
        "--azimuth-range-deg",
        "0:360",
        "--host",
        "gpu-5090",
        "--environment-name",
        "atomistic-sim-gpu",
        "--output-dir",
        str(output_dir),
    ]


def _env_with_failing_ssh(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    ssh = fake_bin / "ssh"
    ssh.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Connection reset by peer' >&2\n"
        "exit 255\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}{os.pathsep}{env.get('PATH', '')}"
    return env
