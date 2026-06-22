from __future__ import annotations

import sys
import tarfile
from pathlib import Path
from shutil import copytree, rmtree


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_source_payload_includes_only_remote_execution_scripts(tmp_path: Path) -> None:
    from sim_agent.compute import stage_compute_source_payload

    payload = stage_compute_source_payload(SOURCE_ROOT, tmp_path)

    with tarfile.open(payload.archive_path, "r:gz") as archive:
        names = set(archive.getnames())

    assert "02.Source_code/asa_runtime/scripts/probe_worker_capability.py" in names
    assert "02.Source_code/asa_runtime/scripts/prepare_amorphous_structure_job.py" in names
    assert "02.Source_code/asa_runtime/scripts/run_md_campaign_job.py" in names
    assert "02.Source_code/asa_runtime/scripts/run_lammps_execution_plan.py" in names
    assert "02.Source_code/asa_runtime/scripts/postprocess_lammps_execution.py" in names
    assert (
        "02.Source_code/mss_agent/md_agent_window/Reference/force_field_library/"
        "potentials/Si.tersoff"
    ) in names
    assert (
        "02.Source_code/mss_agent/md_agent_window/results/run_Ar_Si_3evts/"
        "Si_periodic.data"
    ) in names
    assert "02.Source_code/asa_runtime/scripts/run_demo.py" not in names
    assert not any(name.endswith(":Zone.Identifier") for name in names)


def test_source_payload_reuses_process_snapshot_after_first_stage(tmp_path: Path) -> None:
    from sim_agent.compute import stage_compute_source_payload

    source_root = tmp_path / "source"
    _copy_payload_inputs(source_root)
    first = stage_compute_source_payload(source_root, tmp_path / "first")
    (source_root / "asa_runtime" / "scripts" / "probe_worker_capability.py").unlink()

    second = stage_compute_source_payload(source_root, tmp_path / "second")

    assert first.manifest_payload["entries"] == second.manifest_payload["entries"]
    assert second.archive_path.exists()


def _copy_payload_inputs(source_root: Path) -> None:
    copytree(SOURCE_ROOT / "sim_agent", source_root / "asa_runtime" / "sim_agent")
    copytree(
        SOURCE_ROOT / "tests" / "fixtures" / "materials",
        source_root / "asa_runtime" / "tests" / "fixtures" / "materials",
    )
    _copy_file(
        SOURCE_ROOT.parent
        / "mss_agent"
        / "md_agent_window"
        / "Reference"
        / "force_field_library"
        / "potentials"
        / "Si.tersoff",
        source_root
        / "mss_agent"
        / "md_agent_window"
        / "Reference"
        / "force_field_library"
        / "potentials"
        / "Si.tersoff",
    )
    _copy_file(
        SOURCE_ROOT.parent
        / "mss_agent"
        / "md_agent_window"
        / "results"
        / "run_Ar_Si_3evts"
        / "Si_periodic.data",
        source_root
        / "mss_agent"
        / "md_agent_window"
        / "results"
        / "run_Ar_Si_3evts"
        / "Si_periodic.data",
    )
    scripts_dir = source_root / "asa_runtime" / "scripts"
    scripts_dir.mkdir(parents=True)
    for script_name in (
        "prepare_amorphous_structure_job.py",
        "probe_worker_capability.py",
        "run_md_campaign_job.py",
        "run_lammps_execution_plan.py",
        "postprocess_lammps_execution.py",
    ):
        (scripts_dir / script_name).write_text(
            (SOURCE_ROOT / "scripts" / script_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    rmtree(source_root / "asa_runtime" / "sim_agent" / "__pycache__", ignore_errors=True)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
