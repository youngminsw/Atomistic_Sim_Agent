from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_apply_graphdb_import_bundle_cli_writes_blocked_json_report(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "graphdb_write_report.json"
    export = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "export_graphdb_import_bundle.py"),
            "--dry-run",
            "--existing-db",
            "neo4j",
            "--sync-run-id",
            "cli-product-graphdb-json",
            "--out",
            str(bundle_dir),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert export.returncode == 0, export.stdout + export.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "apply_graphdb_import_bundle.py"),
            "--bundle-dir",
            str(bundle_dir),
            "--database-name",
            "atomistic_sim_agent_knowledge",
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.returncode == 2
    assert "graphdb_write_status=blocked" in result.stdout
    assert report["applied"] is False
    assert report["status"] == "blocked"
    assert report["blocker_reasons"] == ["user_db_approval_required"]
    assert report["database_name"] == "atomistic_sim_agent_knowledge"
