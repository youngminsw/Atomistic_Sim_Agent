from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_required_fixture_inventory_loads_all_declared_types() -> None:
    from sim_agent.testing.fixture_inventory import REQUIRED_FIXTURES, validate_required_fixtures

    report = validate_required_fixtures(SOURCE_ROOT)

    assert {item.kind for item in REQUIRED_FIXTURES} >= {"json", "jsonl", "image", "mesh", "log"}
    assert report.ok is True
    assert report.missing == []
    assert report.invalid == []
    assert report.loaded_kinds >= {"json", "jsonl", "image", "mesh", "log"}


def test_fixture_inventory_cli_reports_success() -> None:
    result = subprocess.run(
        [sys.executable, str(SOURCE_ROOT / "scripts" / "list_fixtures.py"), "--assert-required"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "fixtures_ok=true" in result.stdout
    assert "missing_count=0" in result.stdout
    assert "invalid_count=0" in result.stdout


def test_fixture_inventory_cli_fails_for_missing_fixture() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SOURCE_ROOT / "scripts" / "list_fixtures.py"),
            "--fixture",
            "does_not_exist.json",
            "--assert-required",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "fixture_not_found" in result.stdout
