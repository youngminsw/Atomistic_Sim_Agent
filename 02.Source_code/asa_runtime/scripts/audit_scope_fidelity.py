from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ALLOWED_RELATIVE_PATHS = frozenset(
    {
        "docs/runtime_spine_contract.md",
        "sim_agent/agents_sdk_runtime/spine_contract.py",
        "sim_agent/agents_sdk_runtime/__init__.py",
        "tests/test_runtime_spine_contract.py",
        "tests/test_gajae_like_gap.py",
        "tests/test_runtime_spine_audit.py",
        "scripts/audit_runtime_spines.py",
        "scripts/audit_plan_completion.py",
        "scripts/audit_scope_fidelity.py",
        ".omo/evidence/task-1-audit-dry-run.json",
        ".omo/evidence/task-1-git-root.json",
    }
)
FORBIDDEN_PATH_FRAGMENTS = ("G:/", "G:\\", "/mnt/g/", "02.Source_code/mss_agent")
RUNTIME_ROOT_PREFIX = "02.Source_code/asa_runtime/"
REPORT_PATH_LIMIT = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    payload = audit_scope_fidelity(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"scope_fidelity_audit_path={out_path}")
    print(f"status={payload['status']}")
    return 0


def audit_scope_fidelity(root: Path) -> dict[str, str | int | list[str] | tuple[str, ...]]:
    git_paths = _git_changed_paths(root)
    changed_paths = _runtime_relative_paths(git_paths)
    out_of_scope = [path for path in changed_paths if path not in ALLOWED_RELATIVE_PATHS]
    forbidden_runtime = [path for path in changed_paths if any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS)]
    forbidden_baseline = [path for path in git_paths if any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS)]
    status = "clean" if not out_of_scope and not forbidden_runtime else "scope_review_required"
    return {
        "status": status,
        "root": str(root),
        "changed_path_count": len(changed_paths),
        "changed_paths": changed_paths[:REPORT_PATH_LIMIT],
        "out_of_scope_count": len(out_of_scope),
        "out_of_scope_paths": out_of_scope[:REPORT_PATH_LIMIT],
        "forbidden_runtime_path_count": len(forbidden_runtime),
        "forbidden_paths": forbidden_runtime[:REPORT_PATH_LIMIT],
        "outside_runtime_status_count": len(git_paths) - len(changed_paths),
        "forbidden_baseline_status_count": len(forbidden_baseline),
        "path_report_limit": REPORT_PATH_LIMIT,
        "allowed_relative_paths": sorted(ALLOWED_RELATIVE_PATHS),
        "forbidden_path_fragments": FORBIDDEN_PATH_FRAGMENTS,
    }


def _git_changed_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        paths.append(line[3:])
    return paths


def _runtime_relative_paths(git_paths: list[str]) -> list[str]:
    runtime_paths: list[str] = []
    for path in git_paths:
        normalized = path.strip('"')
        if normalized.startswith(RUNTIME_ROOT_PREFIX):
            runtime_paths.append(normalized[len(RUNTIME_ROOT_PREFIX) :])
    return runtime_paths


if __name__ == "__main__":
    raise SystemExit(main())
