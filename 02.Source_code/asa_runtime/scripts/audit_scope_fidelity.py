from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Final


ALLOWED_RELATIVE_PATHS: Final = frozenset(
    {
        "docs/runtime_spine_contract.md",
        "pyproject.toml",
        "sim_agent/agents_sdk_runtime/spine_contract.py",
        "sim_agent/agents_sdk_runtime/__init__.py",
        "tests/test_runtime_spine_contract.py",
        "tests/test_gajae_like_gap.py",
        "tests/test_runtime_spine_audit.py",
        "tests/test_runtime_spine_audit_hardening.py",
        "scripts/audit_runtime_spines.py",
        "scripts/audit_plan_completion.py",
        "scripts/audit_scope_fidelity.py",
        "scripts/render_remote_worker_plan.py",
        ".omo/evidence/task-1-audit-dry-run.json",
        ".omo/evidence/task-1-git-root.json",
    }
)
APPROVED_HARDENING_PREFIXES: Final = (
    "model_gateway/src/",
    "model_gateway/test/",
    "scripts/audit_",
    "sim_agent/agent_harness/",
    "sim_agent/agent_runtime/",
    "sim_agent/agents_sdk_runtime/",
    "sim_agent/cli/",
    "sim_agent/compute/",
    "sim_agent/knowledge/",
    "sim_agent/llm_endpoints/",
    "sim_agent/md/",
    "sim_agent/ui/",
    "tests/",
)
DISALLOWED_RUNTIME_PREFIXES: Final = (
    "prompts/",
    "sim_agent/agents/",
    "sim_agent/domain_agents/",
)
IGNORED_OUTSIDE_RUNTIME_PREFIXES: Final = (
    ".asa/",
    ".omc/",
    ".omo/",
    ".omx/",
)
FORBIDDEN_PATH_FRAGMENTS: Final = ("G:/", "G:\\", "/mnt/g/", "02.Source_code/mss_agent")
RUNTIME_ROOT_PREFIX: Final = "02.Source_code/asa_runtime/"
REPORT_PATH_LIMIT: Final = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    payload = audit_scope_fidelity(root)
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"scope_fidelity_audit_path={out_path}")
    print(f"status={payload['status']}")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def audit_scope_fidelity(root: Path) -> dict[str, str | int | list[str] | tuple[str, ...]]:
    git_paths = _git_changed_paths(root)
    return audit_scope_paths(root, git_paths)


def audit_scope_paths(root: Path, git_paths: list[str]) -> dict[str, str | int | list[str] | tuple[str, ...]]:
    changed_paths = _runtime_relative_paths(git_paths)
    out_of_scope = [path for path in changed_paths if not _approved_runtime_hardening_path(path)]
    domain_prompt_paths = [path for path in changed_paths if _is_disallowed_runtime_path(path)]
    forbidden_runtime = [path for path in changed_paths if any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS)]
    forbidden_baseline = [path for path in git_paths if any(fragment in path for fragment in FORBIDDEN_PATH_FRAGMENTS)]
    outside_runtime = _outside_runtime_paths(git_paths)
    outside_runtime_violations = [
        path for path in outside_runtime if not _is_ignored_outside_runtime_path(path)
    ]
    status = (
        "clean"
        if not out_of_scope
        and not domain_prompt_paths
        and not forbidden_runtime
        and not outside_runtime_violations
        else "scope_review_required"
    )
    return {
        "status": status,
        "root": str(root),
        "changed_path_count": len(changed_paths),
        "changed_paths": changed_paths[:REPORT_PATH_LIMIT],
        "out_of_scope_count": len(out_of_scope),
        "out_of_scope_paths": out_of_scope[:REPORT_PATH_LIMIT],
        "domain_prompt_path_count": len(domain_prompt_paths),
        "domain_prompt_paths": domain_prompt_paths[:REPORT_PATH_LIMIT],
        "forbidden_runtime_path_count": len(forbidden_runtime),
        "forbidden_paths": forbidden_runtime[:REPORT_PATH_LIMIT],
        "outside_runtime_status_count": len(outside_runtime),
        "outside_runtime_paths": outside_runtime[:REPORT_PATH_LIMIT],
        "outside_runtime_violation_count": len(outside_runtime_violations),
        "outside_runtime_violations": outside_runtime_violations[:REPORT_PATH_LIMIT],
        "forbidden_baseline_status_count": len(forbidden_baseline),
        "path_report_limit": REPORT_PATH_LIMIT,
        "allowed_relative_paths": sorted(ALLOWED_RELATIVE_PATHS),
        "approved_hardening_prefixes": APPROVED_HARDENING_PREFIXES,
        "disallowed_runtime_prefixes": DISALLOWED_RUNTIME_PREFIXES,
        "ignored_outside_runtime_prefixes": IGNORED_OUTSIDE_RUNTIME_PREFIXES,
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


def _outside_runtime_paths(git_paths: list[str]) -> list[str]:
    return [
        path.strip('"')
        for path in git_paths
        if not path.strip('"').startswith(RUNTIME_ROOT_PREFIX)
    ]


def _approved_runtime_hardening_path(path: str) -> bool:
    return path in ALLOWED_RELATIVE_PATHS or any(
        path.startswith(prefix) for prefix in APPROVED_HARDENING_PREFIXES
    )


def _is_disallowed_runtime_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in DISALLOWED_RUNTIME_PREFIXES)


def _is_ignored_outside_runtime_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in IGNORED_OUTSIDE_RUNTIME_PREFIXES)


if __name__ == "__main__":
    raise SystemExit(main())
