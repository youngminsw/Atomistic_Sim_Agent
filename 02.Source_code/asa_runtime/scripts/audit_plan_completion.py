from __future__ import annotations

import argparse
import json
from pathlib import Path


PLAN_PATH = Path(".omo/plans/asa-runtime-spine-gap-closure.md")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--todo", default="1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    payload = audit_plan_completion(root, args.todo)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"plan_completion_audit_path={out_path}")
    print(f"todo_found={str(payload['todo_found']).lower()}")
    return 0


def audit_plan_completion(root: Path, todo_id: str) -> dict[str, str | bool]:
    plan_path = root / PLAN_PATH
    text = plan_path.read_text(encoding="utf-8") if plan_path.is_file() else ""
    prefix = f"- [ ] {todo_id}. "
    todo_line = next((line for line in text.splitlines() if line.startswith(prefix)), "")
    return {
        "status": "plan_audited",
        "root": str(root),
        "plan_path": str(plan_path),
        "todo_id": todo_id,
        "todo_found": bool(todo_line),
        "todo_line": todo_line,
    }


if __name__ == "__main__":
    raise SystemExit(main())
