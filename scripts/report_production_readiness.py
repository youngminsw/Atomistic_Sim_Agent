from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.production_readiness import assess_production_readiness


def main() -> int:
    parser = argparse.ArgumentParser(description="Report production go-live readiness from agent evidence.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--model-endpoint-smoke-report")
    parser.add_argument("--graphdb-ingest-report", dest="graphdb_report")
    parser.add_argument("--graphdb-write-report", dest="graphdb_report")
    parser.add_argument("--feature-qa-report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = assess_production_readiness(
        ledger_path=Path(args.ledger),
        model_endpoint_smoke_report_path=_optional_path(args.model_endpoint_smoke_report),
        graphdb_ingest_report_path=_optional_path(args.graphdb_report),
        feature_qa_report_path=_optional_path(args.feature_qa_report),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"production_ready={str(report.production_ready).lower()}")
    print(f"hard_blockers={','.join(report.payload['hard_blockers'])}")
    print(f"user_action_required={str(bool(report.payload['user_actions'])).lower()}")
    print(f"user_actions={','.join(report.payload['user_actions'])}")
    print(f"agent_actions={','.join(report.payload['agent_actions'])}")
    print(f"readiness_report_path={out_path}")
    return 0 if report.production_ready else 1


def _optional_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw)


if __name__ == "__main__":
    raise SystemExit(main())
