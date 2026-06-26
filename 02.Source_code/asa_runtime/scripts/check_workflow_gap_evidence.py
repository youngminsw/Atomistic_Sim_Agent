#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///
# --- How to run ---
# python3 scripts/check_workflow_gap_evidence.py --manifest ../../.omo/evidence/asa-gajae-workflow-gap-closure/final-manifest.json --evidence-dir ../../.omo/evidence/asa-gajae-workflow-gap-closure --parity tests/fixtures/workflow_parity/gajae-workflow-parity-matrix.json
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.agents_sdk_runtime.workflow_gap_evidence import check_workflow_gap_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ASA workflow gap-closure evidence cannot pass superficially.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    args = parser.parse_args()
    result = check_workflow_gap_evidence(
        manifest_path=args.manifest,
        evidence_dir=args.evidence_dir,
        parity_path=args.parity,
    )
    if result.passed:
        print("workflow_gap_evidence_status=passed")
        return 0
    print("workflow_gap_evidence_status=blocked")
    for blocker in result.blockers:
        print(f"workflow_gap_evidence_blocker={blocker}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
