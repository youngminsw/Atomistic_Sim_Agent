from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.compute import (
    ComputePolicyError,
    build_lammps_execution_job_from_md_campaign_job,
    build_md_campaign_job_from_plan_dir,
    build_md_postprocess_job_from_lammps_execution_job,
    build_worker_bundle,
    job_bundle_payload,
    stage_compute_source_payload,
    worker_bundle_payload,
)
from sim_agent.schemas._parse import JsonMap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--environment-name", required=True)
    parser.add_argument("--remote-user", default="swym")
    parser.add_argument("--job-out", required=True)
    parser.add_argument("--worker-out", required=True)
    parser.add_argument("--lammps-job-out")
    parser.add_argument("--lammps-worker-out")
    parser.add_argument("--postprocess-job-out")
    parser.add_argument("--postprocess-worker-out")
    args = parser.parse_args()

    try:
        plan_dir = Path(args.plan_dir)
        source_payload = stage_compute_source_payload(SOURCE_ROOT, plan_dir)
        job = build_md_campaign_job_from_plan_dir(plan_dir, args.environment_name)
        bundle = build_worker_bundle(args.host, job, remote_user=args.remote_user)
        _write_json(Path(args.job_out), job_bundle_payload(job))
        _write_json(Path(args.worker_out), worker_bundle_payload(bundle))
        lammps_job = build_lammps_execution_job_from_md_campaign_job(job)
        if args.lammps_job_out or args.lammps_worker_out:
            if not args.lammps_job_out or not args.lammps_worker_out:
                raise ComputePolicyError("lammps_job_and_worker_outputs_required")
            lammps_bundle = build_worker_bundle(
                args.host,
                lammps_job,
                remote_user=args.remote_user,
            )
            _write_json(Path(args.lammps_job_out), job_bundle_payload(lammps_job))
            _write_json(Path(args.lammps_worker_out), worker_bundle_payload(lammps_bundle))
        postprocess_job = build_md_postprocess_job_from_lammps_execution_job(
            lammps_job,
            material_id="Si",
        )
        if args.postprocess_job_out or args.postprocess_worker_out:
            if not args.postprocess_job_out or not args.postprocess_worker_out:
                raise ComputePolicyError("postprocess_job_and_worker_outputs_required")
            postprocess_bundle = build_worker_bundle(
                args.host,
                postprocess_job,
                remote_user=args.remote_user,
            )
            _write_json(Path(args.postprocess_job_out), job_bundle_payload(postprocess_job))
            _write_json(
                Path(args.postprocess_worker_out),
                worker_bundle_payload(postprocess_bundle),
            )
    except (ComputePolicyError, OSError) as exc:
        print(str(exc))
        return 1

    print("md_campaign_worker_bundle_ok=true")
    print(f"job_id={job.job_id}")
    print(f"host={bundle.host_alias}")
    print(f"job_path={args.job_out}")
    print(f"worker_path={args.worker_out}")
    print(f"source_payload_path={source_payload.archive_path}")
    if args.lammps_job_out and args.lammps_worker_out:
        print(f"lammps_job_id={lammps_job.job_id}")
        print(f"lammps_job_path={args.lammps_job_out}")
        print(f"lammps_worker_path={args.lammps_worker_out}")
    if args.postprocess_job_out and args.postprocess_worker_out:
        print(f"postprocess_job_id={postprocess_job.job_id}")
        print(f"postprocess_job_path={args.postprocess_job_out}")
        print(f"postprocess_worker_path={args.postprocess_worker_out}")
    return 0


def _write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
