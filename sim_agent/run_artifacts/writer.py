from __future__ import annotations

import json
from pathlib import Path

from sim_agent.geometry import PatternGeometry3D
from sim_agent.level_set import ProfileTimeline
from sim_agent.schemas._parse import JsonMap

from .serialize import diagnostics_payload, manifest_payload, timeline_payload
from .types import RunArtifactError, RunBundle


def write_profile_run_bundle(
    output_dir: Path,
    run_id: str,
    geometry: PatternGeometry3D,
    timeline: ProfileTimeline,
    click_points_nm: tuple[tuple[float, float, float], ...],
) -> RunBundle:
    if not click_points_nm:
        raise RunArtifactError("click_points_required")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    timeline_path = output_dir / "timeline.json"
    diagnostics_path = output_dir / "diagnostics.json"
    _write_json(manifest_path, manifest_payload(run_id, geometry, timeline))
    _write_json(timeline_path, timeline_payload(timeline))
    _write_json(diagnostics_path, diagnostics_payload(geometry, timeline, click_points_nm))
    return RunBundle(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        timeline_path=timeline_path,
        diagnostics_path=diagnostics_path,
    )


def _write_json(path: Path, payload: JsonMap) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
