from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


FIXTURE_ROOT = Path("tests") / "fixtures"


@dataclass(frozen=True, slots=True)
class FixtureSpec:
    path: str
    kind: str


@dataclass(frozen=True, slots=True)
class FixtureReport:
    ok: bool
    checked: list[str]
    missing: list[str]
    invalid: list[str]
    loaded_kinds: set[str]


REQUIRED_FIXTURES = (
    FixtureSpec("requests/valid_ar_si_pr_hole.json", "json"),
    FixtureSpec("requests/missing_iedf.json", "json"),
    FixtureSpec("requests/direct_openai_valid.json", "json"),
    FixtureSpec("requests/openclaw_provider_bad_base_url.json", "json"),
    FixtureSpec("requests/missing_recipe.json", "json"),
    FixtureSpec("requests/no_artifacts_complete_request.json", "json"),
    FixtureSpec("requests/2d_image_missing_distribution.json", "json"),
    FixtureSpec("requests/ar_on_unknown_material.json", "json"),
    FixtureSpec("config/openclaw_valid.json", "json"),
    FixtureSpec("config/direct_openai_valid.json", "json"),
    FixtureSpec("config/openclaw_missing_model.json", "json"),
    FixtureSpec("config/openclaw_helper_valid.json", "json"),
    FixtureSpec("config/openclaw_helper_physics_invalid.json", "json"),
    FixtureSpec("config/openclaw_oauth_refresh_valid.json", "json"),
    FixtureSpec("geometry/pr_trench.png", "image"),
    FixtureSpec("geometry/pr_hole_mask.png", "image"),
    FixtureSpec("geometry/pr_trench.stl", "mesh"),
    FixtureSpec("geometry/pr_hole.stl", "mesh"),
    FixtureSpec("materials/si_crystal_descriptor.json", "json"),
    FixtureSpec("materials/si_amorphous_descriptor.json", "json"),
    FixtureSpec("md_events/md_events_small.jsonl", "jsonl"),
    FixtureSpec("md_logs/success_lammps.log", "log"),
    FixtureSpec("md_logs/failed_lammps.log", "log"),
    FixtureSpec("kernels/offline_ar_si_kernel.json", "json"),
    FixtureSpec("scenes/pr_hole_scene.json", "json"),
)


def fixture_root(source_root: Path) -> Path:
    return source_root / FIXTURE_ROOT


def resolve_fixture(source_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = fixture_root(source_root)
    rooted = root / candidate
    if rooted.exists():
        return rooted
    return source_root / path


def load_fixture(path: Path, kind: str) -> object:
    if kind == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if kind == "jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if kind == "image":
        header = path.read_bytes()[:8]
        if header != b"\x89PNG\r\n\x1a\n":
            raise ValueError("expected PNG image fixture")
        return {"format": "png", "bytes": path.stat().st_size}
    if kind == "mesh":
        first_line = path.read_text(encoding="utf-8").splitlines()[0].strip().lower()
        if not first_line.startswith("solid"):
            raise ValueError("expected ASCII STL mesh fixture")
        return {"format": "ascii-stl"}
    if kind == "log":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("expected non-empty log fixture")
        return {"lines": len(text.splitlines())}
    raise ValueError(f"unknown fixture kind: {kind}")


def validate_required_fixtures(source_root: Path, specs: tuple[FixtureSpec, ...] = REQUIRED_FIXTURES) -> FixtureReport:
    checked: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    loaded_kinds: set[str] = set()

    for spec in specs:
        path = resolve_fixture(source_root, spec.path)
        checked.append(spec.path)
        if not path.exists():
            missing.append(spec.path)
            continue
        try:
            load_fixture(path, spec.kind)
        except (OSError, ValueError, json.JSONDecodeError, IndexError) as exc:
            invalid.append(f"{spec.path}: {exc}")
            continue
        loaded_kinds.add(spec.kind)

    return FixtureReport(
        ok=not missing and not invalid,
        checked=checked,
        missing=missing,
        invalid=invalid,
        loaded_kinds=loaded_kinds,
    )
