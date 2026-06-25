from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ASA_PROJECT_ROOT_ENV: Final = "ASA_PROJECT_ROOT"
PROJECT_MARKERS: Final = ("pyproject.toml", "AGENTS.md", ".asa", ".git")
PROJECT_GUIDANCE_FILES: Final = ("AGENTS.md", ".asa/AGENTS.md", ".asa/SKILLS.md")
MAX_GUIDANCE_BYTES: Final = 64_000


class ProjectLayoutError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProjectStateLayout:
    project_root: Path
    asa_root: Path
    session_root: Path
    evidence_root: Path
    skills_root: Path
    runs_root: Path
    agents_file: Path
    skills_file: Path


def discover_project_root(start: Path | None = None) -> Path:
    explicit = _env_project_root()
    if explicit is not None:
        return explicit
    current = (start or Path.cwd()).expanduser().resolve()
    for candidate in (current, *current.parents):
        if _has_project_marker(candidate):
            return candidate
    return Path(__file__).resolve().parents[1]


def ensure_project_state_layout(project_root: Path | None = None) -> ProjectStateLayout:
    root = (project_root or discover_project_root()).expanduser().resolve()
    layout = ProjectStateLayout(
        project_root=root,
        asa_root=root / ".asa",
        session_root=root / ".asa" / "sessions",
        evidence_root=root / ".asa" / "evidence",
        skills_root=root / ".asa" / "skills",
        runs_root=root / ".asa" / "runs",
        agents_file=root / ".asa" / "AGENTS.md",
        skills_file=root / ".asa" / "SKILLS.md",
    )
    _ensure_inside(root, layout.asa_root)
    for path in (layout.session_root, layout.evidence_root, layout.skills_root, layout.runs_root):
        _ensure_inside(layout.asa_root, path)
        path.mkdir(parents=True, exist_ok=True)
    _write_template_if_absent(layout.agents_file, _agents_template(root))
    _write_template_if_absent(layout.skills_file, _skills_template(root))
    return layout


def project_guidance_text(project_root: Path | None = None) -> str:
    root = (project_root or discover_project_root()).expanduser().resolve()
    sections: list[str] = []
    for relative in PROJECT_GUIDANCE_FILES:
        path = root / relative
        text = _read_guidance_file(path)
        if text:
            sections.append(f"[{relative}]\n{text}")
    return "\n\n".join(sections).strip()


def _env_project_root() -> Path | None:
    raw = os.environ.get(ASA_PROJECT_ROOT_ENV, "").strip()
    if not raw:
        return None
    try:
        root = Path(raw).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return root if root.is_dir() else None


def _has_project_marker(path: Path) -> bool:
    return any((path / marker).exists() for marker in PROJECT_MARKERS)


def _ensure_inside(root: Path, path: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve() if path.exists() else path.parent.resolve() / path.name
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise ProjectLayoutError(f"project_state_path_outside_root:{path}")


def _write_template_if_absent(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    path.write_text(content, encoding="utf-8")


def _read_guidance_file(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > MAX_GUIDANCE_BYTES:
        data = data[:MAX_GUIDANCE_BYTES]
    try:
        return data.decode("utf-8").strip()
    except UnicodeDecodeError:
        return ""


def _agents_template(root: Path) -> str:
    return "\n".join(
        (
            "# ASA Project Agents",
            "",
            f"Project root: {root}",
            "",
            "- Keep ASA runtime state under this project-local `.asa` directory.",
            "- Keep credentials and personal runtime preferences in the user config home.",
            "- Use domain agent roles and markdown skills as separate prompt layers.",
            "",
        )
    )


def _skills_template(root: Path) -> str:
    return "\n".join(
        (
            "# ASA Project Skills",
            "",
            f"Project root: {root}",
            "",
            "Place reusable markdown skills in `.asa/skills`, `.codex/skills`, or `.claude/skills`.",
            "Domain knowledge belongs in system or role prompts, not in this catalog file.",
            "",
        )
    )
