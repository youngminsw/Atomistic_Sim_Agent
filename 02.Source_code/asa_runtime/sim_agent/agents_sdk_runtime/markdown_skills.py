from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from sim_agent.project_layout import ASA_PROJECT_ROOT_ENV, discover_project_root
from sim_agent.schemas._parse import JsonMap


DEFAULT_SKILL_ROOT: Final = Path(__file__).resolve().parent / "default_skills"
ASA_SKILL_ROOTS_ENV: Final = "ASA_SKILL_ROOTS"
PROJECT_SKILL_DIRS: Final = (".asa/skills", ".codex/skills", ".claude/skills")
USER_SKILL_HOMES: Final = (("ASA_HOME", ".asa"), ("CODEX_HOME", ".codex"), ("CLAUDE_HOME", ".claude"))
SKILL_CONTEXT_MARKER: Final = "ASA_SKILL_CONTEXT_V1"


@dataclass(frozen=True, slots=True)
class MarkdownSkillSpec:
    name: str
    command: str
    agent_id: str
    summary: str
    path: Path
    body: str

    def to_prompt_layer(self) -> str:
        return "\n".join(
            (
                f"Skill: {self.name}",
                f"Command: {self.command}",
                f"Agent: {self.agent_id}",
                f"Summary: {self.summary}",
                "",
                self.body,
            )
        )


def markdown_skill_specs(roots: tuple[Path, ...] | None = None) -> tuple[MarkdownSkillSpec, ...]:
    specs: list[MarkdownSkillSpec] = []
    seen_commands: set[str] = set()
    for root in roots or markdown_skill_roots():
        if not root.is_dir():
            continue
        for path in _skill_paths(root):
            spec = _spec_from_path(path)
            if spec is not None and spec.command not in seen_commands:
                specs.append(spec)
                seen_commands.add(spec.command)
    return tuple(specs)


def markdown_skill_command_names(roots: tuple[Path, ...] | None = None) -> tuple[str, ...]:
    return tuple(spec.command for spec in markdown_skill_specs(roots))


def markdown_skill_by_command(command: str, roots: tuple[Path, ...] | None = None) -> MarkdownSkillSpec | None:
    for spec in markdown_skill_specs(roots):
        if spec.command == command:
            return spec
    return None


def markdown_skill_summary_rows(roots: tuple[Path, ...] | None = None) -> tuple[tuple[str, str], ...]:
    return tuple((spec.name, spec.summary) for spec in markdown_skill_specs(roots))


def markdown_skill_roots(project_root: Path | None = None) -> tuple[Path, ...]:
    return tuple(_unique_paths((*_env_roots(), *_project_skill_roots(project_root), *_user_skill_roots(), DEFAULT_SKILL_ROOT)))


def skill_context_message(spec: MarkdownSkillSpec) -> str:
    return "\n".join((SKILL_CONTEXT_MARKER, spec.to_prompt_layer()))


def skill_context_body(content: object) -> str:
    if not isinstance(content, str):
        return ""
    marker, _separator, body = content.partition("\n")
    if marker.strip() != SKILL_CONTEXT_MARKER:
        return ""
    return body.strip()


def _spec_from_path(path: Path) -> MarkdownSkillSpec | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    metadata, body = _frontmatter(text)
    name = _string_field(metadata, "name", path.stem)
    command = _command_field(metadata, name)
    agent_id = _string_field(metadata, "agent_id", "orchestrator")
    summary = _string_field(metadata, "summary", _string_field(metadata, "description", _first_body_line(body)))
    if not name or not command.startswith("/") or not agent_id:
        return None
    return MarkdownSkillSpec(
        name=name,
        command=command,
        agent_id=agent_id,
        summary=summary,
        path=path,
        body=body.strip(),
    )


def _skill_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    ignored_root_files = {"AGENTS.md", "README.md", "SKILLS.md", "SKILL.md"}
    try:
        children = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError:
        return ()
    root_skill = root / "SKILL.md"
    if root_skill.is_file():
        paths.append(root_skill)
    paths.extend(
        path for path in children if path.suffix == ".md" and path.name not in ignored_root_files and path.is_file()
    )
    paths.extend(path / "SKILL.md" for path in children if (path / "SKILL.md").is_file())
    return tuple(_unique_paths(paths))


def _env_roots() -> tuple[Path, ...]:
    raw = os.environ.get(ASA_SKILL_ROOTS_ENV, "")
    return tuple(Path(item).expanduser() for item in raw.split(os.pathsep) if item.strip())


def _project_skill_roots(project_root: Path | None) -> tuple[Path, ...]:
    roots: list[Path] = []
    for base in _candidate_project_bases(project_root):
        roots.extend(base / relative for relative in PROJECT_SKILL_DIRS)
    return tuple(roots)


def _candidate_project_bases(project_root: Path | None) -> tuple[Path, ...]:
    explicit = project_root or _env_project_root()
    discovered = discover_project_root(explicit) if explicit is not None else discover_project_root()
    starts = [path for path in (explicit, discovered, Path.cwd(), Path(__file__).resolve().parents[2]) if path is not None]
    bases: list[Path] = []
    for start in starts:
        bases.append(start)
        bases.extend(start.parents)
    return tuple(_unique_paths(bases))


def _env_project_root() -> Path | None:
    raw = os.environ.get(ASA_PROJECT_ROOT_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def _user_skill_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for env_name, default_name in USER_SKILL_HOMES:
        raw = os.environ.get(env_name)
        home = Path(raw).expanduser() if raw else Path.home() / default_name
        roots.append(home / "skills")
    return tuple(roots)


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = _lexical_path_key(path)
        if key not in seen:
            unique.append(path.expanduser())
            seen.add(key)
    return tuple(unique)


def _lexical_path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path.expanduser())))


def _frontmatter(text: str) -> tuple[JsonMap, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    metadata: dict[str, object] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :])
            return metadata, body
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip()
    return {}, text


def _command_field(metadata: JsonMap, name: str) -> str:
    value = _string_field(metadata, "command", _command_from_name(name))
    return value if value.startswith("/") else f"/{value}"


def _command_from_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", name.strip().casefold()).strip("-")
    return f"/{slug or 'skill'}"


def _string_field(metadata: JsonMap, field: str, fallback: str) -> str:
    value = metadata.get(field)
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _first_body_line(body: str) -> str:
    for line in body.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned
    return ""
