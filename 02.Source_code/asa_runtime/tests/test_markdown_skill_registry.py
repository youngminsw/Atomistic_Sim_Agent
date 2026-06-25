from __future__ import annotations

from pathlib import Path


def test_markdown_skill_registry_loads_reusable_md_files() -> None:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_specs

    specs = {spec.name: spec for spec in markdown_skill_specs()}

    assert {"md", "ml", "research", "feature-scale", "qa", "controller"}.issubset(specs)
    assert specs["md"].agent_id == "md_agent"
    assert specs["ml"].agent_id == "ml_agent"
    assert specs["research"].agent_id == "research_agent"
    assert specs["md"].command == "/md"
    assert specs["md"].path.name == "md.md"
    assert "LAMMPS" in specs["md"].summary
    assert "system prompt layer" in specs["md"].body


def test_markdown_skill_command_names_are_direct_slash_commands() -> None:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_command_names

    assert "/md" in markdown_skill_command_names()
    assert "/ml" in markdown_skill_command_names()
    assert "/research" in markdown_skill_command_names()


def test_markdown_skills_are_first_class_tui_palette_commands() -> None:
    from sim_agent.cli.tui_catalog import command_names, suggested_commands

    names = command_names()
    suggestions = suggested_commands("/m")

    assert "/md" in names
    assert "/ml" in names
    assert "/md" in {command.name for command in suggestions}
    assert any("md skill -> md_agent" in command.summary for command in suggestions)


def test_markdown_skill_roots_include_env_project_and_user_homes(tmp_path: Path, monkeypatch) -> None:
    from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_by_command, markdown_skill_specs

    env_root = _write_skill(tmp_path / "env-skills", "env-skill", "/env-skill", "qa_agent")
    project_root = tmp_path / "project"
    _write_skill(project_root / ".asa" / "skills", "project-skill", "/project-skill", "md_agent")
    _write_skill(project_root / ".codex" / "skills", "codex-skill", "/codex-skill", "ml_agent")
    _write_skill(project_root / ".claude" / "skills", "claude-skill", "/claude-skill", "research_agent")
    asa_home = tmp_path / "asa-home"
    codex_home = tmp_path / "codex-home"
    claude_home = tmp_path / "claude-home"
    _write_skill(asa_home / "skills", "asa-home-skill", "/asa-home-skill", "orchestrator")
    _write_skill(codex_home / "skills", "codex-home-skill", "/codex-home-skill", "qa_agent")
    _write_skill(claude_home / "skills", "claude-home-skill", "/claude-home-skill", "feature_scale_agent")
    monkeypatch.setenv("ASA_SKILL_ROOTS", str(env_root))
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("ASA_HOME", str(asa_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CLAUDE_HOME", str(claude_home))

    commands = {spec.command for spec in markdown_skill_specs()}

    assert {
        "/env-skill",
        "/project-skill",
        "/codex-skill",
        "/claude-skill",
        "/asa-home-skill",
        "/codex-home-skill",
        "/claude-home-skill",
    }.issubset(commands)
    assert markdown_skill_by_command("/project-skill").agent_id == "md_agent"


def test_markdown_skill_discovery_ignores_boundaries_and_defaults_commands(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime.markdown_skills import (
        markdown_skill_by_command,
        markdown_skill_specs,
        skill_context_body,
        skill_context_message,
    )

    root = tmp_path / "skills"
    root.mkdir()
    (root / "README.md").write_text("# ignored readme\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# ignored agents\n", encoding="utf-8")
    (root / "bad.md").write_bytes(b"\xff\xfe\x00not utf-8")
    _write_skill(root, "alpha", "/same-command", "qa_agent", body="alpha body")
    _write_skill(root, "omega", "/same-command", "md_agent", body="omega body")
    _write_skill(root, "Needs Default Command", None, "research_agent", body="default body")

    specs = markdown_skill_specs((root,))
    commands = {spec.command for spec in specs}

    assert "/same-command" in commands
    assert "/needs-default-command" in commands
    assert "README" not in {spec.name for spec in specs}
    assert "AGENTS" not in {spec.name for spec in specs}
    assert markdown_skill_by_command("/same-command", (root,)).name == "alpha"

    defaulted = markdown_skill_by_command("/needs-default-command", (root,))
    assert defaulted is not None
    assert defaulted.agent_id == "research_agent"
    assert skill_context_body(skill_context_message(defaulted)).startswith("Skill: Needs Default Command")
    assert skill_context_body("not-a-skill-context\nSkill: fake") == ""
    assert skill_context_body({"content": "not string"}) == ""


def _write_skill(root: Path, name: str, command: str | None, agent_id: str, *, body: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.md"
    lines = [
        "---",
        f"name: {name}",
    ]
    if command is not None:
        lines.append(f"command: {command}")
    lines.extend(
        [
            f"agent_id: {agent_id}",
            f"summary: {name} summary",
            "---",
            f"# {name}",
            "",
            body or f"{name} body",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return root
