from __future__ import annotations

import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from sim_agent.agents_sdk_runtime.markdown_skills import markdown_skill_specs
from sim_agent.cli.tui_catalog import all_commands
from sim_agent.cli.tui import run_tui
from sim_agent.schemas._parse import JsonMap


@dataclass(frozen=True, slots=True)
class SkillWorkflowSmokeRequest:
    output_dir: Path


@dataclass(frozen=True, slots=True)
class SkillWorkflowSmokeResult:
    skill_matrix_path: Path
    workflow_matrix_path: Path
    transcript_path: Path
    status: str
    blockers: tuple[str, ...]


def run_skill_workflow_smoke(request: SkillWorkflowSmokeRequest) -> SkillWorkflowSmokeResult:
    output_dir = request.output_dir.expanduser().resolve()
    scenario_root = output_dir / "skill-workflow-project"
    session_dir = scenario_root / ".asa" / "sessions" / "g006-skill-workflow"
    missing_workflow_dir = output_dir / "workflow-missing"
    passed_workflow_dir = output_dir / "workflow-passed"
    transcript_path = output_dir / "task-8-skills.txt"
    skill_matrix_path = output_dir / "task-8-skill-parity-matrix.json"
    workflow_matrix_path = output_dir / "task-10-workflow-gate-parity-matrix.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_scenario_project(scenario_root)
    runtime_config_path = _write_static_runtime_config(output_dir, scenario_root)

    transcript = _run_tui_surface(
        scenario_root,
        session_dir,
        runtime_config_path,
        missing_workflow_dir,
        passed_workflow_dir,
    )
    transcript_path.write_text(transcript, encoding="utf-8")
    skill_matrix = _skill_matrix(scenario_root, transcript, session_dir)
    workflow_matrix = _workflow_matrix(missing_workflow_dir, passed_workflow_dir, transcript)
    skill_matrix_path.write_text(json.dumps(skill_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    workflow_matrix_path.write_text(json.dumps(workflow_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    blockers = tuple(skill_matrix["blockers"]) + tuple(workflow_matrix["blockers"])
    return SkillWorkflowSmokeResult(
        skill_matrix_path=skill_matrix_path,
        workflow_matrix_path=workflow_matrix_path,
        transcript_path=transcript_path,
        status="succeeded" if not blockers else "blocked",
        blockers=blockers,
    )


def _write_scenario_project(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname = \"asa-g006-skill-workflow-smoke\"\n", encoding="utf-8")
    (root / "AGENTS.md").write_text("# ASA G006 smoke project\n", encoding="utf-8")
    _write_skill(root / ".asa" / "skills", "asa-probe", "/asa-probe", "md_agent")
    _write_skill(root / ".codex" / "skills", "codex-probe", "/codex-probe", "qa_agent")
    _write_skill(root / ".claude" / "skills", "claude-probe", "/claude-probe", "research_agent")


def _write_skill(root: Path, name: str, command: str, agent_id: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(
        "\n".join(
            (
                "---",
                f"name: {name}",
                f"command: {command}",
                f"agent_id: {agent_id}",
                f"summary: {name} reusable markdown skill",
                "---",
                f"# {name}",
                "",
                "This reusable skill is injected as a system prompt layer during the smoke.",
                "",
            )
        ),
        encoding="utf-8",
    )


def _write_static_runtime_config(output_dir: Path, project_root: Path) -> Path:
    path = output_dir / "runtime-config.json"
    payload: JsonMap = {
        "workspace_root": str(project_root),
        "evidence_root": str(project_root / ".asa" / "evidence"),
        "team_mode_default": True,
        "model_endpoint": {
            "provider": "static",
            "model": "explicit-static",
            "reasoning_effort": "high",
            "base_url": "http://static.local/v1",
            "auth_mode": "none",
            "api_key_env": "STATIC_TOKEN",
        },
        "active_profile": {"name": "", "customized": False},
        "agent_model_overrides": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_tui_surface(
    project_root: Path,
    session_dir: Path,
    runtime_config_path: Path,
    missing_workflow_dir: Path,
    passed_workflow_dir: Path,
) -> str:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_STARTUP_WIZARD"] = "0"
    env["ASA_PROJECT_ROOT"] = str(project_root)
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(runtime_config_path)
    commands = "\n".join(
        (
            "/",
            "/asa-probe Route this markdown skill to the MD agent.",
            "/codex-probe Route this markdown skill to the QA agent.",
            "/claude-probe Route this markdown skill to the research agent.",
            f"/workflow ralplan --output-dir {missing_workflow_dir}",
            f"/workflow ralplan --evidence-key prd_path,test_spec_path --output-dir {passed_workflow_dir}",
            "/exit",
            "",
        )
    )
    old_env = {key: os.environ.get(key) for key in env}
    try:
        os.environ.update(env)
        output = StringIO()
        returncode = run_tui(StringIO(commands), output, session_dir=session_dir)
        return "\n".join((f"returncode={returncode}", "--- STDOUT ---", output.getvalue(), "--- STDERR ---", ""))
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _skill_matrix(project_root: Path, transcript: str, session_dir: Path) -> JsonMap:
    specs = _project_specs(project_root)
    commands = {command.name for command in _project_commands(project_root)}
    sources = {source: any(source in str(spec.path) for spec in specs) for source in (".asa/skills", ".codex/skills", ".claude/skills")}
    expected_commands = {"/asa-probe", "/codex-probe", "/claude-probe"}
    context_records = _skill_context_records(session_dir)
    blockers: list[str] = []
    if not all(sources.values()):
        blockers.append("skill_source_discovery_incomplete")
    if not expected_commands.issubset(commands):
        blockers.append("skill_palette_commands_missing")
    if transcript.count("markdown_skill_invoked=true") < 3:
        blockers.append("slash_skill_invocation_missing")
    if len(context_records) < 3:
        blockers.append("skill_prompt_context_not_injected")
    return {
        "matrix_version": "asa_skill_parity_v1",
        "status": "succeeded" if not blockers else "blocked",
        "blockers": blockers,
        "skill_sources": sources,
        "expected_commands": sorted(expected_commands),
        "palette_commands_present": sorted(command for command in expected_commands if command in commands),
        "slash_invocation_count": transcript.count("markdown_skill_invoked=true"),
        "prompt_context_record_count": len(context_records),
        "prompt_context_agents": sorted(record["agent_id"] for record in context_records),
        "session_dir": str(session_dir),
    }


def _project_specs(project_root: Path):
    old = os.environ.get("ASA_PROJECT_ROOT")
    os.environ["ASA_PROJECT_ROOT"] = str(project_root)
    try:
        return markdown_skill_specs()
    finally:
        if old is None:
            os.environ.pop("ASA_PROJECT_ROOT", None)
        else:
            os.environ["ASA_PROJECT_ROOT"] = old


def _project_commands(project_root: Path):
    old = os.environ.get("ASA_PROJECT_ROOT")
    os.environ["ASA_PROJECT_ROOT"] = str(project_root)
    try:
        return all_commands()
    finally:
        if old is None:
            os.environ.pop("ASA_PROJECT_ROOT", None)
        else:
            os.environ["ASA_PROJECT_ROOT"] = old


def _skill_context_records(session_dir: Path) -> list[JsonMap]:
    records: list[JsonMap] = []
    for path in sorted((session_dir / "agent_sessions").glob("*/messages.jsonl")):
        agent_id = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = payload.get("content")
            if payload.get("role") == "system" and isinstance(content, str) and content.startswith("ASA_SKILL_CONTEXT_V1\n"):
                records.append({"agent_id": agent_id, "path": str(path)})
    return records


def _workflow_matrix(missing_dir: Path, passed_dir: Path, transcript: str) -> JsonMap:
    missing = _read_json(missing_dir / "ralplan" / "workflow_harness_ledger.json")
    passed = _read_json(passed_dir / "ralplan" / "workflow_harness_ledger.json")
    blockers: list[str] = []
    if missing.get("status") != "blocked" or missing.get("gate_status") != "blocked":
        blockers.append("missing_evidence_did_not_block")
    if "workflow_gate_missing_evidence" not in missing.get("blockers", []):
        blockers.append("missing_evidence_blocker_absent")
    if passed.get("status") != "ready" or passed.get("gate_status") != "passed":
        blockers.append("provided_evidence_did_not_unlock")
    if "workflow_missing_evidence=prd_path,test_spec_path" not in transcript:
        blockers.append("tui_missing_evidence_not_rendered")
    if "workflow_evidence_keys=prd_path,test_spec_path" not in transcript:
        blockers.append("tui_evidence_keys_not_rendered")
    return {
        "matrix_version": "asa_workflow_gate_parity_v1",
        "status": "succeeded" if not blockers else "blocked",
        "blockers": blockers,
        "missing_case": {
            "status": missing.get("status", ""),
            "gate_status": missing.get("gate_status", ""),
            "missing_evidence": missing.get("missing_evidence", []),
            "blockers": missing.get("blockers", []),
        },
        "passed_case": {
            "status": passed.get("status", ""),
            "gate_status": passed.get("gate_status", ""),
            "evidence_keys": passed.get("evidence_keys", []),
            "missing_evidence": passed.get("missing_evidence", []),
        },
        "missing_ledger_path": str(missing_dir / "ralplan" / "workflow_harness_ledger.json"),
        "passed_ledger_path": str(passed_dir / "ralplan" / "workflow_harness_ledger.json"),
    }


def _read_json(path: Path) -> JsonMap:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
