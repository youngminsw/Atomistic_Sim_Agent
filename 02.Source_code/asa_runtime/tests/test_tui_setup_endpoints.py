from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sim_agent.runtime_config import default_runtime_config, save_runtime_config


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT


def test_setup_graphdb_saves_custom_env_names_and_database_for_memory(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/setup graphdb "
            "--uri bolt://asa-neo4j.test:7687 "
            "--uri-env LAB_NEO4J_URI "
            "--user-env LAB_NEO4J_USER "
            "--password-env LAB_NEO4J_PASS "
            "--database asa_brain_test",
            "/memory",
            "/setup graphdb --list",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
        extra_env={"NEO4J_DATABASE": "stale_global_database"},
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "graphdb_config_saved=true" in result.stdout
    assert "graphdb_uri=bolt://asa-neo4j.test:7687" in result.stdout
    assert "graphdb_uri_env=LAB_NEO4J_URI" in result.stdout
    assert "graphdb_user_env=LAB_NEO4J_USER" in result.stdout
    assert "graphdb_password_env=LAB_NEO4J_PASS" in result.stdout
    assert "graphdb_database=asa_brain_test" in result.stdout
    assert "database_name=asa_brain_test" in result.stdout
    assert "stale_global_database" not in result.stdout
    assert saved["graphdb"] == {
        "database": "asa_brain_test",
        "password_env": "LAB_NEO4J_PASS",
        "uri": "bolt://asa-neo4j.test:7687",
        "uri_env": "LAB_NEO4J_URI",
        "user_env": "LAB_NEO4J_USER",
    }


def test_setup_endpoint_persists_runtime_model_and_updates_current_session(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/setup endpoint "
            "--provider local_gateway "
            "--model gpt-5.5 "
            "--reasoning-effort high "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            "/model status",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    session = json.loads((tmp_path / "session" / "asa_session.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "endpoint_config_saved=true" in result.stdout
    assert "provider=local_gateway model=gpt-5.5 reasoning_effort=high" in result.stdout
    assert "base_url=http://runtime-gateway.test/v1 auth_mode=none" in result.stdout
    assert saved["model_endpoint"] == {
        "api_key_env": "RUNTIME_GATEWAY_TOKEN",
        "auth_mode": "none",
        "base_url": "http://runtime-gateway.test/v1",
        "model": "gpt-5.5",
        "provider": "local_gateway",
        "reasoning_effort": "high",
    }
    assert session["model"]["provider"] == "local_gateway"
    assert session["model"]["api_key_env"] == "RUNTIME_GATEWAY_TOKEN"


def test_setup_endpoint_accepts_thinking_level_alias(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/setup endpoint "
            "--provider local_gateway "
            "--model gpt-5.3 "
            "--reasoning-effort low "
            "--thinking-level high "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "provider=local_gateway model=gpt-5.3 reasoning_effort=high" in result.stdout
    assert saved["model_endpoint"]["reasoning_effort"] == "high"


def test_model_catalog_lists_selectable_gateway_models(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(["/model list", "/exit"], config_path, session_dir=tmp_path / "session")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_catalog=true" in result.stdout
    assert "model_group=OpenAI" in result.stdout
    assert "model_provider_group=openai-codex" in result.stdout
    assert "source_provider=openai-codex model=gpt-5-codex provider=openai-codex" in result.stdout
    assert "source_provider=openai-codex model=gpt-5.4 provider=openai-codex" in result.stdout
    assert "source_provider=openai-codex model=gpt-5.5 provider=openai-codex" in result.stdout
    assert "model_group=Google" in result.stdout
    assert "source_provider=google-gemini-cli model=gemini-3-pro-preview provider=google-gemini-cli" in result.stdout
    assert "source_provider=github-copilot model=claude-sonnet-4.5 provider=github-copilot" in result.stdout
    assert "source_provider=local_gateway model=gpt-5.3-codex-spark provider=local_gateway" in result.stdout
    assert "source_provider=anthropic model=claude-sonnet-4.5 provider=anthropic" in result.stdout
    assert "provider=oauth_gateway" not in result.stdout
    assert "provider=openclaw" not in result.stdout


def test_model_set_without_reference_does_not_silently_resave_current_model(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(["/model set", "/exit"], config_path, session_dir=tmp_path / "session")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_error=model_reference_required" in result.stdout
    assert "model_hint=/model set <provider/model> or /model list" in result.stdout
    assert "model_set_ok=true" not in result.stdout


def test_model_use_persists_catalog_model_as_runtime_default(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/model use local_gateway/gpt-5.3 "
            "--reasoning-effort high "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            "/model status",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_saved=true" in result.stdout
    assert "provider=local_gateway model=gpt-5.3 reasoning_effort=high" in result.stdout
    assert saved["model_endpoint"] == {
        "api_key_env": "RUNTIME_GATEWAY_TOKEN",
        "auth_mode": "none",
        "base_url": "http://runtime-gateway.test/v1",
        "model": "gpt-5.3",
        "provider": "local_gateway",
        "reasoning_effort": "high",
    }


def test_model_use_accepts_thinking_level_alias(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/model use local_gateway/gpt-5.3-codex-spark --thinking-level high "
            "--base-url http://runtime-gateway.test/v1 --auth-mode none",
            "/model status",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "provider=local_gateway model=gpt-5.3-codex-spark reasoning_effort=high" in result.stdout
    assert saved["model_endpoint"]["reasoning_effort"] == "high"


def test_model_thinking_command_updates_current_session_model(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(["/model thinking medium", "/model status", "/exit"], config_path, session_dir=tmp_path / "session")
    session = json.loads((tmp_path / "session" / "asa_session.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "model_thinking_level_saved=true" in result.stdout
    assert "thinking_level=medium reasoning_effort=medium" in result.stdout
    assert "provider=openai-codex model=gpt-5-codex reasoning_effort=medium" in result.stdout
    assert session["model"]["reasoning_effort"] == "medium"


def test_model_assign_persists_agent_specific_model_override(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/model assign "
            "--agent md_agent "
            "--provider local_gateway "
            "--model gpt-5.3-codex-spark "
            "--reasoning-effort medium "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            "/model agents",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_model_saved=md_agent" in result.stdout
    assert "agent_models=true" in result.stdout
    assert "agent=md_agent provider=local_gateway model=gpt-5.3-codex-spark reasoning_effort=medium override=true" in result.stdout
    assert saved["agent_model_overrides"] == [
        {
            "agent_id": "md_agent",
            "api_key_env": "RUNTIME_GATEWAY_TOKEN",
            "auth_mode": "none",
            "base_url": "http://runtime-gateway.test/v1",
            "model": "gpt-5.3-codex-spark",
            "provider": "local_gateway",
            "reasoning_effort": "medium",
        }
    ]


def test_model_assign_accepts_thinking_level_alias(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(
        [
            "/model assign "
            "--agent qa_agent "
            "--provider local_gateway "
            "--model gpt-5.3-codex-spark "
            "--thinking-level medium "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            "/model agents",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent=qa_agent provider=local_gateway model=gpt-5.3-codex-spark reasoning_effort=medium override=true" in result.stdout
    assert saved["agent_model_overrides"][0]["reasoning_effort"] == "medium"


def test_runtime_ledger_records_agent_model_overrides(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    runtime_dir = tmp_path / "runtime"

    result = _run_tui(
        [
            "/model assign "
            "--agent md_agent "
            "--provider local_gateway "
            "--model gpt-5.3-codex-spark "
            "--reasoning-effort medium "
            "--base-url http://runtime-gateway.test/v1 "
            "--auth-mode none "
            "--api-key-env RUNTIME_GATEWAY_TOKEN",
            f"/runtime --output-dir {runtime_dir}",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    ledger = json.loads((runtime_dir / "agents_sdk_runtime_ledger.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "agent_model=md_agent:local_gateway/gpt-5.3-codex-spark" in result.stdout
    assert ledger["agent_model_assignments"]["md_agent"] == {
        "api_key_env": "RUNTIME_GATEWAY_TOKEN",
        "auth_mode": "none",
        "base_url": "http://runtime-gateway.test/v1",
        "model": "gpt-5.3-codex-spark",
        "provider": "local_gateway",
        "reasoning_effort": "medium",
        "source": "override",
    }
    assert ledger["agent_model_assignments"]["qa_agent"]["model"] == "gpt-5-codex"
    assert ledger["agent_model_assignments"]["qa_agent"]["source"] == "default"


def test_setup_help_lists_graphdb_and_endpoint_scopes(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(["/setup", "/exit"], config_path, session_dir=tmp_path / "session")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "setup_scope=runtime" in result.stdout
    assert "setup_scope=graphdb" in result.stdout
    assert "setup_scope=endpoint" in result.stdout
    assert "usage=/setup graphdb --uri-env" in result.stdout
    assert "usage=/setup endpoint --provider" in result.stdout


def test_skills_command_shows_callable_agent_skill_handlers(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")

    result = _run_tui(["/skills", "/exit"], config_path, session_dir=tmp_path / "session")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "skill_catalog=true" in result.stdout
    assert "skill_registry_dispatch=callable_handlers" in result.stdout
    assert "skill_impl=md_agent:prepare_and_verify_lammps_md" in result.stdout
    assert "skill_impl=research_agent:research_and_ingest_graphdb_catalog" in result.stdout
    assert "skill_impl=qa_agent:qa_physics_and_runtime_evidence" in result.stdout


def _write_runtime_config(path: Path) -> Path:
    save_runtime_config(default_runtime_config(), path)
    return path


def _run_tui(
    lines: list[str],
    config_path: Path,
    *,
    session_dir: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ASA_SESSION_DIR"] = str(session_dir)
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
