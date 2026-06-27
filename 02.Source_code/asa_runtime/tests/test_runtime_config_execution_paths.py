from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.runtime_config import (
    ComputeResourceConfig,
    ModelEndpointRuntimeConfig,
    default_runtime_config,
    save_runtime_config,
)


def test_asa_chat_defaults_to_saved_runtime_model_and_compute_resource(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    output_dir = tmp_path / "chat"
    result = _run_module(
        [
            "chat",
            "--message",
            "Plan Ar etching on amorphous Si with a 3D hole pattern",
            "--output-dir",
            str(output_dir),
            "--source-root",
            str(SOURCE_ROOT),
        ],
        config_path,
    )

    request = json.loads((output_dir / "validated_request.json").read_text(encoding="utf-8"))
    worker = json.loads((output_dir / "worker_bundle.json").read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "llm_endpoint" not in request
    assert request["model_provider"]["provider"] == "local_gateway"
    assert request["model_provider"]["model"] == "configured-gpt-5.5"
    assert request["model_provider"]["base_url"] == "https://configured.gateway/v1"
    assert worker["host_alias"] == "controller-gpu"
    assert worker["environment_name"] == "configured-env"


def test_tui_plain_goal_uses_team_mode_default_and_writes_team_runtime_ledger(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    output_dir = tmp_path / "plain-goal"
    result = _run_module_interactive(
        [
            "--output-dir "
            f"{shlex.quote(str(output_dir))} "
            f"--source-root {shlex.quote(str(SOURCE_ROOT))} "
            "Plan Ar etching on amorphous Si with a 3D hole pattern",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    team_ledger_path = output_dir / "team" / "agent_team_session_ledger.json"
    ledger = json.loads(team_ledger_path.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "team_runtime_primary=true" in result.stdout
    assert "run_prepared=true" in result.stdout
    assert ledger["execution_mode"] == "team_contract_runtime"
    assert ledger["call_matrix"]["md_agent"] == ["orchestrator", "research_agent", "qa_agent"]
    assert any(event["event_type"] == "team_runtime_primary" for event in ledger["events"])


def test_tui_setup_runtime_adds_and_removes_compute_resource(tmp_path: Path) -> None:
    config_path = _write_runtime_config(tmp_path / "runtime-config.json")
    result = _run_module_interactive(
        [
            "/setup runtime "
            "--compute-resource setup-gpu "
            "--roles gpu,mdn,feature_scale "
            "--priority 1 "
            "--environment-name setup-env "
            "--ssh-target swym@setup-host "
            "--ssh-port 2222",
            "/setup runtime "
            "--compute-resource local-test "
            "--roles gpu "
            "--priority 2 "
            "--environment-name local-env "
            "--local",
            "/setup runtime --remove-compute-resource controller-gpu",
            "/exit",
        ],
        config_path,
        session_dir=tmp_path / "session",
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    resources = {item["host_alias"]: item for item in saved["compute_resources"]}

    assert result.returncode == 0, result.stdout + result.stderr
    assert "runtime_compute_resource_saved=setup-gpu" in result.stdout
    assert "runtime_compute_resource_saved=local-test" in result.stdout
    assert "runtime_compute_resource_removed=controller-gpu" in result.stdout
    assert "setup-gpu" in resources
    assert "local-test" in resources
    assert "controller-gpu" not in resources
    assert resources["setup-gpu"]["ssh_target"] == "swym@setup-host"
    assert resources["setup-gpu"]["ssh_port"] == 2222
    assert resources["local-test"]["local"] is True


def test_agents_runtime_skill_registry_invokes_handlers_and_persists_session(tmp_path: Path) -> None:
    from sim_agent.agents_sdk_runtime import run_agents_sdk_runtime_dry_run, write_agents_sdk_runtime_ledger
    from sim_agent.llm_endpoints import ModelProviderConfig

    payload = {
        "request_id": "skill-ledger",
        "user_goal": "Plan Ar etching on amorphous Si",
        "material": "Si",
        "phase": "amorphous",
        "ion": "Ar",
        "md_incident_count": 500,
    }
    endpoint = ModelProviderConfig.from_mapping(
        {
            "provider": "local_gateway",
            "model": "configured-gpt-5.5",
            "reasoning_effort": "high",
            "base_url": "https://configured.gateway/v1",
            "auth_mode": "oauth",
            "api_key_env": "CONFIGURED_GATEWAY_TOKEN",
        }
    )
    result = run_agents_sdk_runtime_dry_run(payload, endpoint, output_dir=tmp_path)
    ledger_path = write_agents_sdk_runtime_ledger(tmp_path, result)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    assert Path(ledger["session_path"]).is_file()
    assert ledger["session_path"].endswith("sim-agent-sdk-skill-ledger.sqlite")
    assert ledger["skill_registry"]["dispatch_mode"] == "callable_handlers"
    assert len(ledger["skill_invocations"]) == 7
    adapter_actions = {invocation["result"]["adapter_action"] for invocation in ledger["skill_invocations"]}
    assert len(adapter_actions) == 7
    for invocation in ledger["skill_invocations"]:
        assert invocation["status"] in {"ready", "blocked"}
        assert invocation["execution_status"] in {"adapter_contract_ready", "adapter_preflight_blocked"}
        assert invocation["domain_adapter"]
        assert invocation["artifact_ref"].startswith("skill_invocations/skill-ledger/")
        assert invocation["result"]["adapter_invoked"] is True


def test_browser_controller_uses_runtime_config_compute_source_of_truth() -> None:
    controller_js = UI_ROOT / "run_bundle_controller.js"
    runtime_js = UI_ROOT / "run_bundle_runtime_config.js"
    script = "\n".join(
        [
            f"globalThis.RunBundleController = require({str(controller_js)!r});",
            f"const runtime = require({str(runtime_js)!r});",
            "class Node { constructor(id) { this.id = id; this.value = ''; this.children = []; } replaceChildren(...children) { this.children = children; } }",
            "const nodes = { 'compute-target': new Node('compute-target'), 'runtime-workspace-root': new Node('runtime-workspace-root'), 'runtime-evidence-root': new Node('runtime-evidence-root'), 'runtime-compute-resources': new Node('runtime-compute-resources'), 'model-provider': new Node('model-provider'), 'model-name': new Node('model-name'), 'reasoning-effort': new Node('reasoning-effort'), 'auth-mode': new Node('auth-mode'), 'model-base-url': new Node('model-base-url'), 'model-api-key-env': new Node('model-api-key-env') };",
            "const documentRef = { getElementById(id) { return nodes[id] || null; }, createElement(tag) { return new Node(tag); } };",
            "runtime.applyRuntimeConfig(documentRef, { workspace_root: '/work', evidence_root: '/evidence', model_endpoint: { provider: 'local_gateway', model: 'configured-gpt-5.5', reasoning_effort: 'high', auth_mode: 'oauth', base_url: 'https://configured.gateway/v1', api_key_env: 'CONFIGURED_GATEWAY_TOKEN' }, compute_resources: [{ host_alias: 'controller-gpu', roles: ['gpu'], priority: 1, environment_name: 'configured-env', remote_user: 'swym' }] });",
            "const valid = globalThis.RunBundleController.validateControllerInput({ mode: '3d', geometryPath: 'scene.json', kernelPath: 'kernel.json', eventsPath: 'events.jsonl', computeTarget: 'controller-gpu', steps: 3, ions: 4, runId: 'runtime-configured', iedfReady: true, iadfReady: true });",
            "if (!valid.canRun) throw new Error('custom runtime compute rejected');",
            "const stale = globalThis.RunBundleController.validateControllerInput({ mode: '3d', geometryPath: 'scene.json', kernelPath: 'kernel.json', eventsPath: 'events.jsonl', computeTarget: 'gpu-5090', steps: 3, ions: 4, runId: 'stale-default', iedfReady: true, iadfReady: true });",
            "if (stale.canRun) throw new Error('removed static compute accepted');",
            "if (!stale.missingFields.includes('compute')) throw new Error('stale compute did not report compute missing');",
            "if (nodes['compute-target'].children[0].value !== 'controller-gpu') throw new Error('select was not populated from runtime config');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_agent_client_blocks_without_saved_or_form_model_endpoint() -> None:
    agent_js = UI_ROOT / "run_bundle_agent_client.js"
    script = "\n".join(
        [
            f"const agent = require({str(agent_js)!r});",
            "const documentRef = { getElementById() { return { value: '' }; } };",
            "globalThis.__ASA_MODEL_ENDPOINT__ = {};",
            "const result = agent.applyModelSettings(documentRef, { user_goal: 'Plan etch' });",
            "if (!result.error || !result.error.startsWith('model_endpoint_not_configured:')) throw new Error('missing endpoint not blocked');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def _write_runtime_config(path: Path) -> Path:
    default = default_runtime_config()
    config = replace(
        default,
        workspace_root=str(SOURCE_ROOT),
        evidence_root=str(path.parent / "evidence"),
        model_endpoint=ModelEndpointRuntimeConfig(
            provider="local_gateway",
            model="configured-gpt-5.5",
            reasoning_effort="high",
            base_url="https://configured.gateway/v1",
            auth_mode="oauth",
            api_key_env="CONFIGURED_GATEWAY_TOKEN",
        ),
        compute_resources=(
            ComputeResourceConfig(
                host_alias="controller-gpu",
                roles=("gpu", "mdn", "feature_scale"),
                priority=1,
                environment_name="configured-env",
                remote_user="swym",
                ssh_target="swym@controller-gpu",
                ssh_port=2222,
                local=False,
            ),
        ),
    )
    save_runtime_config(config, path)
    return path


def _run_module(args: list[str], config_path: Path) -> subprocess.CompletedProcess[str]:
    env = _env(config_path)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent", *args],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_module_interactive(
    lines: list[str],
    config_path: Path,
    *,
    session_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = _env(config_path)
    env["ASA_SESSION_DIR"] = str(session_dir)
    return subprocess.run(
        [sys.executable, "-m", "sim_agent"],
        cwd=PROJECT_ROOT,
        env=env,
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )


def _env(config_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT)
    env["ATOMISTIC_SIM_AGENT_RUNTIME_CONFIG"] = str(config_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env
