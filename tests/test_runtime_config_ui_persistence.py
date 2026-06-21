from __future__ import annotations

import subprocess
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"


def test_browser_runtime_config_save_preserves_loaded_team_mode_and_graphdb() -> None:
    runtime_js = UI_ROOT / "run_bundle_runtime_config.js"
    script = "\n".join(
        [
            f"const runtime = require({str(runtime_js)!r});",
            "class Node { constructor(id) { this.id = id; this.value = ''; this.children = []; } replaceChildren(...children) { this.children = children; } }",
            "const nodes = { 'compute-target': new Node('compute-target'), 'runtime-workspace-root': new Node('runtime-workspace-root'), 'runtime-evidence-root': new Node('runtime-evidence-root'), 'runtime-compute-resources': new Node('runtime-compute-resources'), 'model-provider': new Node('model-provider'), 'model-name': new Node('model-name'), 'reasoning-effort': new Node('reasoning-effort'), 'auth-mode': new Node('auth-mode'), 'model-base-url': new Node('model-base-url'), 'model-api-key-env': new Node('model-api-key-env') };",
            "const documentRef = { getElementById(id) { return nodes[id] || null; }, createElement(tag) { return new Node(tag); } };",
            "runtime.applyRuntimeConfig(documentRef, { workspace_root: '/work', evidence_root: '/evidence', team_mode_default: false, graphdb: { uri_env: 'CUSTOM_URI', user_env: 'CUSTOM_USER', password_env: 'CUSTOM_PASS', database: 'custom_db' }, model_endpoint: { provider: 'local_gateway', model: 'configured-gpt-5.5', reasoning_effort: 'high', auth_mode: 'oauth', base_url: 'https://configured.gateway/v1', api_key_env: 'CONFIGURED_GATEWAY_TOKEN' }, compute_resources: [{ host_alias: 'old-gpu', roles: ['gpu'], priority: 1, environment_name: 'old-env', remote_user: 'swym' }] });",
            "nodes['runtime-compute-resources'].value = JSON.stringify([{ host_alias: 'new-gpu', roles: ['gpu'], priority: 2, environment_name: 'new-env', remote_user: 'swym' }]);",
            "const parsed = runtime.parseRuntimeConfig(documentRef);",
            "if (parsed.error) throw new Error(parsed.error);",
            "if (parsed.config.team_mode_default !== false) throw new Error('team mode default was clobbered');",
            "if (parsed.config.graphdb.database !== 'custom_db') throw new Error('graphdb database was clobbered');",
            "if (parsed.config.graphdb.uri_env !== 'CUSTOM_URI') throw new Error('graphdb uri env was clobbered');",
            "if (parsed.config.compute_resources[0].host_alias !== 'new-gpu') throw new Error('compute resources were not edited');",
        ]
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
