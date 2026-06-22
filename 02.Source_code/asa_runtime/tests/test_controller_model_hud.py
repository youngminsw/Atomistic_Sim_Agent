from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sim_agent.schemas._parse import as_mapping, as_str


def test_ui_status_payload_includes_model_auth_hud(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATOMISTIC_MODEL_GATEWAY_CREDENTIAL_STORE", str(tmp_path / "credentials.json"))

    from sim_agent.ui import build_ui_api_status
    from sim_agent.ui.response_payload import status_payload

    payload = status_payload(build_ui_api_status())
    model_auth = as_mapping(payload["model_auth"], "model_auth")

    assert model_auth["connected_provider_count"] == 0
    assert "Model is not connected" in as_str(model_auth["friendly_message"], "friendly_message")
    assert "/login" in as_str(model_auth["action_hint"], "action_hint")
    assert "credential_store" not in model_auth


def test_controller_model_auth_renders_friendly_disconnected_hud() -> None:
    js = UI_ROOT / "run_bundle_model_auth.js"
    script = "\n".join(
        [
            f"const auth = require({str(js)!r});",
            "class Node {",
            "  constructor(id) { this.id = id; this.value = ''; this.textContent = ''; this.className = 'controller-errors'; this.listeners = {}; }",
            "  addEventListener(name, fn) { this.listeners[name] = fn; }",
            "}",
            "const nodes = {",
            "  'model-auth-form': new Node('model-auth-form'),",
            "  'model-gateway-smoke': new Node('model-gateway-smoke'),",
            "  'model-auth-status': new Node('model-auth-status'),",
            "};",
            "const documentRef = { getElementById(id) { return nodes[id] || null; } };",
            "const fetcher = () => Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true, providers: [], connected_provider_count: 0 }) });",
            "async function main() {",
            "  await auth.refreshStatus(documentRef, fetcher);",
            "  if (!nodes['model-auth-status'].textContent.includes('Model is not connected')) throw new Error('missing disconnected guidance');",
            "  if (!nodes['model-auth-status'].textContent.includes('/login')) throw new Error('missing login guidance');",
            "  if (!nodes['model-auth-status'].className.includes('blocked')) throw new Error('missing blocked class');",
            "}",
            "main().catch((error) => { console.error(error); process.exit(1); });",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_controller_model_auth_treats_expired_providers_as_disconnected() -> None:
    js = UI_ROOT / "run_bundle_model_auth.js"
    script = "\n".join(
        [
            f"const auth = require({str(js)!r});",
            "class Node {",
            "  constructor(id) { this.id = id; this.textContent = ''; this.className = 'controller-errors'; }",
            "}",
            "const status = new Node('model-auth-status');",
            "const documentRef = { getElementById(id) { return id === 'model-auth-status' ? status : null; } };",
            "auth.renderStatus(documentRef, {",
            "  ok: true,",
            "  providers: [{ provider: 'openai', logged_in: false, expires: 1 }],",
            "  connected_provider_count: 0,",
            "  friendly_message: 'Model is not connected. The stored credential is expired.',",
            "  action_hint: 'Run /login again.',",
            "});",
            "if (!status.textContent.includes('Model is not connected')) throw new Error('expired provider shown as connected');",
            "if (status.textContent.includes('Model credentials: openai')) throw new Error('expired provider listed as available');",
            "if (!status.className.includes('blocked')) throw new Error('expired provider not blocked');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
