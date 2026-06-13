from __future__ import annotations

import subprocess
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"


def test_model_auth_controller_posts_login_and_redacts_token() -> None:
    html = (UI_ROOT / "run_bundle_viewer.html").read_text(encoding="utf-8")
    js = UI_ROOT / "run_bundle_model_auth.js"
    script = "\n".join(
        [
            f"const auth = require({str(js)!r});",
            "class Node {",
            "  constructor(id) { this.id = id; this.value = ''; this.textContent = ''; this.listeners = {}; this.children = []; this.type = 'text'; }",
            "  addEventListener(name, fn) { this.listeners[name] = fn; }",
            "  append(node) { this.children.push(node); this.textContent += node.textContent; }",
            "  trigger(name) { return this.listeners[name]({ preventDefault() {} }); }",
            "}",
            "const ids = ['model-auth-form', 'model-provider', 'model-access-token', 'model-refresh-token', 'model-auth-status'];",
            "const nodes = Object.fromEntries(ids.map((id) => [id, new Node(id)]));",
            "nodes['model-provider'].value = 'oauth_gateway';",
            "nodes['model-access-token'].value = 'ui-secret-token';",
            "nodes['model-refresh-token'].value = 'ui-refresh-token';",
            "const documentRef = { getElementById(id) { return nodes[id] || null; }, createElement(id) { return new Node(id); } };",
            "const calls = [];",
            "const fetcher = (url, options) => {",
            "  calls.push({ url, body: options.body });",
            "  return Promise.resolve({ ok: true, json: () => Promise.resolve({ ok: true, provider: 'oauth_gateway', logged_in: true }) });",
            "};",
            "async function main() {",
            "  auth.mount(documentRef, fetcher);",
            "  await nodes['model-auth-form'].trigger('submit');",
            "  if (calls[0].url !== '/api/model/auth/login') throw new Error('bad auth endpoint');",
            "  if (!calls[0].body.includes('ui-secret-token')) throw new Error('token not submitted');",
            "  if (!nodes['model-auth-status'].textContent.includes('oauth_gateway logged in')) throw new Error('status not rendered');",
            "  if (nodes['model-auth-status'].textContent.includes('ui-secret-token')) throw new Error('token leaked to UI');",
            "}",
            "main().catch((error) => { console.error(error); process.exit(1); });",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert 'id="model-auth-form"' in html
    assert 'id="model-access-token"' in html
    assert 'id="model-auth-status"' in html
    assert "run_bundle_model_auth.js" in html
    assert result.returncode == 0, result.stdout + result.stderr
