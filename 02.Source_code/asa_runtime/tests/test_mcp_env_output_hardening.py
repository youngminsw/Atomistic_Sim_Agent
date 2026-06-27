from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import TracebackType

import pytest

from sim_agent.schemas._parse import JsonMap


SECRET_ENV_NAME = "TASK8_SECRET_MARKER_PARENT"
SECRET_ENV_VALUE = "task8-parent-secret-value"
TOKEN_VALUE = "sk-task8SecretToken123456789"
REDACTION_MARKER = "[redacted]"


def test_mcp_stdio_does_not_inherit_secret_parent_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a parent-only secret and explicit MCP env values for a stdio server.
    from sim_agent.knowledge.mcp_manager import call_mcp_tool

    fake_server = _write_env_echo_server(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("FILE_SAFE_VALUE=file-allowed\n", encoding="utf-8")
    config_path = tmp_path / "mcp-config.json"
    _write_config(
        config_path,
        fake_server,
        env_file=env_file,
        env={"CONFIG_SAFE_VALUE": "config-allowed"},
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    monkeypatch.setenv(SECRET_ENV_NAME, SECRET_ENV_VALUE)
    _isolate_default_config_paths(tmp_path, monkeypatch)

    # When: the stdio tool runs in a child process.
    result = call_mcp_tool("local-stdio", "env-check", {})

    # Then: only explicit config/env-file values are visible to the child.
    assert result.status == "succeeded"
    assert result.call_result["structuredContent"] == {
        "config_safe": "config-allowed",
        "file_safe": "file-allowed",
        "parent_secret": None,
    }


def test_mcp_stdio_passes_explicit_allowed_env_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: safe values are provided only through MCP config and envFile.
    from sim_agent.knowledge.mcp_manager import call_mcp_tool

    fake_server = _write_env_echo_server(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text('FILE_SAFE_VALUE="quoted-file-allowed"\n', encoding="utf-8")
    config_path = tmp_path / "mcp-config.json"
    _write_config(
        config_path,
        fake_server,
        env_file=env_file,
        env={"CONFIG_SAFE_VALUE": "config-allowed"},
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    monkeypatch.setenv(SECRET_ENV_NAME, SECRET_ENV_VALUE)
    _isolate_default_config_paths(tmp_path, monkeypatch)

    # When: the stdio MCP server reads its environment.
    result = call_mcp_tool("local-stdio", "env-check", {})

    # Then: explicit values reach the child while unrelated parent secrets do not.
    assert result.status == "succeeded"
    structured = result.call_result["structuredContent"]
    assert structured["config_safe"] == "config-allowed"
    assert structured["file_safe"] == "quoted-file-allowed"
    assert structured["parent_secret"] is None


def test_mcp_errors_redact_stderr_and_http_body_tails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: stdio stderr and HTTP error bodies contain test secret markers and token-like strings.
    from sim_agent.knowledge.mcp_manager import call_mcp_tool

    stderr_server = tmp_path / "stderr_mcp.py"
    stderr_server.write_text(
        f"""
from __future__ import annotations

import sys

sys.stderr.write("diagnostic before {SECRET_ENV_NAME}={SECRET_ENV_VALUE} token={TOKEN_VALUE}\\n")
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    stdio_config = tmp_path / "stdio-config.json"
    _write_config(stdio_config, stderr_server)
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(stdio_config))
    _isolate_default_config_paths(tmp_path, monkeypatch)

    # When: stdio transport fails before returning JSON responses.
    stdio_result = call_mcp_tool("local-stdio", "env-check", {})

    # Then: diagnostics remain bounded and useful without secret literals.
    assert stdio_result.status == "blocked"
    assert stdio_result.blocker == "mcp_stdio_transport_error"
    assert "diagnostic before" in str(stdio_result.call_result)
    assert REDACTION_MARKER in str(stdio_result.call_result)
    assert SECRET_ENV_NAME not in str(stdio_result.call_result)
    assert SECRET_ENV_VALUE not in str(stdio_result.call_result)
    assert TOKEN_VALUE not in str(stdio_result.call_result)

    with _SecretHTTPErrorServer() as server:
        http_config = tmp_path / "http-config.json"
        _write_config(http_config, tmp_path / "unused.py", transport="http", url=server.url)
        monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(http_config))

        # When: HTTP transport returns an error body with secrets.
        http_result = call_mcp_tool("local-stdio", "env-check", {})

    # Then: HTTP body tails are redacted before reaching runtime-visible output.
    assert http_result.status == "blocked"
    assert http_result.blocker == "mcp_http_transport_error"
    assert "http diagnostic" in str(http_result.call_result)
    assert REDACTION_MARKER in str(http_result.call_result)
    assert SECRET_ENV_NAME not in str(http_result.call_result)
    assert SECRET_ENV_VALUE not in str(http_result.call_result)
    assert TOKEN_VALUE not in str(http_result.call_result)


def test_mcp_json_rpc_error_payload_is_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a stdio MCP server returns a JSON-RPC error containing secrets in stdout.
    from sim_agent.knowledge.mcp_manager import call_mcp_tool

    error_server = tmp_path / "json_error_mcp.py"
    error_server.write_text(
        f"""
from __future__ import annotations

import json
import sys

for raw in sys.stdin:
    request = json.loads(raw)
    if request["method"] == "tools/call":
        result = {{"error": {{"message": "result leak {SECRET_ENV_NAME}={SECRET_ENV_VALUE} token={TOKEN_VALUE}"}}}}
    else:
        result = {{"result": {{"tools": [{{"name": "env-check", "side_effect_class": "read", "read_only": True, "requires_approval": False, "allow_without_approval": True}}]}}}}
    print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], **result}}), flush=True)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp-config.json"
    _write_config(config_path, error_server)
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    _isolate_default_config_paths(tmp_path, monkeypatch)

    # When: the JSON-RPC response reports an error.
    result = call_mcp_tool("local-stdio", "env-check", {})

    # Then: stdout-derived JSON-RPC error payloads do not expose secret literals.
    assert result.status == "blocked"
    assert result.blocker == "mcp_json_rpc_error"
    assert "result leak" in str(result.call_result)
    assert REDACTION_MARKER in str(result.call_result)
    assert SECRET_ENV_NAME not in str(result.call_result)
    assert SECRET_ENV_VALUE not in str(result.call_result)
    assert TOKEN_VALUE not in str(result.call_result)


class _SecretHTTPErrorServer:
    def __init__(self) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _SecretHTTPErrorHandler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/mcp"

    def __enter__(self) -> "_SecretHTTPErrorServer":
        self._thread.start()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _SecretHTTPErrorHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: str) -> None:
        return

    def do_POST(self) -> None:
        raw = (
            f"http diagnostic {SECRET_ENV_NAME}={SECRET_ENV_VALUE} "
            f"authorization: Bearer {TOKEN_VALUE}"
        ).encode("utf-8")
        self.send_response(500)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _write_env_echo_server(tmp_path: Path) -> Path:
    server = tmp_path / "env_echo_mcp.py"
    server.write_text(
        f"""
from __future__ import annotations

import json
import os
import sys

for raw in sys.stdin:
    request = json.loads(raw)
    method = request["method"]
    if method == "initialize":
        result = {{"protocolVersion": "2025-06-18"}}
    elif method == "tools/list":
        result = {{"tools": [{{"name": "env-check", "inputSchema": {{"type": "object", "properties": {{}}}}, "side_effect_class": "read", "read_only": True, "requires_approval": False, "allow_without_approval": True}}]}}
    elif method == "tools/call":
        result = {{
            "structuredContent": {{
                "config_safe": os.environ.get("CONFIG_SAFE_VALUE"),
                "file_safe": os.environ.get("FILE_SAFE_VALUE"),
                "parent_secret": os.environ.get("{SECRET_ENV_NAME}"),
            }},
            "isError": False,
        }}
    else:
        result = {{"isError": True}}
    print(json.dumps({{"jsonrpc": "2.0", "id": request["id"], "result": result}}), flush=True)
""",
        encoding="utf-8",
    )
    return server


def _write_config(
    path: Path,
    server_script: Path,
    *,
    env_file: Path | None = None,
    env: JsonMap | None = None,
    transport: str = "stdio",
    url: str = "",
) -> None:
    server: dict[str, object] = {
        "transport": transport,
        "tools": [
            {
                "name": "env-check",
                "side_effect_class": "read",
                "read_only": True,
                "requires_approval": False,
                "allow_without_approval": True,
            }
        ],
    }
    if transport == "stdio":
        server["command"] = sys.executable
        server["args"] = [str(server_script)]
    if url:
        server["url"] = url
    if env_file is not None:
        server["envFile"] = str(env_file)
    if env is not None:
        server["env"] = dict(env)
    path.write_text(json.dumps({"mcpServers": {"local-stdio": server}}), encoding="utf-8")


def _isolate_default_config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(project))
