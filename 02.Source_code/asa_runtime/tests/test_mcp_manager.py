from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import TracebackType

import pytest

from sim_agent.schemas._parse import as_mapping, as_sequence, as_str


SOURCE_ROOT = Path(__file__).resolve().parents[1]

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_mcp_list_tools_blocks_when_no_server_configured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: isolated user and project config roots with no MCP config file.
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(project))
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", raising=False)

    # When: an agent asks the runtime to list MCP tools.
    result = execute_runtime_tool(
        RuntimeToolCall(
            tool_name="mcp_list_tools",
            arguments={"server_name": "neo4j-memory"},
            run_id="tool-run",
            session_id="tool-session",
        ),
        default_tool_registry(),
        tmp_path,
    )

    # Then: the tool fails closed instead of returning a descriptor-only success.
    assert result.status == "blocked"
    assert result.blocker == "mcp_server_not_configured"
    assert result.output["server_name"] == "neo4j-memory"
    assert as_sequence(result.output["tools"], "tools") == []
    assert (tmp_path / result.artifact_ref).is_file()


def test_mcp_list_tools_reads_fake_stdio_and_http_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: user and project MCP configs with fake stdio and HTTP servers.
    from sim_agent.knowledge.mcp_manager import list_mcp_tools, mcp_list_tools_payload

    project = tmp_path / "project"
    config_dir = project / ".asa"
    config_dir.mkdir(parents=True)
    home = tmp_path / "home"
    user_config_dir = home / ".asa"
    user_config_dir.mkdir(parents=True)
    (user_config_dir / "mcp-config.json").write_text(
        """{
  "servers": [
    {
      "name": "stdio-memory",
      "transport": "stdio",
      "command": "python3",
      "args": ["fake_stdio_mcp.py"],
      "tools": [{"name": "graph_search", "description": "Search graph memory."}]
    }
  ]
}
""",
        encoding="utf-8",
    )
    (config_dir / "mcp-config.json").write_text(
        """{
  "servers": [
    {
      "name": "http-memory",
      "transport": "http",
      "url": "http://127.0.0.1:9999/mcp",
      "tools": [{"name": "graph_status", "description": "Read graph status."}]
    }
  ]
}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(project))
    monkeypatch.delenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", raising=False)

    # When: the manager lists configured fake server tools.
    stdio = mcp_list_tools_payload(list_mcp_tools("stdio-memory"))
    http = mcp_list_tools_payload(list_mcp_tools("http-memory"))

    # Then: both transports are parsed without opening a live Neo4j or MCP connection.
    assert stdio["server_name"] == "stdio-memory"
    assert stdio["transport"] == "stdio"
    stdio_tool = as_mapping(as_sequence(stdio["tools"], "tools")[0], "tool")
    http_tool = as_mapping(as_sequence(http["tools"], "tools")[0], "tool")
    assert as_str(stdio_tool["name"], "name") == "graph_search"
    assert http["server_name"] == "http-memory"
    assert http["transport"] == "http"
    assert as_str(http_tool["name"], "name") == "graph_status"


def test_mcp_call_tool_runs_fake_http_initialize_list_and_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim_agent.agent_harness.tools import RuntimeToolCall, default_tool_registry, execute_runtime_tool

    with _FakeMCPHTTPServer() as server:
        config_path = tmp_path / "mcp-config.json"
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "neo4j-memory": {
                            "transport": "http",
                            "url": server.url,
                            "tools": [
                                {
                                    "name": "read-cypher",
                                    "description": "Read Neo4j memory with Cypher.",
                                    "side_effect_class": "read",
                                    "read_only": True,
                                    "requires_approval": False,
                                    "allow_without_approval": True,
                                    "inputSchema": {
                                        "type": "object",
                                        "properties": {"query": {"type": "string"}},
                                        "required": ["query"],
                                    },
                                },
                                {
                                    "name": "write-cypher",
                                    "description": "Write Neo4j memory with Cypher.",
                                },
                            ],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))

        runtime_result = execute_runtime_tool(
            RuntimeToolCall(
                "mcp_call_tool",
                {
                    "server_name": "neo4j-memory",
                    "tool_name": "read-cypher",
                    "arguments": {"query": "MATCH (n) RETURN count(n) AS n"},
                },
                "tool-run",
                "tool-session",
            ),
            default_tool_registry(),
            tmp_path,
        )
        blocked_result = execute_runtime_tool(
            RuntimeToolCall(
                "mcp_call_tool",
                {
                    "server_name": "neo4j-memory",
                    "tool_name": "write-cypher",
                    "arguments": {"query": "CREATE (:X)"},
                },
                "tool-run",
                "tool-session",
            ),
            default_tool_registry(),
            tmp_path,
        )
        result = runtime_result.output
        blocked = blocked_result.output

    assert runtime_result.status == "succeeded"
    assert (tmp_path / runtime_result.artifact_ref).is_file()
    assert result["server_name"] == "neo4j-memory"
    assert result["transport"] == "http"
    assert result["tool_name"] == "read-cypher"
    assert result["request_methods"] == ["initialize", "tools/list", "tools/call"]
    assert result["call_result"]["structuredContent"] == {"rows": [{"n": 1}]}
    assert result["call_result"]["capability"] == {
        "tool_name": "read-cypher",
        "side_effect_class": "read",
        "read_only": True,
        "requires_approval": False,
        "allow_without_approval": True,
    }
    assert server.methods == ["initialize", "tools/list", "tools/call"]
    tool = as_mapping(as_sequence(result["listed_tools"], "listed_tools")[0], "tool")
    assert as_str(tool["name"], "name") == "read-cypher"
    assert blocked_result.status == "blocked"
    assert blocked_result.blocker == "mcp_tool_capability_unknown"
    assert (tmp_path / blocked_result.artifact_ref).is_file()
    assert blocked["tool_name"] == "write-cypher"
    assert blocked["request_methods"] == []
    assert blocked["call_result"] == {
        "capability": {
            "tool_name": "write-cypher",
            "side_effect_class": "unknown",
            "read_only": False,
            "requires_approval": True,
            "allow_without_approval": False,
        }
    }
    assert server.methods == ["initialize", "tools/list", "tools/call"]


def test_mcp_call_tool_blocks_unsafe_browser_code_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim_agent.knowledge.mcp_manager import call_mcp_tool, mcp_call_tool_payload

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "playwright": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": ["should-not-run.py"],
                        "allow_write_tools": False,
                        "tools": [{"name": "browser_run_code_unsafe"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))

    result = mcp_call_tool_payload(call_mcp_tool("playwright", "browser_run_code_unsafe", {}))

    assert result["status"] == "blocked"
    assert result["blocker"] == "mcp_tool_capability_unknown"
    assert result["request_methods"] == []


def test_mcp_stdio_supports_type_alias_and_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sim_agent.knowledge.mcp_manager import call_mcp_tool, mcp_call_tool_payload

    fake_server = tmp_path / "fake_stdio_mcp.py"
    fake_server.write_text(
        """
from __future__ import annotations

import json
import os
import sys

for raw in sys.stdin:
    request = json.loads(raw)
    method = request["method"]
    if method == "initialize":
        result = {"protocolVersion": "2025-06-18", "serverInfo": {"name": "fake-stdio", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "env-check", "description": "Check env.", "inputSchema": {"type": "object", "properties": {}}, "side_effect_class": "read", "read_only": True, "requires_approval": False, "allow_without_approval": True}]}
    elif method == "tools/call":
        result = {"structuredContent": {"neo4j_uri": os.environ.get("NEO4J_URI"), "read_only": os.environ.get("NEO4J_READ_ONLY")}, "isError": False}
    else:
        result = {"isError": True}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        'NEO4J_URI="bolt://localhost:7687"\nNEO4J_READ_ONLY=true\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "mcp-config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "local-neo4j": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(fake_server)],
                        "envFile": str(env_file),
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
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))

    result = mcp_call_tool_payload(call_mcp_tool("local-neo4j", "env-check", {}))

    assert result["status"] == "succeeded", result
    assert result["transport"] == "stdio"
    assert result["request_methods"] == ["initialize", "tools/list", "tools/call"]
    assert result["call_result"]["structuredContent"] == {
        "neo4j_uri": "bolt://localhost:7687",
        "read_only": "true",
    }


class _FakeMCPHTTPServer:
    def __init__(self) -> None:
        self._handler = type("_FakeMCPHTTPHandler", (_FakeMCPHTTPHandler,), {"methods": []})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/mcp"

    @property
    def methods(self) -> list[str]:
        return list(self._handler.methods)

    def __enter__(self) -> "_FakeMCPHTTPServer":
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


class _FakeMCPHTTPHandler(BaseHTTPRequestHandler):
    methods: list[str] = []

    def log_message(self, format: str, *args: str) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers["content-length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        method = as_str(payload["method"], "method")
        self.__class__.methods.append(method)
        self._write({"jsonrpc": "2.0", "id": payload["id"], "result": _result_for_method(method)})

    def _write(self, payload: dict[str, object]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _result_for_method(method: str) -> dict[str, object]:
    if method == "initialize":
        return {
            "protocolVersion": "2025-06-18",
            "serverInfo": {"name": "fake-neo4j-mcp", "version": "0.1"},
        }
    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "read-cypher",
                    "description": "Read Neo4j memory with Cypher.",
                    "side_effect_class": "read",
                    "read_only": True,
                    "requires_approval": False,
                    "allow_without_approval": True,
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ]
        }
    if method == "tools/call":
        return {
            "content": [{"type": "text", "text": "n=1"}],
            "structuredContent": {"rows": [{"n": 1}]},
            "isError": False,
        }
    return {"isError": True, "content": [{"type": "text", "text": f"unknown method {method}"}]}
