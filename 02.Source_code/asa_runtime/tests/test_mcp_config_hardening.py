from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim_agent.schemas._parse import JsonMap, as_sequence


def test_corrupt_mcp_config_returns_typed_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the only configured MCP source contains invalid JSON.
    from sim_agent.knowledge import mcp_manager

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text('{"mcpServers": {"neo4j-memory": ', encoding="utf-8")
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    _isolate_default_config_paths(tmp_path, monkeypatch)
    launches = _LaunchSpy()
    monkeypatch.setattr(mcp_manager, "_http_rpc_sequence", launches.http)
    monkeypatch.setattr(mcp_manager, "_stdio_rpc_sequence", launches.stdio)

    # When: configured MCP tools are listed through the manager boundary.
    result = mcp_manager.list_mcp_tools("neo4j-memory")

    # Then: callers receive a typed blocker and no MCP transport is launched.
    assert result.status == "blocked"
    assert result.blocker == "corrupt_mcp_config"
    assert result.server_name == "neo4j-memory"
    assert result.config_sources == (str(config_path),)
    assert result.configured_servers == ()
    assert result.tools == ()
    assert launches.calls == ()


def test_corrupt_mcp_config_blocks_tool_listing_and_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the MCP config file is corrupt and transport launchers are observed.
    from sim_agent.knowledge import mcp_manager

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text('{"servers": [', encoding="utf-8")
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    _isolate_default_config_paths(tmp_path, monkeypatch)
    launches = _LaunchSpy()
    monkeypatch.setattr(mcp_manager, "_http_rpc_sequence", launches.http)
    monkeypatch.setattr(mcp_manager, "_stdio_rpc_sequence", launches.stdio)

    # When: callers try both MCP listing and tool invocation.
    listed = mcp_manager.list_mcp_tools("neo4j-memory")
    called = mcp_manager.call_mcp_tool("neo4j-memory", "read-cypher", {"query": "RETURN 1"})

    # Then: both calls fail closed before subprocess or HTTP setup.
    assert listed.status == "blocked"
    assert listed.blocker == "corrupt_mcp_config"
    assert called.status == "blocked"
    assert called.blocker == "corrupt_mcp_config"
    assert called.request_methods == ()
    assert called.call_result == {}
    assert called.config_sources == (str(config_path),)
    assert launches.calls == ()


def test_invalid_mcp_config_entry_blocks_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the MCP config is valid JSON but contains an invalid server entry.
    from sim_agent.knowledge import mcp_manager

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text(json.dumps({"servers": [{"name": "broken", "transport": "pipe"}]}), encoding="utf-8")
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    _isolate_default_config_paths(tmp_path, monkeypatch)
    launches = _LaunchSpy()
    monkeypatch.setattr(mcp_manager, "_http_rpc_sequence", launches.http)
    monkeypatch.setattr(mcp_manager, "_stdio_rpc_sequence", launches.stdio)

    # When: a tool call is attempted against the invalid config.
    result = mcp_manager.call_mcp_tool("broken", "read", {})

    # Then: schema corruption is typed and no MCP transport is launched.
    assert result.status == "blocked"
    assert result.blocker == "invalid_mcp_config"
    assert result.config_sources == (str(config_path),)
    assert result.request_methods == ()
    assert launches.calls == ()


def test_valid_mcp_config_lists_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a valid MCP config with configured descriptor metadata.
    from sim_agent.knowledge.mcp_manager import list_mcp_tools, mcp_list_tools_payload

    config_path = tmp_path / "mcp-config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "neo4j-memory": {
                        "transport": "stdio",
                        "command": "python3",
                        "args": ["fake.py"],
                        "tools": [
                            {
                                "name": "read-cypher",
                                "description": "Read Neo4j memory.",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ATOMISTIC_SIM_AGENT_MCP_CONFIG", str(config_path))
    _isolate_default_config_paths(tmp_path, monkeypatch)

    # When: the configured server is listed.
    payload = mcp_list_tools_payload(list_mcp_tools("neo4j-memory"))

    # Then: the descriptor metadata is returned without launching a transport.
    tools = as_sequence(payload["tools"], "tools")
    assert payload["server_name"] == "neo4j-memory"
    assert payload["transport"] == "stdio"
    assert payload["configured_servers"] == ["neo4j-memory"]
    assert payload["config_sources"] == [str(config_path)]
    assert len(tools) == 1


class _LaunchSpy:
    def __init__(self) -> None:
        self.calls: tuple[str, ...] = ()

    def http(self, *_args: object) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
        self.calls = (*self.calls, "http")
        raise AssertionError("HTTP MCP transport launched")

    def stdio(self, *_args: object) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
        self.calls = (*self.calls, "stdio")
        raise AssertionError("stdio MCP transport launched")


def _isolate_default_config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ASA_PROJECT_ROOT", str(project))
