from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from sim_agent.project_layout import discover_project_root
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str
from sim_agent.schemas.errors import SchemaValidationError


MCP_CONFIG_ENV: Final = "ATOMISTIC_SIM_AGENT_MCP_CONFIG"


class MCPTransport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    name: str
    description: str
    parameters: JsonMap


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    transport: MCPTransport
    command: str
    url: str
    args: tuple[str, ...]
    tools: tuple[MCPToolDescriptor, ...]
    env: JsonMap
    env_file: str
    headers: JsonMap
    allow_write_tools: bool
    timeout_s: float


@dataclass(frozen=True, slots=True)
class MCPConfig:
    servers: tuple[MCPServerConfig, ...]
    source_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MCPListToolsResult:
    status: str
    blocker: str
    server_name: str
    transport: str
    tools: tuple[MCPToolDescriptor, ...]
    configured_servers: tuple[str, ...]
    config_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MCPCallToolResult:
    status: str
    blocker: str
    server_name: str
    transport: str
    tool_name: str
    initialize_result: JsonMap
    listed_tools: tuple[MCPToolDescriptor, ...]
    call_result: JsonMap
    configured_servers: tuple[str, ...]
    config_sources: tuple[str, ...]
    request_methods: tuple[str, ...]


def list_mcp_tools(server_name: str = "") -> MCPListToolsResult:
    config = load_mcp_config()
    if not config.servers:
        return _blocked("mcp_server_not_configured", server_name, config)
    if server_name:
        server = _server_by_name(config, server_name)
        if server is None:
            return _blocked("mcp_server_not_configured", server_name, config)
        return _listed(server, config)
    if len(config.servers) == 1:
        return _listed(config.servers[0], config)
    return MCPListToolsResult(
        status="succeeded",
        blocker="",
        server_name="",
        transport="mixed",
        tools=tuple(tool for server in config.servers for tool in server.tools),
        configured_servers=tuple(server.name for server in config.servers),
        config_sources=config.source_paths,
    )


def call_mcp_tool(server_name: str, tool_name: str, arguments: JsonMap) -> MCPCallToolResult:
    config = load_mcp_config()
    server = _server_by_name(config, server_name)
    if server is None:
        return _call_blocked("mcp_server_not_configured", server_name, "", tool_name, config, ())
    if _write_tool_blocker(server, tool_name):
        return _call_blocked("mcp_tool_write_not_allowed", server_name, server.transport.value, tool_name, config, ())
    try:
        return _call_configured_tool(server, tool_name, arguments, config)
    except SchemaValidationError as exc:
        return _call_blocked("invalid_arguments", server_name, server.transport.value, tool_name, config, (), {"error": str(exc)})
    except MCPTransportError as exc:
        return _call_blocked(exc.blocker, server_name, server.transport.value, tool_name, config, exc.request_methods, exc.payload)


def load_mcp_config() -> MCPConfig:
    servers: dict[str, MCPServerConfig] = {}
    sources: list[str] = []
    for path in _candidate_config_paths():
        if not path.is_file():
            continue
        payload = as_mapping(json.loads(path.read_text(encoding="utf-8")), "mcp_config")
        sources.append(str(path))
        for server in _servers_from_payload(payload):
            servers[server.name] = server
    return MCPConfig(servers=tuple(servers.values()), source_paths=tuple(sources))


def mcp_list_tools_payload(result: MCPListToolsResult) -> JsonMap:
    return {
        "server_name": result.server_name,
        "transport": result.transport,
        "configured_servers": list(result.configured_servers),
        "config_sources": list(result.config_sources),
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in result.tools
        ],
    }


def mcp_call_tool_payload(result: MCPCallToolResult) -> JsonMap:
    return {
        "status": result.status,
        "blocker": result.blocker,
        "server_name": result.server_name,
        "transport": result.transport,
        "tool_name": result.tool_name,
        "configured_servers": list(result.configured_servers),
        "config_sources": list(result.config_sources),
        "request_methods": list(result.request_methods),
        "initialize_result": dict(result.initialize_result),
        "listed_tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in result.listed_tools
        ],
        "call_result": dict(result.call_result),
    }


def _candidate_config_paths() -> tuple[Path, ...]:
    env_path = os.environ.get(MCP_CONFIG_ENV, "").strip()
    env_paths = (Path(env_path).expanduser(),) if env_path else ()
    project_path = discover_project_root() / ".asa" / "mcp-config.json"
    user_path = Path.home() / ".asa" / "mcp-config.json"
    return (user_path, project_path, *env_paths)


def _servers_from_payload(payload: JsonMap) -> tuple[MCPServerConfig, ...]:
    if payload.get("mcpServers") is not None:
        return tuple(
            _server_from_named_payload(name, as_mapping(server, f"mcpServers.{name}"))
            for name, server in as_mapping(payload["mcpServers"], "mcpServers").items()
        )
    return tuple(
        _server_from_payload(as_mapping(server, "server"))
        for server in as_sequence(payload.get("servers", ()), "servers")
    )


def _server_from_named_payload(name: str, payload: JsonMap) -> MCPServerConfig:
    return _server_from_payload({"name": name, **dict(payload)})


def _server_from_payload(payload: JsonMap) -> MCPServerConfig:
    transport = _transport_from_text(as_str(payload.get("transport", payload.get("type", "stdio")), "transport"))
    tools = tuple(
        _tool_from_payload(as_mapping(tool, "tool"))
        for tool in as_sequence(payload.get("tools", ()), "tools")
    )
    return MCPServerConfig(
        name=as_str(payload.get("name"), "name"),
        transport=transport,
        command=_optional_text(payload, "command"),
        url=_optional_text(payload, "url"),
        args=tuple(as_str(item, "arg") for item in as_sequence(payload.get("args", ()), "args")),
        tools=tools,
        env=_string_map(payload.get("env", {}), "env"),
        env_file=_optional_text(payload, "envFile"),
        headers=_string_map(payload.get("headers", {}), "headers"),
        allow_write_tools=_optional_bool(payload, "allow_write_tools", False),
        timeout_s=_optional_float(payload, "timeout_s", 30.0),
    )


def _tool_from_payload(payload: JsonMap) -> MCPToolDescriptor:
    parameters = as_mapping(
        payload.get("parameters", payload.get("inputSchema", {"type": "object", "properties": {}})),
        "parameters",
    )
    return MCPToolDescriptor(
        name=as_str(payload.get("name"), "tool.name"),
        description=_optional_text(payload, "description"),
        parameters=parameters,
    )


def _transport_from_text(value: str) -> MCPTransport:
    match value:
        case MCPTransport.STDIO.value:
            return MCPTransport.STDIO
        case MCPTransport.HTTP.value:
            return MCPTransport.HTTP
        case unreachable:
            raise SchemaValidationError(f"unsupported MCP transport:{unreachable}")


def _server_by_name(config: MCPConfig, server_name: str) -> MCPServerConfig | None:
    for server in config.servers:
        if server.name == server_name:
            return server
    return None


def _listed(server: MCPServerConfig, config: MCPConfig) -> MCPListToolsResult:
    return MCPListToolsResult(
        status="succeeded",
        blocker="",
        server_name=server.name,
        transport=server.transport.value,
        tools=server.tools,
        configured_servers=tuple(item.name for item in config.servers),
        config_sources=config.source_paths,
    )


class MCPTransportError(RuntimeError):
    def __init__(
        self,
        blocker: str,
        payload: JsonMap | None = None,
        request_methods: tuple[str, ...] = (),
    ) -> None:
        super().__init__(blocker)
        self.blocker = blocker
        self.payload = payload or {}
        self.request_methods = request_methods


def _call_configured_tool(
    server: MCPServerConfig,
    tool_name: str,
    arguments: JsonMap,
    config: MCPConfig,
) -> MCPCallToolResult:
    match server.transport:
        case MCPTransport.HTTP:
            initialize, listed, called, methods = _http_rpc_sequence(server, tool_name, arguments)
        case MCPTransport.STDIO:
            initialize, listed, called, methods = _stdio_rpc_sequence(server, tool_name, arguments)
    listed_tools = _listed_tools_from_rpc(server, listed)
    if not _tool_is_available(tool_name, listed_tools):
        raise MCPTransportError(
            "mcp_tool_not_available",
            {"tool_name": tool_name, "available_tools": [tool.name for tool in listed_tools]},
            methods,
        )
    if bool(called.get("isError")):
        raise MCPTransportError("mcp_tool_returned_error", called, methods)
    return MCPCallToolResult(
        status="succeeded",
        blocker="",
        server_name=server.name,
        transport=server.transport.value,
        tool_name=tool_name,
        initialize_result=initialize,
        listed_tools=listed_tools,
        call_result=called,
        configured_servers=tuple(item.name for item in config.servers),
        config_sources=config.source_paths,
        request_methods=methods,
    )


def _http_rpc_sequence(
    server: MCPServerConfig,
    tool_name: str,
    arguments: JsonMap,
) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
    if not server.url:
        raise MCPTransportError("mcp_server_url_required")
    requests = _rpc_requests(tool_name, arguments)
    responses = tuple(_http_rpc(server, request) for request in requests)
    return (
        as_mapping(responses[0].get("result", {}), "initialize.result"),
        as_mapping(responses[1].get("result", {}), "tools.list.result"),
        as_mapping(responses[2].get("result", {}), "tools.call.result"),
        _request_methods(requests),
    )


def _stdio_rpc_sequence(
    server: MCPServerConfig,
    tool_name: str,
    arguments: JsonMap,
) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
    if not server.command:
        raise MCPTransportError("mcp_server_command_required")
    requests = _rpc_requests(tool_name, arguments)
    env = os.environ.copy()
    if server.env_file:
        env.update(_env_file_values(server.env_file))
    env.update({key: as_str(value, f"env.{key}") for key, value in server.env.items()})
    try:
        completed = subprocess.run(
            [server.command, *server.args],
            input="".join(json.dumps(request) + "\n" for request in requests),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=server.timeout_s,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MCPTransportError("mcp_stdio_transport_error", {"error": str(exc)}, _request_methods(requests)) from exc
    if completed.returncode != 0 and not _has_json_response(completed.stdout):
        raise MCPTransportError(
            "mcp_stdio_transport_error",
            {"returncode": completed.returncode, "stderr": completed.stderr[-2000:]},
            _request_methods(requests),
        )
    responses = _stdio_responses(completed.stdout, requests)
    return (
        as_mapping(responses[0].get("result", {}), "initialize.result"),
        as_mapping(responses[1].get("result", {}), "tools.list.result"),
        as_mapping(responses[2].get("result", {}), "tools.call.result"),
        _request_methods(requests),
    )


def _http_rpc(server: MCPServerConfig, payload: JsonMap) -> JsonMap:
    body = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}
    headers.update({as_str(k, "header"): as_str(v, f"headers.{k}") for k, v in server.headers.items()})
    method = as_str(payload["method"], "method")
    request = urllib.request.Request(server.url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=server.timeout_s) as response:
            return _json_rpc_response(response.read().decode("utf-8"), method)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise MCPTransportError("mcp_http_transport_error", {"status": exc.code, "body": raw[-2000:]}, (method,)) from exc
    except OSError as exc:
        raise MCPTransportError("mcp_http_transport_error", {"error": str(exc)}, (method,)) from exc


def _json_rpc_response(raw: str, method: str) -> JsonMap:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPTransportError("mcp_json_response_required", {"method": method}, (method,)) from exc
    response = as_mapping(value, f"{method}.response")
    error = response.get("error")
    if isinstance(error, dict):
        raise MCPTransportError("mcp_json_rpc_error", {"method": method, "error": error}, (method,))
    if "result" not in response:
        raise MCPTransportError("mcp_json_rpc_result_required", {"method": method}, (method,))
    return response


def _stdio_responses(raw: str, requests: tuple[JsonMap, ...]) -> tuple[JsonMap, ...]:
    by_id: dict[object, JsonMap] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            response = as_mapping(json.loads(line), "stdio.response")
        except (json.JSONDecodeError, SchemaValidationError):
            continue
        if response.get("id") is not None:
            by_id[response["id"]] = response
    responses: list[JsonMap] = []
    for request in requests:
        response = by_id.get(request.get("id"))
        if response is None:
            raise MCPTransportError("mcp_stdio_response_missing", {"id": request.get("id")}, _request_methods(requests))
        error = response.get("error")
        if isinstance(error, dict):
            raise MCPTransportError("mcp_json_rpc_error", {"method": request.get("method"), "error": error}, _request_methods(requests))
        responses.append(response)
    return tuple(responses)


def _has_json_response(raw: str) -> bool:
    return any(line.strip().startswith("{") for line in raw.splitlines())


def _rpc_requests(tool_name: str, arguments: JsonMap) -> tuple[JsonMap, JsonMap, JsonMap]:
    return (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "asa-runtime", "version": "0.1"},
            },
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": dict(arguments)},
        },
    )


def _request_methods(requests: tuple[JsonMap, ...]) -> tuple[str, ...]:
    return tuple(as_str(request["method"], "method") for request in requests)


def _listed_tools_from_rpc(server: MCPServerConfig, listed: JsonMap) -> tuple[MCPToolDescriptor, ...]:
    tools = tuple(
        _tool_from_payload(as_mapping(tool, "tools.list.tool"))
        for tool in as_sequence(listed.get("tools", ()), "tools")
    )
    return tools or server.tools


def _tool_is_available(tool_name: str, tools: tuple[MCPToolDescriptor, ...]) -> bool:
    return any(tool.name == tool_name for tool in tools)


def _write_tool_blocker(server: MCPServerConfig, tool_name: str) -> bool:
    if server.allow_write_tools:
        return False
    normalized = tool_name.strip().lower()
    return any(
        part in normalized
        for part in ("write", "delete", "create", "update", "merge", "execute", "unsafe", "run_code")
    )


def _call_blocked(
    blocker: str,
    server_name: str,
    transport: str,
    tool_name: str,
    config: MCPConfig,
    request_methods: tuple[str, ...],
    payload: JsonMap | None = None,
) -> MCPCallToolResult:
    return MCPCallToolResult(
        status="blocked",
        blocker=blocker,
        server_name=server_name,
        transport=transport,
        tool_name=tool_name,
        initialize_result={},
        listed_tools=(),
        call_result=payload or {},
        configured_servers=tuple(server.name for server in config.servers),
        config_sources=config.source_paths,
        request_methods=request_methods,
    )


def _blocked(blocker: str, server_name: str, config: MCPConfig) -> MCPListToolsResult:
    return MCPListToolsResult(
        status="blocked",
        blocker=blocker,
        server_name=server_name,
        transport="",
        tools=(),
        configured_servers=tuple(server.name for server in config.servers),
        config_sources=config.source_paths,
    )


def _optional_text(payload: JsonMap, field: str) -> str:
    value = payload.get(field)
    if value is None:
        return ""
    return as_str(value, field)


def _optional_bool(payload: JsonMap, field: str, default: bool) -> bool:
    value = payload.get(field, default)
    if isinstance(value, bool):
        return value
    raise SchemaValidationError(f"{field} must be a boolean")


def _optional_float(payload: JsonMap, field: str, default: float) -> float:
    value = payload.get(field, default)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise SchemaValidationError(f"{field} must be a number")


def _string_map(value: object, field: str) -> JsonMap:
    mapping = as_mapping(value, field)
    for key, item in mapping.items():
        as_str(key, f"{field}.key")
        as_str(item, f"{field}.{key}")
    return mapping


def _env_file_values(path_text: str) -> dict[str, str]:
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise MCPTransportError("mcp_env_file_not_found", {"envFile": str(path)})
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _unquote_env_value(value.strip())
    return values


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
