from __future__ import annotations

import json
import os
import re
import selectors
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, assert_never

from sim_agent.project_layout import discover_project_root
from sim_agent.schemas._parse import JsonMap, as_mapping, as_sequence, as_str
from sim_agent.schemas.errors import SchemaValidationError


MCP_CONFIG_ENV: Final = "ATOMISTIC_SIM_AGENT_MCP_CONFIG"
REDACTION_MARKER: Final = "[redacted]"
MCP_SUBPROCESS_ENV_ALLOWLIST: Final = (
    "PATH",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "NODE_EXTRA_CA_CERTS",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)
SECRET_VALUE_PATTERN: Final = re.compile(
    r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)"
    r"(\s*[:=]\s*)(bearer\s+)?([^\s,;}\"]+)"
)
TOKEN_PATTERN: Final = re.compile(r"\b(?:sk|pk|ghp|gho|github_pat|xox[baprs])-[A-Za-z0-9_=-]{8,}\b")
TEST_SECRET_MARKER_PATTERN: Final = re.compile(r"TASK8_SECRET_MARKER[^\s,;}\"]*")
TAIL_LIMIT: Final = 2000


class MCPTransport(StrEnum):
    STDIO = "stdio"
    HTTP = "http"


class MCPSideEffectClass(StrEnum):
    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    PROCESS = "process"
    CREDENTIAL = "credential"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class MCPToolCapability:
    tool_name: str
    side_effect_class: MCPSideEffectClass
    read_only: bool
    requires_approval: bool
    allow_without_approval: bool
    declared: bool


@dataclass(frozen=True, slots=True)
class MCPToolDescriptor:
    name: str
    description: str
    parameters: JsonMap
    capability: MCPToolCapability


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
    blocker: str = ""


@dataclass(frozen=True, slots=True)
class _ParsedConfigFile:
    payload: JsonMap
    blocker: str


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
    if config.blocker:
        return _blocked(config.blocker, server_name, config)
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
    if config.blocker:
        return _call_blocked(config.blocker, server_name, "", tool_name, config, ())
    server = _server_by_name(config, server_name)
    if server is None:
        return _call_blocked("mcp_server_not_configured", server_name, "", tool_name, config, ())
    configured_tool = _tool_by_name(server.tools, tool_name)
    capability = configured_tool.capability if configured_tool is not None else _unknown_capability(tool_name)
    policy_blocker = _capability_policy_blocker(server, capability)
    if policy_blocker:
        return _call_blocked(
            policy_blocker,
            server_name,
            server.transport.value,
            tool_name,
            config,
            (),
            {"capability": _capability_payload(capability)},
        )
    try:
        if configured_tool is None:
            return _call_blocked(
                "mcp_tool_capability_unknown",
                server_name,
                server.transport.value,
                tool_name,
                config,
                (),
                {"capability": _capability_payload(capability)},
            )
        return _call_configured_tool(server, configured_tool, arguments, config)
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
        sources.append(str(path))
        parsed = _parse_config_file(path)
        if parsed.blocker:
            return MCPConfig(servers=(), source_paths=tuple(sources), blocker=parsed.blocker)
        try:
            for server in _servers_from_payload(parsed.payload):
                servers[server.name] = server
        except SchemaValidationError:
            return MCPConfig(servers=(), source_paths=tuple(sources), blocker="invalid_mcp_config")
    return MCPConfig(servers=tuple(servers.values()), source_paths=tuple(sources))


def _parse_config_file(path: Path) -> _ParsedConfigFile:
    try:
        payload = as_mapping(json.loads(path.read_text(encoding="utf-8")), "mcp_config")
    except json.JSONDecodeError:
        return _ParsedConfigFile(payload={}, blocker="corrupt_mcp_config")
    except OSError:
        return _ParsedConfigFile(payload={}, blocker="mcp_config_unreadable")
    except SchemaValidationError:
        return _ParsedConfigFile(payload={}, blocker="invalid_mcp_config")
    return _ParsedConfigFile(payload=payload, blocker="")


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
                "capability": _capability_payload(tool.capability),
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
    name = as_str(payload.get("name"), "tool.name")
    return MCPToolDescriptor(
        name=name,
        description=_optional_text(payload, "description"),
        parameters=parameters,
        capability=_capability_from_payload(name, payload),
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
    configured_tool: MCPToolDescriptor,
    arguments: JsonMap,
    config: MCPConfig,
) -> MCPCallToolResult:
    match server.transport:
        case MCPTransport.HTTP:
            initialize, listed, called, methods = _http_rpc_sequence(server, configured_tool, arguments)
        case MCPTransport.STDIO:
            initialize, listed, called, methods = _stdio_rpc_sequence(server, configured_tool, arguments)
        case unreachable:
            assert_never(unreachable)
    listed_tools = _listed_tools_from_rpc(listed)
    if bool(called.get("isError")):
        raise MCPTransportError("mcp_tool_returned_error", _redacted_json_map(called), methods)
    call_result = _redacted_json_map(called)
    call_result["capability"] = _capability_payload(configured_tool.capability)
    return MCPCallToolResult(
        status="succeeded",
        blocker="",
        server_name=server.name,
        transport=server.transport.value,
        tool_name=configured_tool.name,
        initialize_result=initialize,
        listed_tools=listed_tools,
        call_result=call_result,
        configured_servers=tuple(item.name for item in config.servers),
        config_sources=config.source_paths,
        request_methods=methods,
    )


def _http_rpc_sequence(
    server: MCPServerConfig,
    configured_tool: MCPToolDescriptor,
    arguments: JsonMap,
) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
    if not server.url:
        raise MCPTransportError("mcp_server_url_required")
    initialize_request = _initialize_request()
    list_request = _tools_list_request()
    try:
        initialize_response = _http_rpc(server, initialize_request)
    except MCPTransportError as exc:
        raise _transport_error_with_methods(exc, ("initialize",)) from exc
    try:
        list_response = _http_rpc(server, list_request)
    except MCPTransportError as exc:
        raise _transport_error_with_methods(exc, ("initialize", "tools/list")) from exc
    listed = as_mapping(list_response.get("result", {}), "tools.list.result")
    _validate_live_tool_authority(server, configured_tool, listed, ("initialize", "tools/list"))
    call_request = _tools_call_request(configured_tool.name, arguments)
    try:
        call_response = _http_rpc(server, call_request)
    except MCPTransportError as exc:
        raise _transport_error_with_methods(exc, ("initialize", "tools/list", "tools/call")) from exc
    return (
        as_mapping(initialize_response.get("result", {}), "initialize.result"),
        listed,
        as_mapping(call_response.get("result", {}), "tools.call.result"),
        ("initialize", "tools/list", "tools/call"),
    )


def _stdio_rpc_sequence(
    server: MCPServerConfig,
    configured_tool: MCPToolDescriptor,
    arguments: JsonMap,
) -> tuple[JsonMap, JsonMap, JsonMap, tuple[str, ...]]:
    if not server.command:
        raise MCPTransportError("mcp_server_command_required")
    env = _stdio_subprocess_env(server)
    try:
        process = subprocess.Popen(
            [server.command, *server.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except OSError as exc:
        raise MCPTransportError("mcp_stdio_transport_error", {"error": _redact_text(str(exc))}) from exc
    try:
        initialize_response = _stdio_request_response(process, _initialize_request(), ("initialize",), server.timeout_s)
        list_response = _stdio_request_response(
            process,
            _tools_list_request(),
            ("initialize", "tools/list"),
            server.timeout_s,
        )
        listed = as_mapping(list_response.get("result", {}), "tools.list.result")
        _validate_live_tool_authority(server, configured_tool, listed, ("initialize", "tools/list"))
        call_response = _stdio_request_response(
            process,
            _tools_call_request(configured_tool.name, arguments),
            ("initialize", "tools/list", "tools/call"),
            server.timeout_s,
        )
    finally:
        _close_stdio_process(process)
    return (
        as_mapping(initialize_response.get("result", {}), "initialize.result"),
        listed,
        as_mapping(call_response.get("result", {}), "tools.call.result"),
        ("initialize", "tools/list", "tools/call"),
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
        raise MCPTransportError("mcp_http_transport_error", {"status": exc.code, "body": _redacted_tail(raw)}, (method,)) from exc
    except OSError as exc:
        raise MCPTransportError("mcp_http_transport_error", {"error": _redact_text(str(exc))}, (method,)) from exc


def _json_rpc_response(raw: str, method: str) -> JsonMap:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPTransportError("mcp_json_response_required", {"method": method}, (method,)) from exc
    response = as_mapping(value, f"{method}.response")
    error = response.get("error")
    if isinstance(error, dict):
        raise MCPTransportError("mcp_json_rpc_error", {"method": method, "error": _redacted_json_map(error)}, (method,))
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
            raise MCPTransportError(
                "mcp_json_rpc_error",
                {"method": request.get("method"), "error": _redacted_json_map(error)},
                _request_methods(requests),
            )
        responses.append(response)
    return tuple(responses)


def _stdio_request_response(
    process: subprocess.Popen[str],
    request: JsonMap,
    request_methods: tuple[str, ...],
    timeout_s: float,
) -> JsonMap:
    if process.stdin is None or process.stdout is None:
        raise MCPTransportError("mcp_stdio_transport_error", {"error": "stdio pipes unavailable"}, request_methods)
    request_id = request.get("id")
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    deadline = time.monotonic() + timeout_s
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_stdio_process(process)
                raise MCPTransportError("mcp_stdio_transport_error", {"error": "stdio response timeout"}, request_methods)
            events = selector.select(remaining)
            if not events:
                continue
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    raise MCPTransportError(
                        "mcp_stdio_transport_error",
                        {"returncode": process.returncode, "stderr": _stdio_stderr_tail(process)},
                        request_methods,
                    )
                continue
            response = _stdio_response_from_line(line, request_methods)
            if response is None or response.get("id") != request_id:
                continue
            error = response.get("error")
            if isinstance(error, dict):
                method = request.get("method", "")
                raise MCPTransportError(
                    "mcp_json_rpc_error",
                    {"method": method, "error": _redacted_json_map(error)},
                    request_methods,
                )
            return response
    finally:
        selector.close()


def _stdio_response_from_line(line: str, request_methods: tuple[str, ...]) -> JsonMap | None:
    stripped = line.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        return as_mapping(json.loads(stripped), "stdio.response")
    except json.JSONDecodeError as exc:
        raise MCPTransportError("mcp_json_response_required", {}, request_methods) from exc
    except SchemaValidationError as exc:
        raise MCPTransportError("mcp_json_response_required", {"error": str(exc)}, request_methods) from exc


def _close_stdio_process(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _terminate_stdio_process(process)


def _terminate_stdio_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=1)


def _stdio_stderr_tail(process: subprocess.Popen[str]) -> str:
    if process.stderr is None:
        return ""
    return _redacted_tail(process.stderr.read())


def _transport_error_with_methods(
    error: MCPTransportError,
    request_methods: tuple[str, ...],
) -> MCPTransportError:
    return MCPTransportError(error.blocker, error.payload, request_methods)


def _stdio_subprocess_env(server: MCPServerConfig) -> dict[str, str]:
    env = {
        key: value
        for key in MCP_SUBPROCESS_ENV_ALLOWLIST
        if (value := os.environ.get(key)) is not None
    }
    if server.env_file:
        env.update(_env_file_values(server.env_file))
    env.update({key: as_str(value, f"env.{key}") for key, value in server.env.items()})
    return env


def _redacted_tail(raw: str) -> str:
    return _redact_text(raw[-TAIL_LIMIT:])


def _redact_text(raw: str) -> str:
    redacted = TEST_SECRET_MARKER_PATTERN.sub(REDACTION_MARKER, raw)
    redacted = SECRET_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTION_MARKER}", redacted)
    return TOKEN_PATTERN.sub(REDACTION_MARKER, redacted)


def _redacted_json_map(payload: JsonMap) -> dict[str, object]:
    redacted = _redacted_json_value(payload)
    if isinstance(redacted, dict):
        return redacted
    return {}


def _redacted_json_value(value: object) -> object:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, Mapping):
        return {
            _redact_text(str(key)): _redacted_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_redacted_json_value(item) for item in value]
    return value


def _has_json_response(raw: str) -> bool:
    return any(line.strip().startswith("{") for line in raw.splitlines())


def _initialize_request() -> JsonMap:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "asa-runtime", "version": "0.1"},
        },
    }


def _tools_list_request() -> JsonMap:
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}


def _tools_call_request(tool_name: str, arguments: JsonMap) -> JsonMap:
    return {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": dict(arguments)},
    }


def _request_methods(requests: tuple[JsonMap, ...]) -> tuple[str, ...]:
    return tuple(as_str(request["method"], "method") for request in requests)


def _listed_tools_from_rpc(listed: JsonMap) -> tuple[MCPToolDescriptor, ...]:
    try:
        return tuple(
            _tool_from_payload(as_mapping(tool, "tools.list.tool"))
            for tool in as_sequence(listed.get("tools", ()), "tools")
        )
    except SchemaValidationError as exc:
        raise MCPTransportError("mcp_tools_list_invalid", {"error": str(exc)}) from exc


def _validate_live_tool_authority(
    server: MCPServerConfig,
    configured_tool: MCPToolDescriptor,
    listed: JsonMap,
    request_methods: tuple[str, ...],
) -> tuple[MCPToolDescriptor, ...]:
    try:
        listed_tools = _listed_tools_from_rpc(listed)
    except MCPTransportError as exc:
        raise _transport_error_with_methods(exc, request_methods) from exc
    live_tool = _tool_by_name(listed_tools, configured_tool.name)
    if live_tool is None:
        raise MCPTransportError(
            "mcp_tool_not_available",
            {"tool_name": configured_tool.name, "available_tools": [tool.name for tool in listed_tools]},
            request_methods,
        )
    live_blocker = _capability_policy_blocker(server, live_tool.capability)
    if live_blocker:
        raise MCPTransportError(live_blocker, {"capability": _capability_payload(live_tool.capability)}, request_methods)
    if _descriptor_drifted(configured_tool, live_tool):
        raise MCPTransportError(
            "mcp_tool_descriptor_drift",
            {
                "configured": _descriptor_fingerprint(configured_tool),
                "live": _descriptor_fingerprint(live_tool),
            },
            request_methods,
        )
    return listed_tools


def _tool_by_name(tools: tuple[MCPToolDescriptor, ...], tool_name: str) -> MCPToolDescriptor | None:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    return None


def _descriptor_drifted(configured_tool: MCPToolDescriptor, live_tool: MCPToolDescriptor) -> bool:
    return (
        configured_tool.parameters != live_tool.parameters
        or configured_tool.capability != live_tool.capability
    )


def _descriptor_fingerprint(tool: MCPToolDescriptor) -> JsonMap:
    return {
        "name": tool.name,
        "parameters": tool.parameters,
        "capability": _capability_payload(tool.capability),
    }


def _capability_from_payload(tool_name: str, payload: JsonMap) -> MCPToolCapability:
    has_metadata = all(
        field in payload
        for field in ("side_effect_class", "read_only", "requires_approval", "allow_without_approval")
    )
    if not has_metadata:
        return _unknown_capability(tool_name)
    return MCPToolCapability(
        tool_name=tool_name,
        side_effect_class=_side_effect_from_text(as_str(payload.get("side_effect_class"), "side_effect_class")),
        read_only=_optional_bool(payload, "read_only", False),
        requires_approval=_optional_bool(payload, "requires_approval", True),
        allow_without_approval=_optional_bool(payload, "allow_without_approval", False),
        declared=True,
    )


def _unknown_capability(tool_name: str) -> MCPToolCapability:
    return MCPToolCapability(
        tool_name=tool_name,
        side_effect_class=MCPSideEffectClass.UNKNOWN,
        read_only=False,
        requires_approval=True,
        allow_without_approval=False,
        declared=False,
    )


def _side_effect_from_text(value: str) -> MCPSideEffectClass:
    match value:
        case MCPSideEffectClass.READ.value:
            return MCPSideEffectClass.READ
        case MCPSideEffectClass.WRITE.value:
            return MCPSideEffectClass.WRITE
        case MCPSideEffectClass.NETWORK.value:
            return MCPSideEffectClass.NETWORK
        case MCPSideEffectClass.FILESYSTEM.value:
            return MCPSideEffectClass.FILESYSTEM
        case MCPSideEffectClass.PROCESS.value:
            return MCPSideEffectClass.PROCESS
        case MCPSideEffectClass.CREDENTIAL.value:
            return MCPSideEffectClass.CREDENTIAL
        case MCPSideEffectClass.UNKNOWN.value:
            return MCPSideEffectClass.UNKNOWN
        case _:
            return MCPSideEffectClass.UNKNOWN


def _capability_policy_blocker(server: MCPServerConfig, capability: MCPToolCapability) -> str:
    if server.allow_write_tools and not capability.declared:
        return "mcp_tool_capability_migration_required"
    match capability.side_effect_class:
        case MCPSideEffectClass.UNKNOWN:
            return "mcp_tool_capability_unknown"
        case MCPSideEffectClass.READ:
            if not capability.read_only:
                return "mcp_tool_capability_conflict"
            if capability.requires_approval or not capability.allow_without_approval:
                return "mcp_tool_approval_required"
            return ""
        case (
            MCPSideEffectClass.WRITE
            | MCPSideEffectClass.NETWORK
            | MCPSideEffectClass.FILESYSTEM
            | MCPSideEffectClass.PROCESS
            | MCPSideEffectClass.CREDENTIAL
        ):
            if capability.read_only:
                return "mcp_tool_capability_conflict"
            return "mcp_tool_approval_required"
        case unreachable:
            assert_never(unreachable)


def _capability_payload(capability: MCPToolCapability) -> JsonMap:
    return {
        "tool_name": capability.tool_name,
        "side_effect_class": capability.side_effect_class.value,
        "read_only": capability.read_only,
        "requires_approval": capability.requires_approval,
        "allow_without_approval": capability.allow_without_approval,
    }


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
