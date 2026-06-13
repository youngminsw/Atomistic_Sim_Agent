from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from json import JSONDecodeError
from pathlib import Path
from typing import assert_never

from sim_agent.schemas._parse import JsonMap, as_bool, as_mapping, as_str, require
from sim_agent.runner import OfflineRunRequest, OfflineRunResult, RunManagerError, run_offline_simulation

from .agent_compute import build_agent_compute_bundle_http_response
from .agent_plan import build_agent_plan_http_response
from .api import build_ui_agent_graph_context, build_ui_api_status, validate_ui_api_request
from .controller import ControllerRunRequest, UiMode
from .model_auth import ModelAuthError, login_model_gateway, model_auth_status_payload, run_model_gateway_smoke_from_controller
from .response_payload import (
    click_diagnostic_contract_payload,
    run_response_payload,
    status_payload,
    validation_payload,
)


MAX_JSON_BODY_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True)
class UiHttpPayloadError(ValueError):
    code: str
    status_code: int = 400

    def __str__(self) -> str:
        return self.code


class UiRequestHandler(SimpleHTTPRequestHandler):
    server_version = "AtomisticSimUi/0.1"

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]
        match route:
            case "/api/status":
                self._write_json(status_payload(build_ui_api_status()), 200)
            case "/api/knowledge/agent-context":
                self._write_json(build_ui_agent_graph_context(), 200)
            case "/api/model/auth/status":
                self._write_json(model_auth_status_payload(), 200)
            case "/api/click-diagnostics":
                self._write_json(click_diagnostic_contract_payload(), 200)
            case "/":
                self.path = "/run_bundle_viewer.html"
                super().do_GET()
            case _:
                super().do_GET()

    def do_POST(self) -> None:
        route = self.path.split("?", 1)[0]
        if route not in {
            "/api/agent/plan",
            "/api/agent/prepare-md-campaign-worker-bundle",
            "/api/model/auth/login",
            "/api/model/gateway/smoke",
            "/api/run/offline",
        }:
            self._write_json({"error": "route_not_found"}, 404)
            return
        try:
            payload = _read_payload(self)
        except JSONDecodeError:
            self._write_json({"error": "invalid_json"}, 400)
            return
        except UiHttpPayloadError as exc:
            self._write_json({"error": str(exc)}, exc.status_code)
            return
        match route:
            case "/api/agent/plan":
                self._write_json(*build_agent_plan_http_response(payload))
            case "/api/agent/prepare-md-campaign-worker-bundle":
                self._write_json(*build_agent_compute_bundle_http_response(payload))
            case "/api/model/auth/login":
                self._handle_model_auth_login(payload)
            case "/api/model/gateway/smoke":
                self._handle_model_gateway_smoke(payload)
            case "/api/run/offline":
                self._handle_offline_run(payload)

    def _handle_model_auth_login(self, payload: JsonMap) -> None:
        try:
            self._write_json(login_model_gateway(payload), 200)
        except ModelAuthError as exc:
            self._write_json({"error": str(exc)}, 400)

    def _handle_model_gateway_smoke(self, payload: JsonMap) -> None:
        try:
            result = run_model_gateway_smoke_from_controller(payload)
        except (ModelAuthError, OSError, ValueError) as exc:
            self._write_json({"error": str(exc)}, 400)
            return
        status_code = 200 if result.get("ok") is True else 400
        self._write_json(result, status_code)

    def _handle_offline_run(self, payload: JsonMap) -> None:
        try:
            request = _request_from_payload(payload)
        except UiHttpPayloadError as exc:
            self._write_json({"error": str(exc)}, exc.status_code)
            return
        validation = validate_ui_api_request(request)
        if not validation.can_run:
            self._write_json(validation_payload(validation), 400)
            return
        try:
            result = _run_offline_request(validation.request)
        except (UiHttpPayloadError, RunManagerError, OSError) as exc:
            status_code = exc.status_code if isinstance(exc, UiHttpPayloadError) else 400
            self._write_json({"error": str(exc)}, status_code)
            return
        status_code = 200 if result.run_status == "complete" else 500
        self._write_json(run_response_payload(validation, result), status_code)

    def _write_json(self, payload: JsonMap, status_code: int) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return


def build_ui_http_server(host: str, port: int, static_root: Path) -> ThreadingHTTPServer:
    handler = partial(UiRequestHandler, directory=str(static_root))
    return ThreadingHTTPServer((host, port), handler)


def _read_payload(handler: SimpleHTTPRequestHandler) -> JsonMap:
    length = _content_length(handler)
    raw = handler.rfile.read(length).decode("utf-8")
    return as_mapping(json.loads(raw), "request")


def _content_length(handler: SimpleHTTPRequestHandler) -> int:
    raw_length = handler.headers.get("Content-Length", "")
    try:
        length = int(raw_length)
    except ValueError as exc:
        raise UiHttpPayloadError("invalid_content_length") from exc
    if length <= 0:
        raise UiHttpPayloadError("empty_request_body")
    if length > MAX_JSON_BODY_BYTES:
        raise UiHttpPayloadError("request_body_too_large", 413)
    return length


def _request_from_payload(payload: JsonMap) -> ControllerRunRequest:
    return ControllerRunRequest(
        mode=_mode(payload),
        geometry_path=as_str(require(payload, "geometry_path"), "geometry_path"),
        kernel_path=as_str(require(payload, "kernel_path"), "kernel_path"),
        events_path=as_str(require(payload, "events_path"), "events_path"),
        steps=_int_field(payload, "steps"),
        ions=_int_field(payload, "ions"),
        run_id=as_str(require(payload, "run_id"), "run_id"),
        compute_target=as_str(require(payload, "compute_target"), "compute_target"),
        iedf_ready=as_bool(require(payload, "iedf_ready"), "iedf_ready"),
        iadf_ready=as_bool(require(payload, "iadf_ready"), "iadf_ready"),
        output_dir=_optional_str(payload, "output_dir"),
    )


def _mode(payload: JsonMap) -> UiMode:
    mode = as_str(require(payload, "mode"), "mode")
    match mode:
        case "2d":
            return "2d"
        case "3d":
            return "3d"
        case _:
            raise UiHttpPayloadError("invalid_mode")


def _int_field(payload: JsonMap, field: str) -> int:
    value = require(payload, field)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise UiHttpPayloadError(f"{field}_must_be_integer")


def _optional_str(payload: JsonMap, field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    return as_str(value, field)


def _run_offline_request(request: ControllerRunRequest) -> OfflineRunResult:
    source_root = Path(__file__).resolve().parents[2]
    output_dir = _resolve_output_dir(source_root, request.output_dir, request.run_id)
    scene_path, image_path = _geometry_paths(request, source_root)
    return run_offline_simulation(
        OfflineRunRequest(
            run_id=request.run_id,
            mode=request.mode,
            source_root=source_root,
            output_dir=output_dir,
            scene_path=scene_path,
            image_path=image_path,
            kernel_path=_resolve_input_path(source_root, request.kernel_path),
            events_path=_resolve_input_path(source_root, request.events_path),
            time_steps=request.steps,
            ion_count=request.ions,
            seed=7,
        )
    )


def _geometry_paths(request: ControllerRunRequest, source_root: Path) -> tuple[Path | None, Path | None]:
    path = _resolve_input_path(source_root, request.geometry_path)
    match request.mode:
        case "3d":
            return path, None
        case "2d":
            return None, path
        case unreachable:
            assert_never(unreachable)


def _resolve_input_path(source_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise UiHttpPayloadError("absolute_input_paths_not_allowed")
    if _contains_parent_reference(path):
        raise UiHttpPayloadError("parent_input_paths_not_allowed")
    candidate = (source_root / path).resolve()
    if not _is_relative_to(candidate, source_root.resolve()):
        raise UiHttpPayloadError("input_path_outside_source_root")
    if not candidate.exists():
        raise UiHttpPayloadError("input_path_not_found")
    return candidate


def _resolve_output_dir(source_root: Path, raw_path: str | None, run_id: str) -> Path:
    evidence_root = (source_root / "evidence").resolve()
    if raw_path is None or not raw_path.strip():
        return evidence_root / run_id
    path = Path(raw_path)
    if path.is_absolute():
        raise UiHttpPayloadError("absolute_output_dir_not_allowed")
    if _contains_parent_reference(path):
        raise UiHttpPayloadError("parent_output_dirs_not_allowed")
    if path.parts and path.parts[0] == "evidence":
        candidate = (source_root / path).resolve()
    else:
        candidate = (evidence_root / path).resolve()
    if not _is_relative_to(candidate, evidence_root):
        raise UiHttpPayloadError("output_dir_outside_evidence_root")
    return candidate


def _contains_parent_reference(path: Path) -> bool:
    return any(part == ".." for part in path.parts)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
