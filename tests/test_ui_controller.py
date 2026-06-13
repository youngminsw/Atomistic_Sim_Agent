from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_run_bundle_viewer_html_exposes_offline_controller_controls() -> None:
    html = (UI_ROOT / "run_bundle_viewer.html").read_text(encoding="utf-8")

    assert 'id="mode-select"' in html
    assert 'id="feature-select"' in html
    assert 'id="geometry-path"' in html
    assert 'id="kernel-path"' in html
    assert 'id="events-path"' in html
    assert 'id="compute-target"' in html
    assert 'id="iedf-ready"' in html
    assert 'id="iadf-ready"' in html
    assert 'id="validate-run"' in html
    assert 'id="run-offline"' in html
    assert 'id="model-provider"' in html
    assert 'id="model-name"' in html
    assert 'id="reasoning-effort"' in html
    assert 'id="auth-mode"' in html
    assert 'id="model-base-url"' in html
    assert 'id="model-api-key-env"' in html
    assert 'id="agent-status-board"' in html
    assert 'id="agent-message-log"' in html
    assert 'id="run-log"' in html
    assert 'id="artifact-links"' in html
    assert 'id="qa-report-summary"' in html
    assert 'id="qa-hard-blockers"' in html
    assert 'id="production-action-plan"' in html
    assert 'id="diag-incidents"' in html
    assert 'id="fact-active-learning"' in html
    assert 'id="agent-plan-form"' in html
    assert 'id="agent-request-json"' in html
    assert "run_bundle_controller.js" in html
    assert "run_bundle_live_model.js" in html
    assert "run_bundle_api_client.js" in html
    assert "run_bundle_agent_client.js" in html
    assert "run_bundle_controller.css" in html


def test_controller_js_blocks_missing_recipe_and_builds_runner_args() -> None:
    js = UI_ROOT / "run_bundle_controller.js"
    script = "\n".join(
        [
            f"const controller = require({str(js)!r});",
            "const missing = controller.validateControllerInput({ mode: '3d', geometryPath: 'scene.json', kernelPath: 'kernel.json', eventsPath: 'events.jsonl', steps: 5, ions: 8, runId: 'ui-run', iedfReady: false, iadfReady: true });",
            "if (missing.canRun) throw new Error('missing recipe allowed');",
            "if (!missing.missingFields.includes('iedf')) throw new Error('missing iedf not reported');",
            "const valid = controller.validateControllerInput({ mode: '2d', geometryPath: 'mask.png', kernelPath: 'kernel.json', eventsPath: 'events.jsonl', steps: 4, ions: 6, runId: 'ui-run-2d', iedfReady: true, iadfReady: true });",
            "if (!valid.canRun) throw new Error('valid request rejected');",
            "const args = controller.buildOfflineRunnerArgs(valid.normalized);",
            "if (!args.includes('--image')) throw new Error('2d image flag missing');",
            "if (args.includes('--scene')) throw new Error('2d scene flag should not be used');",
            "if (!args.includes('--kernel') || !args.includes('--events')) throw new Error('dependency args missing');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_ui_api_builds_remote_first_offline_command_and_blocks_missing_recipe() -> None:
    from sim_agent.ui import ControllerRunRequest, build_offline_runner_command, validate_controller_request

    missing = validate_controller_request(
        ControllerRunRequest(
            mode="3d",
            geometry_path="tests/fixtures/scenes/pr_hole_scene.json",
            kernel_path="tests/fixtures/kernels/offline_ar_si_kernel.json",
            events_path="tests/fixtures/md_events/md_events_small.jsonl",
            steps=5,
            ions=8,
            run_id="api-hole",
            compute_target="gpu-5090",
            iedf_ready=False,
            iadf_ready=True,
        )
    )
    valid = validate_controller_request(
        ControllerRunRequest(
            mode="3d",
            geometry_path="tests/fixtures/scenes/pr_hole_scene.json",
            kernel_path="tests/fixtures/kernels/offline_ar_si_kernel.json",
            events_path="tests/fixtures/md_events/md_events_small.jsonl",
            steps=5,
            ions=8,
            run_id="api-hole",
            compute_target="gpu-5090",
            iedf_ready=True,
            iadf_ready=True,
        )
    )
    command = build_offline_runner_command(valid.request)

    assert missing.can_run is False
    assert missing.missing_fields == ("iedf",)
    assert valid.can_run is True
    assert valid.compute_target == "gpu-5090"
    assert command[:2] == ("python", "02.Source_code/mss_agent/scripts/run_offline_simulation.py")
    assert "--scene" in command
    assert "--kernel" in command


def test_live_client_maps_http_run_response_to_display_state() -> None:
    js = UI_ROOT / "run_bundle_api_client.js"
    script = "\n".join(
        [
            f"const live = require({str(js)!r});",
            "const response = { can_run: true, run_status: 'complete', qa_report: { status: 'pass', hard_blockers: [] }, production_readiness: { production_ready: false, hard_blockers: ['model_endpoint_smoke_required'], user_actions: ['login_to_model_gateway_or_provide_token'], action_plan: [{ action: 'run_model_endpoint_smoke_after_credentials', actor: 'agent', status: 'ready_after_user_action', requires_user_action: 'login_to_model_gateway_or_provide_token', command: ['python3', 'smoke.py'] }] }, agent_statuses: [{ agent_id: 'qa_agent', label: 'QA Agent', status: 'pass', summary: 'clear', detail: 'checked' }], agent_message_log: [{ sender: 'qa_agent', recipient: 'orchestrator', message: 'clear' }], continuous_logs: ['qa_report_path=/tmp/qa_report.json'], artifact_links: { qa_report: '/tmp/qa_report.json' }, bundle: { manifest: { run_id: 'live-hole', feature_type: 'hole', run_status: 'complete' }, timeline: { state_count: 4, states: [{ step_index: 0, time_s: 0, total_removed_volume_nm3: 0, cells: [] }, { step_index: 3, time_s: 0.3, total_removed_volume_nm3: 0.25, cells: [{ ix: 1, iy: 2, iz: 0, material_id: 'Si', cumulative_energy_eV: 70, surface_depth_nm: 0.05 }] }] }, diagnostics: { click_count: 1, clicks: [{ ix: 1, iy: 2, iz: 0, material_id: 'Si', region: 'opening', energy_transfer_eV: 70, damage_dose: 0.5, removed_depth_nm: 0.05, profile_history_nm: [0, 0.02, 0.05], energy_history_eV: [0, 35, 70], uncertainty_ood: false, incident_history: [{ event_id: 'evt-1', energy_eV: 88.5, polar_deg: 35, azimuth_deg: 180, deposited_energy_eV: 70, removed_depth_nm: 0.05 }] }] }, active_learning_plan: { controlled_event_probe_allowed: true, batch_size: 1, requests: [{ protocol: 'controlled_event_probe', sample_count: 3 }] } } };",
            "const state = live.displayStateFromRunResponse(response);",
            "if (state.runId !== 'live-hole') throw new Error('bad run id');",
            "if (state.featureType !== 'hole') throw new Error('bad feature');",
            "if (state.latestStep !== 3) throw new Error('bad step');",
            "if (state.clickMaterial !== 'Si') throw new Error('bad click material');",
            "if (state.energyTransferEV !== 70) throw new Error('bad energy');",
            "if (!state.incidentSummary.includes('88.5 eV')) throw new Error('bad incident energy summary');",
            "if (!state.incidentSummary.includes('35 deg')) throw new Error('bad incident angle summary');",
            "if (state.activeLearningSummary !== 'probe: 1 batch / 3 samples') throw new Error('bad active learning summary');",
            "if (state.qaStatus !== 'pass') throw new Error('bad qa status');",
            "if (state.agentStatuses[0].agent_id !== 'qa_agent') throw new Error('bad agent status');",
            "if (!state.continuousLogs[0].includes('qa_report_path')) throw new Error('bad run log');",
            "if (state.artifactLinks.qa_report !== '/tmp/qa_report.json') throw new Error('bad artifact link');",
            "if (state.productionReadiness.actionPlan[0].action !== 'run_model_endpoint_smoke_after_credentials') throw new Error('bad action plan');",
            "if (!state.productionReadiness.actionPlan[0].command.includes('smoke.py')) throw new Error('bad action command');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_ops_panel_renders_dot_game_agent_board_with_hover_details() -> None:
    js = UI_ROOT / "run_bundle_ops_panel.js"
    script = "\n".join(
        [
            f"const ops = require({str(js)!r});",
            "class Node {",
            "  constructor(id) { this.id = id; this.textContent = ''; this.children = []; this.className = ''; this.title = ''; }",
            "  append(node) { this.children.push(node); this.textContent += node.textContent; }",
            "  replaceChildren(...children) { this.children = children; this.textContent = children.map((child) => child.textContent).join(''); }",
            "}",
            "const nodes = { 'agent-status-board': new Node('agent-status-board') };",
            "const documentRef = { getElementById(id) { return nodes[id] || null; }, createElement(id) { return new Node(id); } };",
            "ops.renderAgentStatusBoard(documentRef, [",
            "  { agent_id: 'orchestrator', label: 'Orchestrator', status: 'routing', summary: 'calling MD Agent', detail: 'orchestrator -> md_agent' },",
            "  { agent_id: 'md_agent', label: 'MD Agent', status: 'working', summary: 'LAMMPS deck review', detail: 'checking physics gates' },",
            "]);",
            "const board = nodes['agent-status-board'];",
            "if (!board.className.includes('agent-dot-board')) throw new Error('dot board class missing');",
            "if (!board.children[0].className.includes('agent-node')) throw new Error('agent node class missing');",
            "if (!board.children[0].title.includes('orchestrator -> md_agent')) throw new Error('hover detail missing');",
            "if (!board.children[1].textContent.includes('MD Agent')) throw new Error('agent label missing');",
            "if (!board.children[1].textContent.includes('LAMMPS deck review')) throw new Error('agent message missing');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_live_client_maps_timeline_cells_to_canvas_overlay() -> None:
    js = UI_ROOT / "run_bundle_api_client.js"
    script = "\n".join(
        [
            f"const live = require({str(js)!r});",
            "const response = { run_status: 'complete', bundle: { manifest: { run_id: 'overlay-hole', feature_type: 'hole' }, timeline: { states: [{ step_index: 0, cells: [] }, { step_index: 2, cells: [{ ix: 2, iy: 4, iz: 0, material_id: 'Si', cumulative_energy_eV: 20, surface_depth_nm: 0.02 }, { ix: 6, iy: 8, iz: 0, material_id: 'Si', cumulative_energy_eV: 40, surface_depth_nm: 0.04 }] }] }, diagnostics: { click_count: 1, clicks: [] } } };",
            "const overlay = live.canvasOverlayFromRunResponse(response, 2);",
            "if (overlay.stepIndex !== 2) throw new Error('bad step');",
            "if (overlay.cells.length !== 2) throw new Error('bad cell count');",
            "if (overlay.cells[0].xRatio !== 0) throw new Error('bad x ratio first');",
            "if (overlay.cells[1].xRatio !== 1) throw new Error('bad x ratio second');",
            "if (overlay.cells[1].energyRatio !== 1) throw new Error('bad energy ratio');",
            "if (overlay.cells[1].depthRatio !== 1) throw new Error('bad depth ratio');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_live_client_selects_clicked_diagnostic_from_overlay() -> None:
    js = UI_ROOT / "run_bundle_api_client.js"
    script = "\n".join(
        [
            f"const live = require({str(js)!r});",
            "const response = { run_status: 'complete', bundle: { manifest: { run_id: 'multi-click', feature_type: 'hole' }, timeline: { states: [{ step_index: 2, cells: [{ ix: 2, iy: 4, iz: 0, material_id: 'Si', cumulative_energy_eV: 20, surface_depth_nm: 0.02 }, { ix: 6, iy: 8, iz: 0, material_id: 'PR', cumulative_energy_eV: 40, surface_depth_nm: 0.004 }] }] }, diagnostics: { click_count: 2, clicks: [{ ix: 2, iy: 4, iz: 0, material_id: 'Si', region: 'opening', energy_transfer_eV: 20, removed_depth_nm: 0.02, profile_history_nm: [0, 0.02], energy_history_eV: [0, 20], event_ids: ['evt-si'] }, { ix: 6, iy: 8, iz: 0, material_id: 'PR', region: 'mask', energy_transfer_eV: 40, removed_depth_nm: 0.004, profile_history_nm: [0, 0.004], energy_history_eV: [0, 40], event_ids: ['evt-pr'] }] } } };",
            "const state = live.displayStateFromRunResponse(response, 1);",
            "if (state.clickCount !== 2) throw new Error('bad click count');",
            "if (state.clickMaterial !== 'PR') throw new Error('bad selected material');",
            "if (state.eventIds[0] !== 'evt-pr') throw new Error('bad selected event');",
            "if (live.clickIndexFromCanvasPoint(response, 2, 1, 1) !== 1) throw new Error('bad picked click');",
            "if (live.clickIndexFromCanvasPoint(response, 2, 0, 0) !== 0) throw new Error('bad first click');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_live_client_timeline_slider_replays_api_bundle_steps() -> None:
    js = UI_ROOT / "run_bundle_api_client.js"
    script = "\n".join(
        [
            f"const live = require({str(js)!r});",
            "class Node { constructor(id) { this.id = id; this.value = ''; this.checked = true; this.textContent = ''; this.max = ''; this.listeners = {}; this.style = {}; this.width = 120; this.height = 80; this.children = []; this.className = ''; this.title = ''; } addEventListener(name, fn) { this.listeners[name] = fn; } append(node) { this.children.push(node); this.textContent += node.textContent; } trigger(name) { return this.listeners[name]({ currentTarget: this, clientX: 0, clientY: 0 }); } getBoundingClientRect() { return { left: 0, top: 0, width: this.width, height: this.height }; } getContext() { return { clearRect() {}, fillRect() {}, set fillStyle(value) { this.color = value; } }; } replaceChildren(...children) { this.children = children; this.textContent = children.map((child) => child.textContent).join(''); } }",
            "const ids = ['run-offline', 'controller-errors', 'run-status', 'fact-run-id', 'fact-feature', 'fact-states', 'fact-volume', 'fact-clicks', 'fact-active-learning', 'diagnostic-title', 'diagnostic-meta', 'diag-material', 'diag-region', 'diag-law', 'diag-events', 'diag-incidents', 'depth-bars', 'energy-bars', 'profile-canvas', 'energy-canvas', 'timeline-range', 'timeline-output', 'mode-select', 'feature-select', 'geometry-path', 'kernel-path', 'events-path', 'compute-target', 'run-steps', 'run-ions', 'iedf-ready', 'iadf-ready', 'agent-status-board', 'agent-message-log', 'run-log', 'artifact-links', 'qa-report-summary', 'qa-hard-blockers', 'production-readiness-summary', 'production-hard-blockers', 'production-user-actions', 'production-action-plan'];",
            "const nodes = Object.fromEntries(ids.map((id) => [id, new Node(id)]));",
            "nodes['mode-select'].value = '3d'; nodes['feature-select'].value = 'hole'; nodes['run-steps'].value = '3'; nodes['run-ions'].value = '5';",
            "const documentRef = { getElementById(id) { return nodes[id] || null; }, createElement(id) { return new Node(id); } };",
            "const response = { run_status: 'complete', qa_report: { status: 'pass', hard_blockers: [] }, production_readiness: { production_ready: false, hard_blockers: ['model_endpoint_smoke_required'], user_actions: ['login_to_model_gateway_or_provide_token'], action_plan: [{ action: 'run_model_endpoint_smoke_after_credentials', actor: 'agent', status: 'ready_after_user_action', requires_user_action: 'login_to_model_gateway_or_provide_token', command: ['python3', 'smoke_production_gateway_client.py'] }] }, agent_statuses: [{ agent_id: 'qa_agent', label: 'QA Agent', status: 'pass', summary: 'clear', detail: 'checked' }], agent_message_log: [{ sender: 'qa_agent', recipient: 'orchestrator', message: 'clear' }], continuous_logs: ['qa_report_path=/tmp/qa_report.json'], artifact_links: { qa_report: '/tmp/qa_report.json' }, bundle: { manifest: { run_id: 'slider-hole', feature_type: 'hole' }, timeline: { states: [{ step_index: 0, time_s: 0, cells: [{ ix: 1, iy: 1, iz: 0, material_id: 'Si', cumulative_energy_eV: 0, surface_depth_nm: 0 }] }, { step_index: 1, time_s: 0.1, cells: [{ ix: 1, iy: 1, iz: 0, material_id: 'Si', cumulative_energy_eV: 10, surface_depth_nm: 0.01 }] }, { step_index: 2, time_s: 0.2, cells: [{ ix: 1, iy: 1, iz: 0, material_id: 'Si', cumulative_energy_eV: 20, surface_depth_nm: 0.02 }] }] }, diagnostics: { click_count: 1, clicks: [{ ix: 1, iy: 1, iz: 0, material_id: 'Si', region: 'opening', energy_transfer_eV: 20, removed_depth_nm: 0.02, profile_history_nm: [0, 0.01, 0.02], energy_history_eV: [0, 10, 20], event_ids: ['evt-1'] }] } } };",
            "const fetcher = () => Promise.resolve({ ok: true, json: () => Promise.resolve(response) });",
            "async function main() { live.mount(documentRef, fetcher); await nodes['run-offline'].trigger('click'); if (nodes['timeline-range'].max !== '2') throw new Error('bad slider max'); if (nodes['timeline-range'].value !== '2') throw new Error('bad slider value'); if (!nodes['agent-status-board'].textContent.includes('QA Agent')) throw new Error('agent board not rendered'); if (!nodes['artifact-links'].textContent.includes('qa_report')) throw new Error('artifact links not rendered'); if (!nodes['qa-hard-blockers'].textContent.includes('none')) throw new Error('qa blockers not rendered'); if (!nodes['production-action-plan'].textContent.includes('run_model_endpoint_smoke_after_credentials')) throw new Error('action plan not rendered'); if (!nodes['production-action-plan'].children[0].title.includes('smoke_production_gateway_client.py')) throw new Error('action plan command not in hover title'); nodes['timeline-range'].value = '1'; nodes['timeline-range'].trigger('input'); if (nodes['run-status'].textContent !== 'complete / step 1') throw new Error('bad replay status'); if (nodes['timeline-output'].textContent !== '1 / 2') throw new Error('bad replay output'); }",
            "main().catch((error) => { console.error(error); process.exit(1); });",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_agent_client_posts_request_json_and_renders_clarification() -> None:
    js = UI_ROOT / "run_bundle_agent_client.js"
    script = "\n".join(
        [
            f"const agent = require({str(js)!r});",
            "class Node {",
            "  constructor(id) {",
            "    this.id = id;",
            "    this.value = '';",
            "    this.textContent = '';",
            "    this.listeners = {};",
            "    this.children = [];",
            "  }",
            "  addEventListener(name, fn) { this.listeners[name] = fn; }",
            "  append(node) {",
            "    this.children.push(node);",
            "    this.textContent += node.textContent;",
            "  }",
            "  trigger(name) {",
            "    return this.listeners[name]({ preventDefault() {}, currentTarget: this });",
            "  }",
            "}",
            "const ids = ['agent-plan-form', 'agent-request-json', 'chat-log', 'model-provider', 'model-name', 'reasoning-effort', 'auth-mode', 'model-base-url', 'model-api-key-env'];",
            "const nodes = Object.fromEntries(ids.map((id) => [id, new Node(id)]));",
            "nodes['model-provider'].value = 'openai';",
            "nodes['model-name'].value = 'gpt-5.5';",
            "nodes['reasoning-effort'].value = 'high';",
            "nodes['auth-mode'].value = 'api_key';",
            "nodes['model-base-url'].value = 'https://api.openai.com/v1';",
            "nodes['model-api-key-env'].value = 'OPENAI_API_KEY';",
            "nodes['agent-request-json'].value = JSON.stringify({",
            "  request_id: 'ui-plan',",
            "  llm_endpoint: {",
            "    provider: 'openclaw',",
            "    model: 'gpt-5.5',",
            "    reasoning_effort: 'high',",
            "    base_url: 'https://openclaw.local/v1',",
            "  },",
            "});",
            "const documentRef = {",
            "  getElementById(id) { return nodes[id] || null; },",
            "  createElement(id) { return new Node(id); },",
            "};",
            "const body = {",
            "  status: 'clarification_required',",
            "  missing_fields: ['geometry', 'material'],",
            "  team_session_contract: {",
            "    heartbeat_interval_s: 3600,",
            "    call_matrix: {",
            "      orchestrator: ['md_agent', 'ml_mdn_agent', 'feature_scale_agent', 'research_graphdb_agent', 'qa_agent'],",
            "      md_agent: ['orchestrator', 'research_graphdb_agent', 'qa_agent'],",
            "    },",
            "    qa_gates: { slurm_job_script: 'qa_before_submit' },",
            "  },",
            "  final_output: 'model_training_required=true ' +",
            "    'training_reason=no_trained_expert_for_Ar_on_Test',",
            "};",
            "const fetcher = (url, options) => {",
            "  if (url !== '/api/agent/plan') throw new Error('bad url');",
            "  if (!options.body.includes('ui-plan')) throw new Error('bad body');",
            "  if (!options.body.includes('https://api.openai.com/v1')) throw new Error('model settings not merged');",
            "  if (!options.body.includes('OPENAI_API_KEY')) throw new Error('api key env not merged');",
            "  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) });",
            "};",
            "async function main() {",
            "  agent.mount(documentRef, fetcher);",
            "  await nodes['agent-plan-form'].trigger('submit');",
            "  const log = nodes['chat-log'].textContent;",
            "  if (!log.includes('Missing: geometry, material')) {",
            "    throw new Error('missing fields not rendered');",
            "  }",
            "  if (!log.includes('no_trained_expert_for_Ar_on_Test')) {",
            "    throw new Error('training reason not rendered');",
            "  }",
            "  if (!log.includes('Team: 6 agents / heartbeat 3600s / QA Slurm gate qa_before_submit')) {",
            "    throw new Error('team session contract not rendered');",
            "  }",
            "}",
            "main().catch((error) => { console.error(error); process.exit(1); });",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
