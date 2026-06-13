from __future__ import annotations

import subprocess
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SOURCE_ROOT
UI_ROOT = SOURCE_ROOT / "ui"


def test_run_bundle_viewer_html_exposes_controller_surface() -> None:
    html = UI_ROOT / "run_bundle_viewer.html"
    css = UI_ROOT / "run_bundle_viewer.css"
    js = UI_ROOT / "run_bundle_viewer.js"

    markup = html.read_text(encoding="utf-8")

    assert css.exists()
    assert js.exists()
    assert 'id="bundle-loader"' in markup
    assert 'id="timeline-range"' in markup
    assert 'id="profile-canvas"' in markup
    assert 'id="energy-canvas"' in markup
    assert 'id="diagnostic-panel"' in markup
    assert 'id="chat-log"' in markup
    assert 'id="production-readiness-summary"' in markup
    assert 'id="production-hard-blockers"' in markup
    assert 'id="production-user-actions"' in markup
    assert 'id="production-action-plan"' in markup
    assert "run_bundle_viewer.css" in markup
    assert "run_bundle_viewer.js" in markup


def test_run_bundle_viewer_js_maps_bundle_to_clickable_state() -> None:
    js = UI_ROOT / "run_bundle_viewer.js"
    script = "\n".join(
        [
            f"const viewer = require({str(js)!r});",
            'const manifest = {"run_id":"ui-fixture","feature_type":"hole","final_removed_volume_nm3":0.14,"state_count":4,"artifact_types":["run_manifest","profile_timeline","click_diagnostics"]};',
            'const timeline = {"state_count":4,"final_removed_volume_nm3":0.14,"states":[{"step_index":0,"time_s":0,"total_removed_volume_nm3":0,"cells":[]},{"step_index":3,"time_s":0.75,"total_removed_volume_nm3":0.14,"cells":[{"ix":16,"iy":16,"iz":0,"material_id":"Si","region":"opening","surface_depth_nm":0.03,"cumulative_energy_eV":65,"removal_law":"target_surrogate_direct","event_ids":["evt-0001"]},{"ix":20,"iy":16,"iz":0,"material_id":"Si","region":"opening","surface_depth_nm":0.02,"cumulative_energy_eV":45,"removal_law":"target_surrogate_direct","event_ids":["evt-0002"]}]}]};',
            'const diagnostics = {"click_count":1,"clicks":[{"ix":16,"iy":16,"iz":0,"material_id":"Si","region":"opening","depth_history_nm":[0,0.01,0.02,0.03],"energy_history_eV":[0,21.666667,43.333333,65],"removal_law":"target_surrogate_direct","event_ids":["evt-0001"]}]};',
            "const summary = viewer.summarizeBundle(manifest, timeline, diagnostics);",
            'if (summary.runId !== "ui-fixture") throw new Error("bad runId");',
            'if (summary.featureType !== "hole") throw new Error("bad featureType");',
            "if (summary.clickCount !== 1) throw new Error('bad clickCount');",
            "const cells = viewer.cellsForStep(timeline, 3);",
            "if (cells.length !== 2) throw new Error('bad cell count');",
            "const normalized = viewer.normalizeCellsForCanvas(timeline, 3);",
            "if (normalized[0].energyRatio !== 1) throw new Error('bad energy ratio');",
            "const diagnostic = viewer.diagnosticForCell(diagnostics, {ix: 16, iy: 16, iz: 0});",
            'if (diagnostic.depth_history_nm.join(",") !== "0,0.01,0.02,0.03") throw new Error("bad diagnostic");',
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_run_bundle_live_model_surfaces_production_readiness() -> None:
    js = UI_ROOT / "run_bundle_live_model.js"
    script = "\n".join(
        [
            f"const model = require({str(js)!r});",
            'const response = {"run_status":"complete","production_readiness":{"production_ready":false,"hard_blockers":["model_endpoint_smoke_required"],"user_actions":["login_to_model_gateway_or_provide_token"],"action_plan":[{"action":"run_model_endpoint_smoke_after_credentials","actor":"agent","status":"ready_after_user_action","command":["python3","smoke.py"]}]},"bundle":{"manifest":{"run_id":"ui-fixture","feature_type":"hole","run_status":"complete"},"timeline":{"states":[]},"diagnostics":{"clicks":[]},"active_learning_plan":{},"qa_report":{"status":"pass","hard_blockers":[]}}};',
            "const state = model.displayStateFromRunResponse(response, 0, 0);",
            "if (state.productionReadiness.productionReady !== false) throw new Error('bad productionReady');",
            "if (state.productionReadiness.hardBlockers[0] !== 'model_endpoint_smoke_required') throw new Error('bad blockers');",
            "if (state.productionReadiness.userActions[0] !== 'login_to_model_gateway_or_provide_token') throw new Error('bad actions');",
            "if (state.productionReadiness.actionPlan[0].action !== 'run_model_endpoint_smoke_after_credentials') throw new Error('bad action plan');",
        ],
    )
    result = subprocess.run(["node", "-e", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
