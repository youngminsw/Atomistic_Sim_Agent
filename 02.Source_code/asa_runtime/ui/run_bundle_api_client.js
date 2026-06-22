(function attach(root, factory) {
  const model = root && root.RunBundleLiveModel
    ? root.RunBundleLiveModel
    : require("./run_bundle_live_model.js");
  const opsPanel = root && root.RunBundleOpsPanel
    ? root.RunBundleOpsPanel
    : require("./run_bundle_ops_panel.js");
  const api = factory(model, opsPanel);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleApiClient = api;
  if (root && root.document) {
    root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document, root.fetch));
  }
})(typeof window !== "undefined" ? window : globalThis, function createLiveApiClient(model, opsPanel) {
  function buildPayloadFromDocument(documentRef) {
    const mode = value(documentRef, "mode-select") === "2d" ? "2d" : "3d";
    const feature = text(value(documentRef, "feature-select")) || (mode === "2d" ? "trench" : "hole");
    const runId = `ui-${feature}-${mode}`;
    return {
      mode,
      geometry_path: value(documentRef, "geometry-path"),
      kernel_path: value(documentRef, "kernel-path"),
      events_path: value(documentRef, "events-path"),
      compute_target: value(documentRef, "compute-target"),
      steps: positiveInteger(value(documentRef, "run-steps"), 5),
      ions: positiveInteger(value(documentRef, "run-ions"), 8),
      run_id: runId,
      iedf_ready: checked(documentRef, "iedf-ready"),
      iadf_ready: checked(documentRef, "iadf-ready"),
      output_dir: `evidence/${runId}`,
    };
  }

  function mount(documentRef, fetcher) {
    const runState = { response: null, clickIndex: 0, stepPosition: 0 };
    const button = documentRef.getElementById("run-offline");
    const range = documentRef.getElementById("timeline-range");
    if (button && typeof fetcher === "function") {
      button.addEventListener("click", () => runFromForm(documentRef, fetcher, runState));
    }
    if (range) range.addEventListener("input", () => replayStep(documentRef, runState, range.value));
    ["profile-canvas", "energy-canvas"].forEach((id) => {
      const canvas = documentRef.getElementById(id);
      if (canvas) canvas.addEventListener("click", (event) => selectFromCanvas(documentRef, runState, event));
    });
  }

  function runFromForm(documentRef, fetcher, runState) {
    const statusNode = documentRef.getElementById("controller-errors");
    statusNode.textContent = "Running offline fixture...";
    return fetcher("/api/run/offline", {
      method: "POST",
      headers: csrfHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(buildPayloadFromDocument(documentRef)),
    })
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then((result) => {
        if (!result.ok) {
          renderBlocked(statusNode, result.body);
          return result.body;
        }
        setRunState(runState, result.body, 0, model.lastStatePosition(result.body));
        renderLiveState(documentRef, runState);
        return result.body;
      });
  }

  function replayStep(documentRef, runState, rawPosition) {
    if (!runState.response) return;
    const parsed = Number.parseInt(rawPosition, 10);
    runState.stepPosition = Number.isFinite(parsed) ? parsed : 0;
    renderLiveState(documentRef, runState);
  }

  function renderLiveState(documentRef, runState) {
    const response = runState.response;
    const stepIndex = model.stepIndexAtPosition(response, runState.stepPosition);
    renderTimeline(documentRef, response, runState.stepPosition);
    renderRun(
      documentRef,
      model.displayStateFromRunResponse(response, runState.clickIndex, runState.stepPosition),
    );
    renderCanvases(documentRef, response, stepIndex);
  }

  function setRunState(runState, response, clickIndex, stepPosition) {
    if (!runState) return;
    runState.response = response;
    runState.clickIndex = clickIndex;
    runState.stepPosition = stepPosition;
  }

  function renderRun(documentRef, state) {
    setText(documentRef, "controller-errors", `Complete: ${state.runId}`);
    setText(documentRef, "run-status", `${state.runStatus} / step ${state.latestStep}`);
    setText(documentRef, "fact-run-id", state.runId);
    setText(documentRef, "fact-feature", state.featureType);
    setText(documentRef, "fact-states", String(state.stateCount));
    setText(documentRef, "fact-volume", `${format(state.removedVolumeNm3)} nm3`);
    setText(documentRef, "fact-clicks", String(state.clickCount));
    setText(documentRef, "fact-active-learning", state.activeLearningSummary);
    setText(documentRef, "diagnostic-title", `Cell diagnostic / step ${state.latestStep}`);
    setText(
      documentRef,
      "diagnostic-meta",
      `${format(state.removedDepthNm)} nm / ${format(state.energyTransferEV)} eV`,
    );
    setText(documentRef, "diag-material", state.clickMaterial);
    setText(documentRef, "diag-region", state.clickRegion);
    setText(documentRef, "diag-law", state.clickUncertainty ? "uncertainty_ood" : "surrogate_transport_level_set");
    setText(documentRef, "diag-events", state.eventIds.join(", ") || "-");
    setText(documentRef, "diag-incidents", state.incidentSummary);
    opsPanel.renderAgentStatusBoard(documentRef, state.agentStatuses);
    opsPanel.renderMessageLog(documentRef, state.agentMessageLog);
    opsPanel.renderRunLog(documentRef, state.continuousLogs);
    opsPanel.renderArtifactLinks(documentRef, state.artifactLinks);
    opsPanel.renderQaReport(documentRef, state.qaStatus, state.qaHardBlockers);
    opsPanel.renderProductionReadiness(documentRef, state.productionReadiness);
    renderBars(documentRef, "depth-bars", state.profileHistory, "bar");
    renderBars(documentRef, "energy-bars", state.energyHistory, "bar energy");
  }

  function renderTimeline(documentRef, response, position) {
    const range = documentRef.getElementById("timeline-range");
    const maxPosition = model.lastStatePosition(response);
    if (range) {
      range.max = String(maxPosition);
      range.value = String(Math.min(Math.max(0, position), maxPosition));
    }
    setText(documentRef, "timeline-output", `${Math.min(Math.max(0, position), maxPosition)} / ${maxPosition}`);
  }

  function renderCanvases(documentRef, response, stepIndex) {
    const overlay = model.canvasOverlayFromRunResponse(response, stepIndex);
    drawCanvas(documentRef.getElementById("profile-canvas"), overlay.cells, "depthRatio");
    drawCanvas(documentRef.getElementById("energy-canvas"), overlay.cells, "energyRatio");
  }

  function selectFromCanvas(documentRef, runState, event) {
    if (!runState.response) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / Math.max(1, rect.width);
    const y = (event.clientY - rect.top) / Math.max(1, rect.height);
    const stepIndex = model.stepIndexAtPosition(runState.response, runState.stepPosition);
    runState.clickIndex = model.clickIndexFromCanvasPoint(runState.response, stepIndex, x, y);
    renderLiveState(documentRef, runState);
  }

  function drawCanvas(canvas, cells, valueField) {
    if (!canvas || typeof canvas.getContext !== "function") return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#eef2ea";
    ctx.fillRect(0, 0, width, height);
    cells.forEach((cell) => {
      const value = cell[valueField];
      const x = 34 + cell.xRatio * (width - 68);
      const y = 34 + cell.yRatio * (height - 68);
      const size = 18 + value * 38;
      ctx.fillStyle = valueField === "energyRatio" ? heat(value) : depth(value);
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
    });
  }

  function renderBlocked(node, body) {
    const missing = Array.isArray(body.missing_fields) ? body.missing_fields.join(", ") : text(body.error);
    node.textContent = `Blocked: ${missing}`;
  }

  function renderBars(documentRef, id, values, className) {
    const node = documentRef.getElementById(id);
    if (!node) return;
    const max = Math.max(1e-9, ...values);
    node.replaceChildren(...values.map((item) => {
      const bar = documentRef.createElement("span");
      bar.className = className;
      bar.style.height = `${Math.max(4, (item / max) * 100)}%`;
      bar.title = format(item);
      return bar;
    }));
  }

  function setText(documentRef, id, content) {
    const node = documentRef.getElementById(id);
    if (node) node.textContent = content;
  }

  function value(documentRef, id) { const node = documentRef.getElementById(id); return node ? node.value : ""; }

  function csrfHeaders(headers) {
    const token = globalThis.__ASA_CONTROLLER_TOKEN__;
    return token ? { ...headers, "X-ASA-CSRF-Token": token } : headers;
  }

  function checked(documentRef, id) {
    const node = documentRef.getElementById(id);
    return Boolean(node && node.checked);
  }

  function positiveInteger(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : 0; }

  function text(value) { return typeof value === "string" && value.length > 0 ? value : "-"; }

  function format(value) { return number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, ""); }

  function heat(value) { return `rgb(${190 + value * 45}, ${79 + value * 58}, ${52 - value * 20})`; }

  function depth(value) { return `rgb(${39 + value * 20}, ${115 - value * 30}, ${95 + value * 60})`; }

  return {
    buildPayloadFromDocument,
    canvasOverlayFromRunResponse: model.canvasOverlayFromRunResponse,
    clickIndexFromCanvasPoint: model.clickIndexFromCanvasPoint,
    displayStateFromRunResponse: model.displayStateFromRunResponse,
    mount,
    runFromForm,
  };
});
