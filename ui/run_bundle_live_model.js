(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleLiveModel = api;
})(typeof window !== "undefined" ? window : globalThis, function createLiveModel() {
  function displayStateFromRunResponse(response, selectedClickIndex, selectedStepPosition) {
    const bundle = response.bundle || {};
    const manifest = bundle.manifest || {};
    const timeline = bundle.timeline || {};
    const diagnostics = bundle.diagnostics || {};
    const activeLearning = bundle.active_learning_plan || {};
    const qaReport = response.qa_report || bundle.qa_report || {};
    const readiness = response.production_readiness || bundle.production_readiness || {};
    const states = timelineStates(response);
    const state = states[boundedIndex(selectedStepPosition, states.length, Math.max(0, states.length - 1))] || {};
    const clicks = Array.isArray(diagnostics.clicks) ? diagnostics.clicks : [];
    const click = clicks[boundedIndex(selectedClickIndex, clicks.length, 0)] || {};
    const incidents = Array.isArray(click.incident_history) ? click.incident_history : [];
    return {
      runId: text(manifest.run_id),
      featureType: text(manifest.feature_type),
      runStatus: text(response.run_status || manifest.run_status),
      latestStep: number(state.step_index),
      stateCount: number(timeline.state_count || manifest.state_count || states.length),
      removedVolumeNm3: number(timeline.final_removed_volume_nm3 || manifest.final_removed_volume_nm3),
      clickCount: clicks.length,
      clickMaterial: text(click.material_id),
      clickRegion: text(click.region),
      clickUncertainty: Boolean(click.uncertainty_ood),
      energyTransferEV: number(click.energy_transfer_eV),
      removedDepthNm: number(click.removed_depth_nm),
      profileHistory: arrayNumbers(click.profile_history_nm || click.depth_history_nm),
      energyHistory: arrayNumbers(click.energy_history_eV),
      eventIds: Array.isArray(click.event_ids) ? click.event_ids.slice() : [],
      incidentSummary: summarizeIncidents(incidents),
      activeLearningSummary: summarizeActiveLearning(activeLearning),
      qaStatus: text(qaReport.status),
      qaHardBlockers: Array.isArray(qaReport.hard_blockers) ? qaReport.hard_blockers.slice() : [],
      productionReadiness: productionReadiness(readiness),
      agentStatuses: agentStatuses(response),
      agentMessageLog: messageLog(response),
      continuousLogs: continuousLogs(response),
      artifactLinks: artifactLinks(response),
    };
  }

  function canvasOverlayFromRunResponse(response, stepIndex) {
    const states = timelineStates(response);
    const state =
      states.find((item) => number(item.step_index) === number(stepIndex)) ||
      states[states.length - 1] ||
      {};
    const cells = Array.isArray(state.cells) ? state.cells : [];
    const xs = cells.map((cell) => number(cell.ix));
    const ys = cells.map((cell) => number(cell.iy));
    const minX = cells.length > 0 ? Math.min(...xs) : 0;
    const maxX = cells.length > 0 ? Math.max(...xs) : 0;
    const minY = cells.length > 0 ? Math.min(...ys) : 0;
    const maxY = cells.length > 0 ? Math.max(...ys) : 0;
    const maxEnergy = Math.max(1e-9, ...cells.map((cell) => number(cell.cumulative_energy_eV)));
    const maxDepth = Math.max(1e-9, ...cells.map((cell) => number(cell.surface_depth_nm)));
    return {
      stepIndex: number(state.step_index),
      cells: cells.map((cell) => ({
        ix: number(cell.ix),
        iy: number(cell.iy),
        iz: number(cell.iz),
        materialId: text(cell.material_id),
        xRatio: axisRatio(number(cell.ix), minX, maxX),
        yRatio: axisRatio(number(cell.iy), minY, maxY),
        energyRatio: ratio(number(cell.cumulative_energy_eV), maxEnergy),
        depthRatio: ratio(number(cell.surface_depth_nm), maxDepth),
      })),
    };
  }

  function clickIndexFromCanvasPoint(response, stepIndex, xRatio, yRatio) {
    const cells = canvasOverlayFromRunResponse(response, stepIndex).cells;
    const picked = cells.reduce(
      (best, cell) => (canvasDistance(cell, xRatio, yRatio) < canvasDistance(best, xRatio, yRatio) ? cell : best),
      cells[0],
    );
    const clicks = (((response.bundle || {}).diagnostics || {}).clicks);
    if (!picked || !Array.isArray(clicks)) return 0;
    const index = clicks.findIndex((click) => sameCell(click, picked));
    return index >= 0 ? index : 0;
  }

  function timelineStateCount(response) { return timelineStates(response).length; }

  function lastStatePosition(response) { return Math.max(0, timelineStateCount(response) - 1); }

  function stepIndexAtPosition(response, position) {
    const states = timelineStates(response);
    const state = states[boundedIndex(position, states.length, Math.max(0, states.length - 1))] || {};
    return number(state.step_index);
  }

  function timelineStates(response) {
    const timeline = ((response.bundle || {}).timeline) || {};
    return Array.isArray(timeline.states) ? timeline.states : [];
  }

  function boundedIndex(value, length, fallback) {
    const parsed = Number(value);
    const index = Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
    return Math.min(Math.max(0, index), Math.max(0, length - 1));
  }

  function arrayNumbers(value) { return Array.isArray(value) ? value.map(number) : []; }

  function sameCell(left, right) {
    return (
      number(left.ix) === number(right.ix) &&
      number(left.iy) === number(right.iy) &&
      number(left.iz) === number(right.iz)
    );
  }

  function canvasDistance(cell, xRatio, yRatio) { return (cell.xRatio - xRatio) ** 2 + (cell.yRatio - yRatio) ** 2; }

  function axisRatio(value, min, max) { return max === min ? 0.5 : ratio(value - min, max - min); }

  function ratio(value, max) {
    return Math.round((number(value) / Math.max(1e-9, number(max))) * 1000000) / 1000000;
  }

  function summarizeIncidents(incidents) {
    if (incidents.length === 0) return "-";
    const first = incidents[0];
    return `${incidents.length} hits / ${format(first.energy_eV)} eV / ` +
      `${format(first.polar_deg)} deg polar / ${format(first.azimuth_deg)} deg az`;
  }

  function summarizeActiveLearning(plan) {
    const allowed = Boolean(plan.controlled_event_probe_allowed);
    const batchSize = number(plan.batch_size);
    const requests = Array.isArray(plan.requests) ? plan.requests : [];
    const sampleCount = requests.reduce((total, request) => total + number(request.sample_count), 0);
    return allowed ? `probe: ${batchSize} batch / ${sampleCount} samples` : "covered";
  }

  function agentStatuses(response) {
    return Array.isArray(response.agent_statuses) ? response.agent_statuses.slice() : [];
  }

  function messageLog(response) {
    return Array.isArray(response.agent_message_log) ? response.agent_message_log.slice() : [];
  }

  function continuousLogs(response) {
    return Array.isArray(response.continuous_logs) ? response.continuous_logs.slice() : [];
  }

  function artifactLinks(response) {
    const links = response.artifact_links || {};
    return typeof links === "object" && links !== null ? { ...links } : {};
  }

  function productionReadiness(readiness) {
    return {
      productionReady: readiness.production_ready === true,
      hardBlockers: Array.isArray(readiness.hard_blockers) ? readiness.hard_blockers.slice() : [],
      userActions: Array.isArray(readiness.user_actions) ? readiness.user_actions.slice() : [],
      agentActions: Array.isArray(readiness.agent_actions) ? readiness.agent_actions.slice() : [],
      actionPlan: actionPlan(readiness),
    };
  }

  function actionPlan(readiness) {
    return Array.isArray(readiness.action_plan)
      ? readiness.action_plan.map((action) => normalizeAction(action))
      : [];
  }

  function normalizeAction(action) {
    if (typeof action !== "object" || action === null) return { action: text(action), command: [] };
    return {
      ...action,
      command: Array.isArray(action.command) ? action.command.slice() : [],
      expected_artifacts: Array.isArray(action.expected_artifacts) ? action.expected_artifacts.slice() : [],
      missing_artifacts: Array.isArray(action.missing_artifacts) ? action.missing_artifacts.slice() : [],
    };
  }

  function format(value) { return number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, ""); }

  function number(value) { const parsed = Number(value); return Number.isFinite(parsed) ? parsed : 0; }

  function text(value) { return typeof value === "string" && value.length > 0 ? value : "-"; }

  return {
    canvasOverlayFromRunResponse,
    clickIndexFromCanvasPoint,
    displayStateFromRunResponse,
    agentStatuses,
    artifactLinks,
    continuousLogs,
    lastStatePosition,
    messageLog,
    stepIndexAtPosition,
    timelineStateCount,
  };
});
