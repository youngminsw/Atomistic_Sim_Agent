(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleViewer = api;
  if (root && root.document) root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document));
})(typeof window !== "undefined" ? window : globalThis, function createApi() {
  function summarizeBundle(manifest, timeline, diagnostics) {
    return {
      runId: text(manifest.run_id),
      featureType: text(manifest.feature_type),
      stateCount: number(timeline.state_count ?? manifest.state_count),
      finalRemovedVolumeNm3: number(timeline.final_removed_volume_nm3 ?? manifest.final_removed_volume_nm3),
      artifactTypes: Array.isArray(manifest.artifact_types) ? manifest.artifact_types.slice() : [],
      clickCount: number(diagnostics.click_count),
    };
  }

  function cellsForStep(timeline, stepIndex) {
    const state = stateForStep(timeline, stepIndex);
    return Array.isArray(state.cells) ? state.cells.slice() : [];
  }

  function normalizeCellsForCanvas(timeline, stepIndex) {
    const cells = cellsForStep(timeline, stepIndex);
    if (cells.length === 0) return [];
    const xs = cells.map((cell) => number(cell.ix));
    const ys = cells.map((cell) => number(cell.iy));
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const maxEnergy = Math.max(1, ...cells.map((cell) => number(cell.cumulative_energy_eV)));
    const maxDepth = Math.max(1e-9, ...cells.map((cell) => number(cell.surface_depth_nm)));
    return cells.map((cell) => ({
      cell,
      x: axisRatio(number(cell.ix), minX, maxX),
      y: axisRatio(number(cell.iy), minY, maxY),
      energyRatio: ratio(number(cell.cumulative_energy_eV), maxEnergy),
      depthRatio: ratio(number(cell.surface_depth_nm), maxDepth),
      key: { ix: number(cell.ix), iy: number(cell.iy), iz: number(cell.iz) },
    }));
  }

  function diagnosticForCell(diagnostics, key) {
    const clicks = Array.isArray(diagnostics.clicks) ? diagnostics.clicks : [];
    return clicks.find((item) => sameKey(item, key)) ?? {
      ix: key.ix,
      iy: key.iy,
      iz: key.iz,
      material_id: "-",
      region: "-",
      depth_history_nm: [],
      energy_history_eV: [],
      removal_law: "-",
      event_ids: [],
    };
  }

  function createRunBundleView(manifest, timeline, diagnostics) {
    return { manifest, timeline, diagnostics, summary: summarizeBundle(manifest, timeline, diagnostics) };
  }

  function mount(documentRef) {
    const model = { bundle: createRunBundleView(...sampleBundle()), stepPosition: 0, selectedKey: null };
    const el = bind(documentRef);
    el.loadSample.addEventListener("click", () => setBundle(model, el, createRunBundleView(...sampleBundle())));
    el.loadFiles.addEventListener("click", () => loadSelectedBundle(el).then((bundle) => setBundle(model, el, bundle)));
    el.timelineRange.addEventListener("input", () => {
      model.stepPosition = Number.parseInt(el.timelineRange.value, 10);
      render(model, el);
    });
    el.profileCanvas.addEventListener("click", (event) => selectFromCanvas(model, el, event));
    el.energyCanvas.addEventListener("click", (event) => selectFromCanvas(model, el, event));
    el.chatForm.addEventListener("submit", (event) => submitChat(model, el, event));
    setBundle(model, el, model.bundle);
  }

  function bind(documentRef) {
    return {
      status: byId(documentRef, "run-status"),
      manifestFile: byId(documentRef, "manifest-file"),
      timelineFile: byId(documentRef, "timeline-file"),
      diagnosticsFile: byId(documentRef, "diagnostics-file"),
      loadFiles: byId(documentRef, "load-files"),
      loadSample: byId(documentRef, "load-sample"),
      factRunId: byId(documentRef, "fact-run-id"),
      factFeature: byId(documentRef, "fact-feature"),
      factStates: byId(documentRef, "fact-states"),
      factVolume: byId(documentRef, "fact-volume"),
      factClicks: byId(documentRef, "fact-clicks"),
      stepTitle: byId(documentRef, "step-title"),
      stepMeta: byId(documentRef, "step-meta"),
      timelineRange: byId(documentRef, "timeline-range"),
      timelineOutput: byId(documentRef, "timeline-output"),
      profileCanvas: byId(documentRef, "profile-canvas"),
      energyCanvas: byId(documentRef, "energy-canvas"),
      diagnosticTitle: byId(documentRef, "diagnostic-title"),
      diagnosticMeta: byId(documentRef, "diagnostic-meta"),
      diagMaterial: byId(documentRef, "diag-material"),
      diagRegion: byId(documentRef, "diag-region"),
      diagLaw: byId(documentRef, "diag-law"),
      diagEvents: byId(documentRef, "diag-events"),
      depthBars: byId(documentRef, "depth-bars"),
      energyBars: byId(documentRef, "energy-bars"),
      chatLog: byId(documentRef, "chat-log"),
      chatForm: byId(documentRef, "chat-form"),
      chatInput: byId(documentRef, "chat-input"),
    };
  }

  function setBundle(model, el, bundle) {
    model.bundle = bundle;
    model.stepPosition = Math.max(0, bundle.timeline.states.length - 1);
    model.selectedKey = null;
    el.timelineRange.max = String(Math.max(0, bundle.timeline.states.length - 1));
    el.timelineRange.value = String(model.stepPosition);
    el.status.textContent = "Bundle ready";
    el.chatLog.replaceChildren();
    pushMessage(el.chatLog, "agent", `Loaded ${bundle.summary.runId}: ${bundle.summary.featureType}, ${bundle.summary.stateCount} states.`);
    render(model, el);
  }

  function render(model, el) {
    const { summary, timeline, diagnostics } = model.bundle;
    const state = timeline.states[model.stepPosition] ?? timeline.states[0];
    const stepIndex = number(state.step_index);
    const key = model.selectedKey ?? firstKey(cellsForStep(timeline, stepIndex));
    const diagnostic = key ? diagnosticForCell(diagnostics, key) : diagnosticForCell(diagnostics, { ix: 0, iy: 0, iz: 0 });
    el.factRunId.textContent = summary.runId;
    el.factFeature.textContent = summary.featureType;
    el.factStates.textContent = String(summary.stateCount);
    el.factVolume.textContent = `${format(summary.finalRemovedVolumeNm3)} nm3`;
    el.factClicks.textContent = String(summary.clickCount);
    el.stepTitle.textContent = `Step ${stepIndex}`;
    el.stepMeta.textContent = `${format(number(state.time_s))} s / ${format(number(state.total_removed_volume_nm3))} nm3`;
    el.timelineOutput.textContent = `${model.stepPosition} / ${Math.max(0, timeline.states.length - 1)}`;
    drawCanvas(el.profileCanvas, normalizeCellsForCanvas(timeline, stepIndex), "depth", key);
    drawCanvas(el.energyCanvas, normalizeCellsForCanvas(timeline, stepIndex), "energy", key);
    renderDiagnostic(el, diagnostic);
  }

  function drawCanvas(canvas, cells, mode, key) {
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#eef2ea";
    ctx.fillRect(0, 0, width, height);
    drawGrid(ctx, width, height);
    cells.forEach((item) => {
      const value = mode === "energy" ? item.energyRatio : item.depthRatio;
      const x = 36 + item.x * (width - 72);
      const y = 36 + item.y * (height - 72);
      const size = 22 + value * 34;
      ctx.fillStyle = mode === "energy" ? heat(value) : depth(value);
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
      if (key && sameKey(item.key, key)) {
        ctx.strokeStyle = "#20251f";
        ctx.lineWidth = 4;
        ctx.strokeRect(x - size / 2 - 4, y - size / 2 - 4, size + 8, size + 8);
      }
    });
  }

  function drawGrid(ctx, width, height) {
    ctx.strokeStyle = "#d7ded1";
    ctx.lineWidth = 1;
    for (let index = 0; index <= 6; index += 1) {
      const x = 36 + (index / 6) * (width - 72);
      const y = 36 + (index / 6) * (height - 72);
      ctx.beginPath();
      ctx.moveTo(x, 36);
      ctx.lineTo(x, height - 36);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(36, y);
      ctx.lineTo(width - 36, y);
      ctx.stroke();
    }
  }

  function selectFromCanvas(model, el, event) {
    const state = model.bundle.timeline.states[model.stepPosition] ?? model.bundle.timeline.states[0];
    const cells = normalizeCellsForCanvas(model.bundle.timeline, number(state.step_index));
    const rect = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - rect.left) / Math.max(1, rect.width);
    const y = (event.clientY - rect.top) / Math.max(1, rect.height);
    const picked = cells.reduce((best, item) => (distance(item, x, y) < distance(best, x, y) ? item : best), cells[0]);
    if (picked) model.selectedKey = picked.key;
    render(model, el);
  }

  function renderDiagnostic(el, diagnostic) {
    el.diagnosticTitle.textContent = `Cell ${diagnostic.ix}, ${diagnostic.iy}, ${diagnostic.iz}`;
    el.diagnosticMeta.textContent = `${last(diagnostic.depth_history_nm)} nm / ${last(diagnostic.energy_history_eV)} eV`;
    el.diagMaterial.textContent = text(diagnostic.material_id);
    el.diagRegion.textContent = text(diagnostic.region);
    el.diagLaw.textContent = text(diagnostic.removal_law);
    el.diagEvents.textContent = Array.isArray(diagnostic.event_ids) ? diagnostic.event_ids.join(", ") : "-";
    drawBars(el.depthBars, diagnostic.depth_history_nm, "depth");
    drawBars(el.energyBars, diagnostic.energy_history_eV, "energy");
  }

  function drawBars(container, values, mode) {
    const numbers = Array.isArray(values) ? values.map(number) : [];
    const max = Math.max(1e-9, ...numbers);
    container.replaceChildren(...numbers.map((value) => {
      const bar = container.ownerDocument.createElement("span");
      bar.className = mode === "energy" ? "bar energy" : "bar";
      bar.style.height = `${Math.max(4, ratio(value, max) * 100)}%`;
      bar.title = format(value);
      return bar;
    }));
  }

  function submitChat(model, el, event) {
    event.preventDefault();
    const prompt = el.chatInput.value.trim();
    if (!prompt) return;
    pushMessage(el.chatLog, "user", prompt);
    pushMessage(el.chatLog, "agent", answer(model.bundle, prompt));
    el.chatInput.value = "";
  }

  function answer(bundle, prompt) {
    const lower = prompt.toLowerCase();
    const lastState = bundle.timeline.states[bundle.timeline.states.length - 1];
    if (lower.includes("energy")) return `Peak transferred energy in the final state is ${format(peak(lastState.cells, "cumulative_energy_eV"))} eV.`;
    if (lower.includes("depth")) return `Final removed volume is ${format(bundle.summary.finalRemovedVolumeNm3)} nm3 across ${bundle.summary.stateCount} states.`;
    if (lower.includes("material")) return `This bundle tracks ${bundle.summary.featureType} etching with PR selectivity stored in the manifest.`;
    return `Run ${bundle.summary.runId} has ${bundle.summary.stateCount} profile states and ${bundle.summary.clickCount} click diagnostics.`;
  }

  function pushMessage(container, role, textValue) {
    const node = container.ownerDocument.createElement("div");
    node.className = role === "user" ? "message user" : "message";
    node.textContent = textValue;
    container.append(node);
    container.scrollTop = container.scrollHeight;
  }

  function loadSelectedBundle(el) {
    return Promise.all([readJson(el.manifestFile), readJson(el.timelineFile), readJson(el.diagnosticsFile)])
      .then(([manifest, timeline, diagnostics]) => createRunBundleView(manifest, timeline, diagnostics))
      .catch((error) => {
        el.status.textContent = error.message;
        return createRunBundleView(...sampleBundle());
      });
  }

  function readJson(input) {
    const file = input.files && input.files[0];
    if (!file) return Promise.reject(new Error(`${input.id} missing`));
    return file.text().then((value) => JSON.parse(value));
  }

  function sampleBundle() {
    const manifest = { run_id: "sample-hole-run", feature_type: "hole", final_removed_volume_nm3: 0.14, state_count: 4, artifact_types: ["run_manifest", "profile_timeline", "click_diagnostics"] };
    const cell = (step, depth, energy) => ({ ix: 16, iy: 16, iz: 0, material_id: "Si", region: "opening", surface_depth_nm: depth, cumulative_energy_eV: energy, removal_law: "target_surrogate_direct", event_ids: step === 0 ? [] : ["evt-0001"] });
    const timeline = { state_count: 4, final_removed_volume_nm3: 0.14, states: [
      { step_index: 0, time_s: 0, total_removed_volume_nm3: 0, cells: [cell(0, 0, 0)] },
      { step_index: 1, time_s: 0.25, total_removed_volume_nm3: 0.04, cells: [cell(1, 0.01, 21.666667)] },
      { step_index: 2, time_s: 0.5, total_removed_volume_nm3: 0.09, cells: [cell(2, 0.02, 43.333333)] },
      { step_index: 3, time_s: 0.75, total_removed_volume_nm3: 0.14, cells: [cell(3, 0.03, 65), { ...cell(3, 0.02, 45), ix: 20, event_ids: ["evt-0002"] }] },
    ] };
    const diagnostics = { click_count: 1, clicks: [{ ix: 16, iy: 16, iz: 0, material_id: "Si", region: "opening", depth_history_nm: [0, 0.01, 0.02, 0.03], energy_history_eV: [0, 21.666667, 43.333333, 65], removal_law: "target_surrogate_direct", event_ids: ["evt-0001"] }] };
    return [manifest, timeline, diagnostics];
  }

  function stateForStep(timeline, stepIndex) {
    const states = Array.isArray(timeline.states) ? timeline.states : [];
    return states.find((state) => number(state.step_index) === stepIndex) ?? states[stepIndex] ?? { cells: [] };
  }

  function firstKey(cells) {
    const first = cells[0];
    return first ? { ix: number(first.ix), iy: number(first.iy), iz: number(first.iz) } : null;
  }

  function sameKey(left, right) {
    return number(left.ix) === number(right.ix) && number(left.iy) === number(right.iy) && number(left.iz) === number(right.iz);
  }

  function byId(documentRef, id) {
    return documentRef.getElementById(id);
  }

  function distance(item, x, y) {
    return Math.hypot(item.x - x, item.y - y);
  }

  function peak(cells, field) {
    return Math.max(0, ...(Array.isArray(cells) ? cells.map((cell) => number(cell[field])) : []));
  }

  function last(values) {
    return format(Array.isArray(values) && values.length > 0 ? values[values.length - 1] : 0);
  }

  function heat(value) {
    return `rgb(${190 + value * 45}, ${79 + value * 58}, ${52 - value * 20})`;
  }

  function depth(value) {
    return `rgb(${39 + value * 20}, ${115 - value * 30}, ${95 + value * 60})`;
  }

  function ratio(value, max) {
    return Math.round((number(value) / Math.max(1e-9, number(max))) * 1000000) / 1000000;
  }

  function axisRatio(value, min, max) {
    if (max === min) return 0.5;
    return ratio(value - min, max - min);
  }

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function text(value) {
    return typeof value === "string" && value.length > 0 ? value : "-";
  }

  function format(value) {
    return number(value).toFixed(6).replace(/0+$/, "").replace(/\.$/, "");
  }

  return { summarizeBundle, cellsForStep, normalizeCellsForCanvas, diagnosticForCell, createRunBundleView, mount };
});
