(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleController = api;
  if (root && root.document) root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document));
})(typeof window !== "undefined" ? window : globalThis, function createControllerApi() {
  const computeTargets = [];

  function configureComputeTargets(targets) {
    computeTargets.splice(0, computeTargets.length, ...cleanTargets(targets));
    return computeTargets;
  }

  function configureFromRuntimeConfig(config) {
    const resources = config && Array.isArray(config.compute_resources) ? config.compute_resources : [];
    configureComputeTargets(resources.map((resource) => resource && resource.host_alias));
    if (config && config.model_endpoint && typeof config.model_endpoint === "object") {
      globalThis.__ASA_MODEL_ENDPOINT__ = config.model_endpoint;
    }
    return computeTargets;
  }

  function validateControllerInput(raw) {
    const normalized = normalize(raw);
    const missingFields = missing(normalized);
    return {
      canRun: missingFields.length === 0,
      missingFields,
      normalized,
    };
  }

  function buildOfflineRunnerArgs(input) {
    const normalized = normalize(input);
    const sourceFlag = normalized.mode === "2d" ? "--image" : "--scene";
    return [
      sourceFlag,
      normalized.geometryPath,
      "--kernel",
      normalized.kernelPath,
      "--events",
      normalized.eventsPath,
      "--steps",
      String(normalized.steps),
      "--ions",
      String(normalized.ions),
      "--out",
      `02.Source_code/mss_agent/evidence/${normalized.runId}`,
      "--run-id",
      normalized.runId,
    ];
  }

  function mount(documentRef) {
    const validateButton = documentRef.getElementById("validate-run");
    if (!validateButton) return;
    validateButton.addEventListener("click", () => {
      const validation = validateControllerInput(readForm(documentRef));
      renderValidation(documentRef, validation);
    });
  }

  function readForm(documentRef) {
    return {
      mode: value(documentRef, "mode-select"),
      featureType: value(documentRef, "feature-select"),
      geometryPath: value(documentRef, "geometry-path"),
      kernelPath: value(documentRef, "kernel-path"),
      eventsPath: value(documentRef, "events-path"),
      computeTarget: value(documentRef, "compute-target"),
      steps: value(documentRef, "run-steps"),
      ions: value(documentRef, "run-ions"),
      runId: `ui-${value(documentRef, "feature-select")}-${value(documentRef, "mode-select")}`,
      iedfReady: checked(documentRef, "iedf-ready"),
      iadfReady: checked(documentRef, "iadf-ready"),
    };
  }

  function renderValidation(documentRef, validation) {
    const node = documentRef.getElementById("controller-errors");
    if (!node) return;
    if (validation.canRun) {
      node.className = "controller-errors";
      node.textContent = `Ready: ${buildOfflineRunnerArgs(validation.normalized).join(" ")}`;
      return;
    }
    node.className = "controller-errors blocked";
    node.textContent = `Missing: ${validation.missingFields.join(", ")}`;
  }

  function normalize(raw) {
    const mode = raw.mode === "2d" ? "2d" : "3d";
    const steps = positiveInteger(raw.steps, 5);
    const ions = positiveInteger(raw.ions, 8);
    const runId = slug(text(raw.runId) || `ui-${mode}-run`);
    const rawComputeTarget = text(raw.computeTarget);
    const computeTarget = computeTargets.includes(rawComputeTarget) ? rawComputeTarget : "";
    return {
      mode,
      featureType: text(raw.featureType) || (mode === "2d" ? "trench" : "hole"),
      geometryPath: text(raw.geometryPath),
      kernelPath: text(raw.kernelPath),
      eventsPath: text(raw.eventsPath),
      computeTarget,
      steps,
      ions,
      runId,
      iedfReady: Boolean(raw.iedfReady),
      iadfReady: Boolean(raw.iadfReady),
    };
  }

  function missing(input) {
    const fields = [];
    if (!input.geometryPath) fields.push(input.mode === "2d" ? "image" : "scene");
    if (!input.kernelPath) fields.push("kernel");
    if (!input.eventsPath) fields.push("events");
    if (!input.computeTarget) fields.push("compute");
    if (!input.iedfReady) fields.push("iedf");
    if (!input.iadfReady) fields.push("iadf");
    if (input.steps <= 0) fields.push("steps");
    if (input.ions <= 0) fields.push("ions");
    return fields;
  }

  function value(documentRef, id) {
    const node = documentRef.getElementById(id);
    return node ? node.value : "";
  }

  function checked(documentRef, id) {
    const node = documentRef.getElementById(id);
    return Boolean(node && node.checked);
  }

  function positiveInteger(value, fallback) {
    const parsed = Number.parseInt(value, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
  }

  function text(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  function slug(value) {
    return text(value).toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "ui-run";
  }

  function cleanTargets(targets) {
    if (!Array.isArray(targets)) return [];
    return targets.map(text).filter(Boolean);
  }

  return {
    buildOfflineRunnerArgs,
    computeTargets,
    configureComputeTargets,
    configureFromRuntimeConfig,
    mount,
    validateControllerInput,
  };
});
