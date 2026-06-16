(function attach(root, factory) {
  const controller = root && root.RunBundleController
    ? root.RunBundleController
    : require("./run_bundle_controller.js");
  const api = factory(controller);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleRuntimeConfig = api;
  if (root && root.document) {
    root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document, root.fetch));
  }
})(typeof window !== "undefined" ? window : globalThis, function createRuntimeConfigApi(controller) {
  function mount(documentRef, fetcher) {
    if (typeof fetcher !== "function") return Promise.resolve(null);
    const saveButton = documentRef.getElementById("runtime-config-save");
    if (saveButton) {
      saveButton.addEventListener("click", () => saveRuntimeConfig(documentRef, fetcher));
    }
    return loadRuntimeConfig(documentRef, fetcher);
  }

  function loadRuntimeConfig(documentRef, fetcher) {
    return fetcher("/api/runtime/config")
      .then((response) => response.json())
      .then((body) => {
        const config = body.runtime_config || {};
        applyRuntimeConfig(documentRef, config);
        return config;
      });
  }

  function saveRuntimeConfig(documentRef, fetcher) {
    const parsed = parseRuntimeConfig(documentRef);
    const status = documentRef.getElementById("runtime-config-status");
    if (parsed.error) {
      setText(status, `Blocked: ${parsed.error}`);
      return Promise.resolve({ error: parsed.error });
    }
    return fetcher("/api/runtime/config", {
      method: "POST",
      headers: csrfHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ runtime_config: parsed.config }),
    })
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then((result) => {
        if (!result.ok) {
          setText(status, `Blocked: ${text(result.body.error) || "runtime_config_save_failed"}`);
          return result.body;
        }
        applyRuntimeConfig(documentRef, result.body.runtime_config);
        setText(status, "Runtime config saved");
        return result.body;
      });
  }

  function applyRuntimeConfig(documentRef, config) {
    controller.configureFromRuntimeConfig(config);
    globalThis.__ASA_RUNTIME_CONFIG__ = cloneConfig(config);
    globalThis.__ASA_MODEL_ENDPOINT__ = config.model_endpoint || {};
    setValue(documentRef, "runtime-workspace-root", config.workspace_root);
    setValue(documentRef, "runtime-evidence-root", config.evidence_root);
    populateComputeTargets(documentRef, config.compute_resources || []);
    populateModelEndpoint(documentRef, config.model_endpoint || {});
    const editor = documentRef.getElementById("runtime-compute-resources");
    if (editor) editor.value = JSON.stringify(config.compute_resources || [], null, 2);
  }

  function parseRuntimeConfig(documentRef) {
    const loaded = runtimeConfig();
    const current = {
      ...loaded,
      workspace_root: value(documentRef, "runtime-workspace-root"),
      evidence_root: value(documentRef, "runtime-evidence-root"),
      team_mode_default: booleanOrDefault(loaded.team_mode_default, true),
      model_endpoint: readModelEndpoint(documentRef),
      graphdb: objectOrDefault(loaded.graphdb, {}),
    };
    const rawResources = value(documentRef, "runtime-compute-resources");
    try {
      return { config: { ...current, compute_resources: JSON.parse(rawResources || "[]") } };
    } catch (error) {
      if (error instanceof SyntaxError) return { error: "invalid_compute_resources_json" };
      throw error;
    }
  }

  function populateComputeTargets(documentRef, resources) {
    const select = documentRef.getElementById("compute-target");
    if (!select) return;
    const owner = select.ownerDocument || documentRef;
    select.replaceChildren(...resources.map((resource) => {
      const option = owner.createElement("option");
      option.value = text(resource.host_alias);
      option.textContent = text(resource.host_alias);
      return option;
    }));
  }

  function populateModelEndpoint(documentRef, endpoint) {
    setValue(documentRef, "model-provider", endpoint.provider);
    setValue(documentRef, "model-name", endpoint.model);
    setValue(documentRef, "reasoning-effort", endpoint.reasoning_effort);
    setValue(documentRef, "auth-mode", endpoint.auth_mode);
    setValue(documentRef, "model-base-url", endpoint.base_url);
    setValue(documentRef, "model-api-key-env", endpoint.api_key_env);
  }

  function readModelEndpoint(documentRef) {
    return {
      provider: value(documentRef, "model-provider"),
      model: value(documentRef, "model-name"),
      reasoning_effort: value(documentRef, "reasoning-effort"),
      auth_mode: value(documentRef, "auth-mode"),
      base_url: value(documentRef, "model-base-url"),
      api_key_env: value(documentRef, "model-api-key-env"),
    };
  }

  function runtimeConfig() {
    return objectOrDefault(globalThis.__ASA_RUNTIME_CONFIG__, {});
  }

  function cloneConfig(config) {
    return JSON.parse(JSON.stringify(objectOrDefault(config, {})));
  }

  function objectOrDefault(value, fallback) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : fallback;
  }

  function booleanOrDefault(value, fallback) {
    return typeof value === "boolean" ? value : fallback;
  }

  function csrfHeaders(headers) {
    const token = globalThis.__ASA_CONTROLLER_TOKEN__;
    return token ? { ...headers, "X-ASA-CSRF-Token": token } : headers;
  }

  function setValue(documentRef, id, content) {
    const node = documentRef.getElementById(id);
    if (node && typeof content === "string") node.value = content;
  }

  function setText(node, content) {
    if (node) node.textContent = content;
  }

  function value(documentRef, id) {
    const node = documentRef.getElementById(id);
    return node ? node.value : "";
  }

  function text(value) {
    return typeof value === "string" ? value.trim() : "";
  }

  return {
    applyRuntimeConfig,
    loadRuntimeConfig,
    mount,
    parseRuntimeConfig,
    saveRuntimeConfig,
  };
});
