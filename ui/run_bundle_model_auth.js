(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleModelAuth = api;
  if (root && root.document) {
    root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document, root.fetch));
  }
})(typeof window !== "undefined" ? window : globalThis, function createModelAuthClient() {
  function mount(documentRef, fetcher) {
    const form = documentRef.getElementById("model-auth-form");
    if (form && typeof fetcher === "function") {
      form.addEventListener("submit", (event) => submitLogin(documentRef, fetcher, event));
    }
    const smoke = documentRef.getElementById("model-gateway-smoke");
    if (smoke && typeof fetcher === "function") {
      smoke.addEventListener("click", () => submitSmoke(documentRef, fetcher));
    }
  }

  function submitLogin(documentRef, fetcher, event) {
    event.preventDefault();
    const provider = value(documentRef, "model-provider") || "oauth_gateway";
    const authMode = value(documentRef, "auth-mode") || "oauth";
    const accessToken = value(documentRef, "model-access-token");
    const refreshToken = value(documentRef, "model-refresh-token");
    return fetcher("/api/model/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider,
        auth_mode: authMode,
        api_key_env: value(documentRef, "model-api-key-env") || "MODEL_GATEWAY_TOKEN",
        access_token: accessToken,
        refresh_token: refreshToken,
        expires_in_s: 3600,
      }),
    })
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then((result) => {
        renderStatus(documentRef, result.body);
        clearSecret(documentRef, "model-access-token");
        clearSecret(documentRef, "model-refresh-token");
        return result.body;
      });
  }

  function submitSmoke(documentRef, fetcher) {
    const request = parseRequest(value(documentRef, "agent-request-json"));
    const endpoint = {
      provider: value(documentRef, "model-provider") || "oauth_gateway",
      model: value(documentRef, "model-name") || "gpt-5.5",
      reasoning_effort: value(documentRef, "reasoning-effort") || "high",
      base_url: value(documentRef, "model-base-url"),
      auth_mode: value(documentRef, "auth-mode") || "gateway",
      api_key_env: value(documentRef, "model-api-key-env") || "MODEL_GATEWAY_TOKEN",
    };
    return fetcher("/api/model/gateway/smoke", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ llm_endpoint: endpoint, request }),
    })
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then((result) => {
        renderStatus(documentRef, result.body);
        return result.body;
      });
  }

  function renderStatus(documentRef, body) {
    const node = documentRef.getElementById("model-auth-status");
    if (!node) return;
    const error = text(body.error);
    if (error) {
      node.textContent = `Blocked: ${error}`;
      return;
    }
    const requestId = text(body.gateway_request_id);
    if (requestId) {
      node.textContent = `gateway smoke ok / ${requestId}`;
      return;
    }
    const provider = text(body.provider) || "model";
    const loggedIn = body.logged_in === true || body.ok === true;
    node.textContent = loggedIn ? `${provider} logged in` : `${provider} not logged in`;
  }

  function parseRequest(raw) {
    try {
      return JSON.parse(raw || "{}");
    } catch (error) {
      if (error instanceof SyntaxError) return {};
      throw error;
    }
  }

  function clearSecret(documentRef, id) {
    const node = documentRef.getElementById(id);
    if (node) node.value = "";
  }

  function value(documentRef, id) {
    const node = documentRef.getElementById(id);
    return node ? node.value : "";
  }

  function text(value) {
    return typeof value === "string" && value.length > 0 ? value : "";
  }

  return { mount, submitLogin, submitSmoke };
});
