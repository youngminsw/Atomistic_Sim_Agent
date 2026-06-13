(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleAgentClient = api;
  if (root && root.document) {
    root.document.addEventListener("DOMContentLoaded", () => api.mount(root.document, root.fetch));
  }
})(typeof window !== "undefined" ? window : globalThis, function createAgentClient() {
  function mount(documentRef, fetcher) {
    const form = documentRef.getElementById("agent-plan-form");
    if (!form || typeof fetcher !== "function") return;
    form.addEventListener("submit", (event) => submitPlan(documentRef, fetcher, event));
  }

  function submitPlan(documentRef, fetcher, event) {
    event.preventDefault();
    const log = documentRef.getElementById("chat-log");
    const parsed = parsePayload(value(documentRef, "agent-request-json"));
    if (parsed.error) {
      pushMessage(documentRef, log, "agent", `Blocked: ${parsed.error}`);
      return Promise.resolve({ error: parsed.error });
    }
    const body = JSON.stringify(applyModelSettings(documentRef, JSON.parse(parsed.body)));
    pushMessage(documentRef, log, "user", "Plan request");
    return fetcher("/api/agent/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    })
      .then((response) => response.json().then((body) => ({ ok: response.ok, body })))
      .then((result) => {
        pushMessage(documentRef, log, "agent", formatAgentPlan(result.body));
        return result.body;
      });
  }

  function formatAgentPlan(body) {
    const error = text(body.error);
    if (error) return `Blocked: ${error}`;
    const status = text(body.status);
    if (status === "clarification_required") {
      return `Missing: ${missingFields(body)}. ${text(body.final_output)}${teamSessionSummary(body)}`;
    }
    if (status === "planned") {
      return `Planned: ${text(body.run_id)} / ${number(body.artifact_count)} artifacts${teamSessionSummary(body)}`;
    }
    return `${text(body.final_output) || "No planner response"}${teamSessionSummary(body)}`;
  }

  function teamSessionSummary(body) {
    const contract = body.team_session_contract;
    if (!contract || typeof contract !== "object") return "";
    const callMatrix = contract.call_matrix && typeof contract.call_matrix === "object" ? contract.call_matrix : {};
    const agents = new Set(Object.keys(callMatrix));
    Object.values(callMatrix).forEach((peers) => {
      if (Array.isArray(peers)) peers.forEach((peer) => agents.add(text(peer)));
    });
    const agentCount = Array.from(agents).filter(Boolean).length;
    const heartbeat = number(contract.heartbeat_interval_s);
    const qaGates = contract.qa_gates && typeof contract.qa_gates === "object" ? contract.qa_gates : {};
    const slurmGate = text(qaGates.slurm_job_script) || "-";
    return ` Team: ${agentCount} agents / heartbeat ${heartbeat}s / QA Slurm gate ${slurmGate}`;
  }

  function parsePayload(raw) {
    try {
      return { body: JSON.stringify(JSON.parse(raw)) };
    } catch (error) {
      if (error instanceof SyntaxError) return { error: "invalid_request_json" };
      throw error;
    }
  }

  function applyModelSettings(documentRef, payload) {
    const endpoint = payload.llm_endpoint && typeof payload.llm_endpoint === "object" ? payload.llm_endpoint : {};
    return {
      ...payload,
      llm_endpoint: {
        ...endpoint,
        provider: text(value(documentRef, "model-provider")) || text(endpoint.provider) || "openclaw",
        model: text(value(documentRef, "model-name")) || text(endpoint.model) || "gpt-5.5",
        reasoning_effort: text(value(documentRef, "reasoning-effort")) || text(endpoint.reasoning_effort) || "high",
        base_url: text(value(documentRef, "model-base-url")) || text(endpoint.base_url) || "https://openclaw.local/v1",
        auth_mode: text(value(documentRef, "auth-mode")) || text(endpoint.auth_mode) || "oauth",
        api_key_env: text(value(documentRef, "model-api-key-env")) || text(endpoint.api_key_env) || "OPENCLAW_OAUTH_TOKEN",
      },
    };
  }

  function missingFields(body) {
    const fields = Array.isArray(body.missing_fields) ? body.missing_fields.map(text).filter(Boolean) : [];
    return fields.length > 0 ? fields.join(", ") : "-";
  }

  function pushMessage(documentRef, container, role, textValue) {
    if (!container) return;
    const owner = container.ownerDocument || documentRef;
    const node = owner.createElement("div");
    node.className = role === "user" ? "message user" : "message";
    node.textContent = textValue;
    container.append(node);
    container.scrollTop = container.scrollHeight;
  }

  function value(documentRef, id) {
    const node = documentRef.getElementById(id);
    return node ? node.value : "";
  }

  function number(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }

  function text(value) {
    return typeof value === "string" && value.length > 0 ? value : "";
  }

  return { applyModelSettings, formatAgentPlan, mount, submitPlan };
});
