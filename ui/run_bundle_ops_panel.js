(function attach(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.RunBundleOpsPanel = api;
})(typeof window !== "undefined" ? window : globalThis, function createOpsPanel() {
  function renderAgentStatusBoard(documentRef, statuses) {
    const node = documentRef.getElementById("agent-status-board");
    if (!node) return;
    node.className = "agent-status-board agent-dot-board";
    replace(node, statuses.map((status, index) => agentNode(documentRef, status, index)));
  }

  function agentNode(documentRef, status, index) {
    const card = documentRef.createElement("div");
    const state = classToken(text(status.status));
    card.className = `agent-node ${state}`;
    card.title = text(status.detail || status.summary);
    const avatar = documentRef.createElement("span");
    avatar.className = `agent-dot agent-dot-${(index % 5) + 1}`;
    avatar.textContent = shortLabel(status);
    const body = documentRef.createElement("span");
    body.className = "agent-node-body";
    const label = documentRef.createElement("strong");
    label.textContent = text(status.label || status.agent_id);
    const activity = documentRef.createElement("span");
    activity.className = "agent-activity";
    activity.textContent = text(status.status);
    const summary = documentRef.createElement("p");
    summary.className = "agent-card-message";
    summary.textContent = text(status.summary);
    body.append(label);
    body.append(activity);
    body.append(summary);
    card.append(avatar);
    card.append(body);
    return card;
  }

  function shortLabel(status) {
    const label = text(status.label || status.agent_id);
    const tokens = label.split(/[^A-Za-z0-9]+/).filter(Boolean);
    if (tokens.length > 1) return `${tokens[0][0] || ""}${tokens[1][0] || ""}`.toUpperCase();
    return label.slice(0, 2).toUpperCase();
  }

  function renderMessageLog(documentRef, messages) {
    const node = documentRef.getElementById("agent-message-log");
    if (!node) return;
    replace(node, messages.map((message) => {
      const row = documentRef.createElement("p");
      row.textContent = `${text(message.sender)} -> ${text(message.recipient)}: ${text(message.message)}`;
      return row;
    }));
  }

  function renderRunLog(documentRef, lines) {
    const node = documentRef.getElementById("run-log");
    if (!node) return;
    replace(node, lines.map((line) => {
      const row = documentRef.createElement("p");
      row.textContent = text(line);
      return row;
    }));
  }

  function renderArtifactLinks(documentRef, links) {
    const node = documentRef.getElementById("artifact-links");
    if (!node) return;
    replace(node, Object.keys(links).sort().map((key) => {
      const row = documentRef.createElement("p");
      row.textContent = `${key}: ${links[key]}`;
      return row;
    }));
  }

  function renderQaReport(documentRef, status, blockers) {
    setText(documentRef, "qa-report-summary", `QA: ${text(status)}`);
    const node = documentRef.getElementById("qa-hard-blockers");
    if (!node) return;
    const values = Array.isArray(blockers) ? blockers : [];
    node.className = values.length > 0 ? "qa-hard-blockers blocked" : "qa-hard-blockers";
    node.textContent = values.length > 0 ? `Hard blockers: ${values.join(", ")}` : "Hard blockers: none";
  }

  function renderProductionReadiness(documentRef, readiness) {
    const ready = readiness && readiness.productionReady === true;
    const blockers = readiness && Array.isArray(readiness.hardBlockers) ? readiness.hardBlockers : [];
    const actions = readiness && Array.isArray(readiness.userActions) ? readiness.userActions : [];
    const actionPlan = readiness && Array.isArray(readiness.actionPlan) ? readiness.actionPlan : [];
    const summary = documentRef.getElementById("production-readiness-summary");
    if (summary) {
      summary.className = ready ? "production-readiness-summary ready" : "production-readiness-summary blocked";
      summary.textContent = ready ? "Production ready: true" : "Production ready: false";
    }
    const blockerNode = documentRef.getElementById("production-hard-blockers");
    if (blockerNode) {
      blockerNode.className = blockers.length > 0 ? "production-hard-blockers blocked" : "production-hard-blockers";
    }
    setText(
      documentRef,
      "production-hard-blockers",
      blockers.length > 0 ? `Hard blockers: ${blockers.join(", ")}` : "Hard blockers: none",
    );
    setText(
      documentRef,
      "production-user-actions",
      actions.length > 0 ? `User actions: ${actions.join(", ")}` : "User actions: none",
    );
    renderActionPlan(documentRef, actionPlan);
  }

  function renderActionPlan(documentRef, actions) {
    const node = documentRef.getElementById("production-action-plan");
    if (!node) return;
    const values = Array.isArray(actions) ? actions : [];
    if (values.length === 0) {
      node.textContent = "Action plan: none";
      return;
    }
    replace(node, values.map((action) => actionRow(documentRef, action)));
  }

  function actionRow(documentRef, action) {
    const row = documentRef.createElement("p");
    const status = text(action.status);
    row.className = `production-action ${classToken(status)}`;
    row.textContent = `${text(action.actor || "agent")}: ${text(action.action)} (${status})`;
    row.title = actionTooltip(action);
    return row;
  }

  function actionTooltip(action) {
    const parts = [];
    if (Array.isArray(action.command) && action.command.length > 0) {
      parts.push(`command: ${action.command.join(" ")}`);
    }
    if (action.requires_user_action) parts.push(`requires: ${text(action.requires_user_action)}`);
    if (Array.isArray(action.missing_artifacts) && action.missing_artifacts.length > 0) {
      parts.push(`missing: ${action.missing_artifacts.join(", ")}`);
    }
    if (Array.isArray(action.hard_blockers) && action.hard_blockers.length > 0) {
      parts.push(`blockers: ${action.hard_blockers.join(", ")}`);
    }
    if (Array.isArray(action.next_actions) && action.next_actions.length > 0) {
      parts.push(`next: ${action.next_actions.join(", ")}`);
    }
    if (Array.isArray(action.acceptable_sources) && action.acceptable_sources.length > 0) {
      parts.push(`sources: ${action.acceptable_sources.join(", ")}`);
    }
    if (Array.isArray(action.expected_artifacts) && action.expected_artifacts.length > 0) {
      parts.push(`expects: ${action.expected_artifacts.join(", ")}`);
    }
    return parts.join(" | ") || text(action.action);
  }

  function replace(node, children) {
    if (typeof node.replaceChildren === "function") {
      node.replaceChildren(...children);
      return;
    }
    node.textContent = children.map((child) => child.textContent).join("");
  }

  function setText(documentRef, id, content) {
    const node = documentRef.getElementById(id);
    if (node) node.textContent = content;
  }

  function classToken(value) {
    return text(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }

  function text(value) { return typeof value === "string" && value.length > 0 ? value : "-"; }

  return {
    renderAgentStatusBoard,
    renderArtifactLinks,
    renderMessageLog,
    renderProductionReadiness,
    renderQaReport,
    renderRunLog,
  };
});
