// agent-harness panel — thin HTTP client over the FastAPI harness.
// No ReAct logic lives here: POST /sessions then POST /chat, render the
// returned envelope. Same-origin, so no CORS handling needed.

"use strict";

const els = {
  workspace: document.getElementById("workspace"),
  provider: document.getElementById("provider"),
  newSession: document.getElementById("new-session"),
  sessionPill: document.getElementById("session-pill"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  envelope: document.getElementById("envelope"),
};

// Persist a stable user_id across reloads so cross-session memory can be
// demoed (facts written in one session surface in the next).
const userId = (() => {
  let id = localStorage.getItem("panel_user_id");
  if (!id) {
    id = "panel-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("panel_user_id", id);
  }
  return id;
})();

let sessionId = null;
let busy = false;

// ---- helpers ----------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function clearEmptyState() {
  const empty = els.messages.querySelector(".empty-state");
  if (empty) empty.remove();
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setBusy(value) {
  busy = value;
  const canSend = !busy && sessionId != null;
  els.send.disabled = !canSend;
  els.input.disabled = sessionId == null;
  els.newSession.disabled = busy;
}

function pretty(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

// ---- rendering --------------------------------------------------------

function addUserMessage(text) {
  clearEmptyState();
  els.messages.appendChild(el("div", "msg user", text));
  scrollToBottom();
}

function addErrorMessage(text) {
  clearEmptyState();
  els.messages.appendChild(el("div", "msg error", text));
  scrollToBottom();
}

function addSpinner() {
  const spinner = el("div", "spinner");
  spinner.appendChild(el("span", "dot"));
  spinner.appendChild(el("span", null, "agent working…"));
  els.messages.appendChild(spinner);
  scrollToBottom();
  return spinner;
}

function toolCard(call) {
  const card = el("details", "tool-card");
  if (call.error) card.classList.add("errored");

  const summary = el("summary");
  summary.appendChild(el("span", "tool-name", call.name));
  if (call.latency_ms != null) {
    summary.appendChild(el("span", "tool-latency", `${call.latency_ms.toFixed(1)}ms`));
  }
  card.appendChild(summary);

  const body = el("div", "tool-body");
  if (call.arguments && Object.keys(call.arguments).length > 0) {
    body.appendChild(el("div", "label", "arguments"));
    body.appendChild(el("pre", null, pretty(call.arguments)));
  }
  if (call.error) {
    body.appendChild(el("div", "label", "error"));
    body.appendChild(el("pre", "err", call.error));
  } else if (call.result != null && call.result !== "") {
    body.appendChild(el("div", "label", "result"));
    body.appendChild(el("pre", null, pretty(call.result)));
  }
  card.appendChild(body);
  return card;
}

function addTurn(data) {
  clearEmptyState();
  const block = el("div", "turn-block");

  const calls = data.tool_calls || [];
  if (calls.length > 0) {
    const tools = el("div", "tools");
    for (const call of calls) tools.appendChild(toolCard(call));
    block.appendChild(tools);
  }

  block.appendChild(el("div", "msg assistant", data.answer || "(no answer)"));
  els.messages.appendChild(block);
  scrollToBottom();
}

function confidenceClass(conf) {
  if (conf == null) return "conf-mid";
  if (conf >= 0.7) return "conf-high";
  if (conf >= 0.5) return "conf-mid";
  return "conf-low";
}

function listSection(label, items) {
  const section = el("div", "env-section");
  if (!items || items.length === 0) section.classList.add("empty");
  section.appendChild(el("div", "label", label));
  const ul = el("ul");
  for (const item of items || []) ul.appendChild(el("li", null, pretty(item)));
  section.appendChild(ul);
  return section;
}

function renderEnvelope(data) {
  els.envelope.replaceChildren();

  const badges = el("div", "badges");
  const conf = data.confidence;
  const confLabel = conf == null ? "conf n/a" : `conf ${conf.toFixed(2)}`;
  badges.appendChild(el("span", `badge ${confidenceClass(conf)}`, confLabel));
  badges.appendChild(
    el(
      "span",
      `badge ${data.escalated ? "escalated" : "ok"}`,
      data.escalated ? "escalated" : "not escalated",
    ),
  );
  badges.appendChild(el("span", "badge ok", data.provider || "—"));
  badges.appendChild(
    el("span", "badge ok", `${(data.latency_ms || 0).toFixed(0)}ms`),
  );
  badges.appendChild(
    el("span", `badge ${data.verification_ran ? "conf-high" : "ok"}`,
      data.verification_ran ? "verified" : "no verify"),
  );
  els.envelope.appendChild(badges);

  els.envelope.appendChild(listSection("citations", data.citations));
  els.envelope.appendChild(listSection("files touched", data.files_touched));
  els.envelope.appendChild(listSection("patch summary", data.patch_summary));
  els.envelope.appendChild(listSection("memory writes", data.memory_writes));
}

// ---- API --------------------------------------------------------------

async function createSession() {
  const body = { user_id: userId };
  const workspace = els.workspace.value.trim();
  if (workspace) body.workspace_root = workspace;

  setBusy(true);
  try {
    const resp = await fetch("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const detail = await resp.text();
      addErrorMessage(`Could not create session (${resp.status}): ${detail}`);
      return;
    }
    const data = await resp.json();
    sessionId = data.session_id;
    els.sessionPill.textContent = sessionId.slice(0, 8);
    els.sessionPill.classList.remove("muted");
    els.messages.replaceChildren();
    els.input.focus();
  } catch (err) {
    addErrorMessage(`Network error creating session: ${err}`);
  } finally {
    setBusy(false);
  }
}

async function sendMessage(message) {
  const body = { user_id: userId, session_id: sessionId, message };
  const provider = els.provider.value;
  if (provider) body.provider = provider;

  addUserMessage(message);
  const spinner = addSpinner();
  setBusy(true);
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    spinner.remove();
    if (!resp.ok) {
      const detail = await resp.text();
      addErrorMessage(`Turn failed (${resp.status}): ${detail}`);
      return;
    }
    const data = await resp.json();
    addTurn(data);
    renderEnvelope(data);
  } catch (err) {
    spinner.remove();
    addErrorMessage(`Network error: ${err}`);
  } finally {
    setBusy(false);
  }
}

// ---- events -----------------------------------------------------------

els.newSession.addEventListener("click", createSession);

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || sessionId == null) return;
  const message = els.input.value.trim();
  if (!message) return;
  els.input.value = "";
  sendMessage(message);
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

setBusy(false);
