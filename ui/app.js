// agent-harness panel — thin HTTP client over the FastAPI harness.
// No ReAct logic lives here: POST /sessions then POST /chat, render the
// returned envelope. Same-origin, so no CORS handling needed.

"use strict";

const els = {
  workspace: document.getElementById("workspace"),
  provider: document.getElementById("provider"),
  checkpointTurns: document.getElementById("checkpoint-turns"),
  cancelTurn: document.getElementById("cancel-turn"),
  newSession: document.getElementById("new-session"),
  sessionPill: document.getElementById("session-pill"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  envelope: document.getElementById("envelope"),
  refreshRuns: document.getElementById("refresh-runs"),
  history: document.getElementById("run-history"),
  previousRuns: document.getElementById("previous-runs"),
  nextRuns: document.getElementById("next-runs"),
  pendingControls: document.getElementById("pending-controls"),
  checkRun: document.getElementById("check-run"),
  retryRun: document.getElementById("retry-run"),
  forgetRun: document.getElementById("forget-run"),
  sessionReview: document.getElementById("session-review"),
  reviewTarget: document.getElementById("review-target"),
  reviewWorkspace: document.getElementById("review-workspace"),
  reviewTranscript: document.getElementById("review-transcript"),
  reviewConfiguration: document.getElementById("review-configuration"),
  checkSuggestions: document.getElementById("check-suggestions"),
  reviewProvider: document.getElementById("review-provider"),
  reviewMessage: document.getElementById("review-message"),
  reviewResults: document.getElementById("session-review-results"),
  prepareContinuation: document.getElementById("prepare-continuation"),
  continuationTarget: document.getElementById("continuation-target"),
  continuationMessage: document.getElementById("continuation-message"),
  continuationAck: document.getElementById("continuation-ack"),
  submitContinuation: document.getElementById("submit-continuation"),
  runProgress: document.getElementById("run-progress"),
  progressTarget: document.getElementById("progress-target"),
  inspectToolProgress: document.getElementById("inspect-tool-progress"),
  progressResults: document.getElementById("tool-progress-results"),
  reversal: document.getElementById("file-reversal"),
  reversalTarget: document.getElementById("reversal-target"),
  reversalPath: document.getElementById("reversal-path"),
  reversalAck: document.getElementById("reversal-ack"),
  submitReversal: document.getElementById("submit-reversal"),
  reversalPending: document.getElementById("reversal-pending"),
  reversalRequest: document.getElementById("reversal-request"),
  checkReversal: document.getElementById("check-reversal"),
  retryReversal: document.getElementById("retry-reversal"),
  forgetReversal: document.getElementById("forget-reversal"),
  reversalResults: document.getElementById("reversal-results"),
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
let sessionUsable = true;
let uncertain = false;
let cancelRequested = false;
let pendingRequest = null;
let historyOffset = 0;
let reviewTarget = null;
let reviewInFlight = false;
let continuationTurnId = null;
const pendingStorageKey = "panel_pending_request";
const reversalStorageKey = "panel_pending_reversal";
let pendingReversal = null;
let reversalStatus = null;
let reversalPairAvailable = false;
let reversalRecoveryBlocked = false;

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
  const canSend = !busy && !pendingReversal && !reversalRecoveryBlocked && sessionId != null && sessionUsable;
  els.send.disabled = !canSend;
  els.input.disabled = sessionId == null || !sessionUsable;
  const cancellationTarget = pendingRequest?.kind === "continuation"
    ? pendingRequest.fork_session_id : pendingRequest?.session_id || sessionId;
  els.cancelTurn.disabled = !!pendingReversal || !cancellationTarget || !(busy || uncertain) || cancelRequested;
  els.cancelTurn.textContent = cancelRequested ? "Cancelling…" : "Cancel turn";
  els.newSession.disabled = busy || pendingRequest != null || pendingReversal != null || reversalRecoveryBlocked;
  els.refreshRuns.disabled = busy;
  els.previousRuns.disabled = busy || historyOffset === 0;
  els.nextRuns.disabled = busy || !els.nextRuns.dataset.more;
  els.pendingControls.hidden = pendingRequest == null;
  els.checkRun.disabled = busy;
  els.retryRun.disabled = busy || !!pendingReversal || reversalRecoveryBlocked;
  els.forgetRun.disabled = busy;
  for (const button of els.history.querySelectorAll("button")) button.disabled = busy;
  els.submitContinuation.disabled = busy || reviewInFlight || pendingRequest != null || pendingReversal != null || reversalRecoveryBlocked ||
    reviewTarget == null || !continuationTurnId || !els.continuationAck.checked ||
    !els.continuationMessage.value.trim();
  els.prepareContinuation.disabled = busy || reviewInFlight || reviewTarget == null;
  for (const button of [els.reviewWorkspace, els.reviewTranscript, els.reviewConfiguration, els.checkSuggestions, els.inspectToolProgress]) {
    button.disabled = busy || reviewInFlight || reviewTarget == null;
  }
  els.reversal.hidden = reviewTarget == null && pendingReversal == null && !reversalRecoveryBlocked;
  els.reversalPending.hidden = pendingReversal == null && !reversalRecoveryBlocked;
  els.submitReversal.disabled = busy || reviewInFlight || !!pendingRequest || !!pendingReversal || reversalRecoveryBlocked ||
    !reviewTarget || !reversalPairAvailable || !els.reversalAck.checked ||
    !canonicalReversalPath(els.reversalPath.value);
  els.checkReversal.disabled = busy || reviewInFlight || !pendingReversal;
  els.retryReversal.disabled = busy || reviewInFlight || !!pendingRequest || !pendingReversal || reversalStatus != null;
  els.forgetReversal.disabled = busy || reviewInFlight;
  if (pendingReversal) els.reversalRequest.textContent = `Retained request: ${pendingReversal.request_id}; run: ${pendingReversal.run_id}; file: ${pendingReversal.path}. Reload submits nothing. Inspect first; retry preserves this exact input. Forgetting does not cancel or reconcile file work.`;
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

// A tool card opened live on `tool_start`, before its result is known.
function runningToolCard(ev) {
  const card = el("details", "tool-card running");
  card.open = true;
  card._tool = ev.tool;
  card._done = false;

  const summary = el("summary");
  summary.appendChild(el("span", "tool-name", ev.tool));
  summary.appendChild(el("span", "tool-latency", "…"));
  card.appendChild(summary);

  const body = el("div", "tool-body");
  if (ev.arguments && Object.keys(ev.arguments).length > 0) {
    body.appendChild(el("div", "label", "arguments"));
    body.appendChild(el("pre", null, pretty(ev.arguments)));
  }
  const placeholder = el("div", "label", "running…");
  placeholder.dataset.pending = "1";
  body.appendChild(placeholder);
  card.appendChild(body);
  return card;
}

// Fill a running card in place when its matching `tool_end` arrives.
function completeToolCard(card, ev) {
  card.classList.remove("running");
  card._done = true;
  const latency = card.querySelector(".tool-latency");
  if (latency) latency.textContent = `${ev.latency_ms.toFixed(1)}ms`;

  const body = card.querySelector(".tool-body");
  const placeholder = body.querySelector('[data-pending="1"]');
  if (placeholder) placeholder.remove();

  if (ev.error) {
    card.classList.add("errored");
    body.appendChild(el("div", "label", "error"));
    body.appendChild(el("pre", "err", ev.error));
  } else if (ev.result_snippet) {
    body.appendChild(el("div", "label", "result"));
    body.appendChild(el("pre", null, ev.result_snippet));
  }
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

function workspaceChangeItems(changes) {
  if (!changes || changes.status === "not_tracked") return [];
  if (changes.status === "unavailable") return [`unavailable: ${changes.reason || "unknown"}`];
  return [
    ...(changes.added || []).map((path) => `added: ${path}`),
    ...(changes.modified || []).map((path) => `modified: ${path}`),
    ...(changes.deleted || []).map((path) => `deleted: ${path}`),
  ];
}

function ignoredEntrySection(data) {
  const section = el("section", "env-section");
  section.appendChild(el("div", "label", `Ignored top-level metadata: ${data?.status || "not_recorded"}`));
  for (const kind of ["added", "modified", "deleted"]) {
    if (data?.[kind]?.length || data?.[kind + "_count"]) {
      section.appendChild(el("p", null, `${kind}: total ${data[kind + "_count"] ?? "unknown"}`));
      section.appendChild(listSection(kind, data[kind]));
    }
  }
  if (data?.paths_truncated) section.appendChild(el("p", null, "Ignored-entry list truncated; counts cover the full metadata comparison."));
  section.appendChild(el("p", "review-note", data?.scope_note || "Metadata only; ignored contents are not compared."));
  return section;
}

function workspaceDiffSection(changes) {
  const section = el("div", "env-section");
  const diffs = changes?.diffs || [];
  if (!diffs.length) section.classList.add("empty");
  section.appendChild(el("div", "label", "turn diffs"));
  for (const entry of diffs) {
    const details = el("details", "workspace-diff");
    details.appendChild(el("summary", null, entry.path));
    details.appendChild(el("pre", null, entry.reason ? `Diff omitted: ${entry.reason}` : entry.diff));
    section.appendChild(details);
  }
  return section;
}

function renderEnvelope(data, includeRunId = true) {
  els.envelope.replaceChildren();

  const badges = el("div", "badges");
  const status = data.completion_status || "unknown";
  const verification = data.verification_status || "unknown";
  badges.appendChild(el("span", `badge ${status === "completed" ? "ok" : "escalated"}`,
    `completion: ${status}`));
  badges.appendChild(el("span", `badge ${verification === "passed" ? "ok" : ["failed", "unavailable", "stale"].includes(verification) ? "escalated" : ""}`,
    `checks: ${verification}`));
  badges.appendChild(el("span", `badge ${data.escalated ? "escalated" : "ok"}`,
    data.escalated ? "review requested" : "no escalation flag"));
  badges.appendChild(el("span", "badge ok", data.provider || "—"));
  badges.appendChild(el("span", "badge ok", `loop: ${(data.latency_ms || 0).toFixed(0)}ms`));
  els.envelope.appendChild(badges);
  els.envelope.appendChild(el("p", "review-note",
    `Workspace review: ${data.workspace_review_ms == null ? "unmeasured" : data.workspace_review_ms.toFixed(1) + "ms"}; separate from loop latency. These are not total request latency or a hard return deadline.`));
  els.envelope.appendChild(el("p", "review-note",
    "Completion records how the run ended. Passed checks cover only the checks run; review the diff and remaining failures."));
  if (includeRunId && data.run_id) els.envelope.appendChild(listSection("run ID", [data.run_id]));


  els.envelope.appendChild(listSection("source locations from tools", data.citations));
  els.envelope.appendChild(listSection("files touched", data.files_touched));
  els.envelope.appendChild(listSection("workspace changes", workspaceChangeItems(data.workspace_changes)));
  if (data.workspace_changes?.status !== "not_tracked") {
    els.envelope.appendChild(ignoredEntrySection(data.workspace_changes?.ignored_entries));
  }
  els.envelope.appendChild(workspaceDiffSection(data.workspace_changes));
  els.envelope.appendChild(listSection("patch summary", data.patch_summary));
  els.envelope.appendChild(listSection("completion", [
    `${data.completion_status || "unknown"}${data.completion_reason ? `: ${data.completion_reason}` : ""}`,
    `verification: ${data.verification_status || "unknown"}`,
  ]));
  els.envelope.appendChild(listSection("model tokens", [
    `prompt: ${data.token_usage?.prompt_tokens ?? "unreported"}`,
    `output: ${data.token_usage?.completion_tokens ?? "unreported"}`,
  ]));
  els.envelope.appendChild(listSection("check attempts", (data.check_attempts || []).map(
    (check) => `${(check.argv || []).join(" ")}: ${check.status}`
      + `${check.exit_code == null ? "" : ` (exit ${check.exit_code})`}`
      + `${check.relevant ? "" : " (not configured check)"}`
      + `${check.superseded_by_edit ? " (superseded by later edit)" : ""}`
      + `${check.error ? `: ${check.error}` : ""}`,
  )));
  els.envelope.appendChild(listSection("tool errors", data.tool_errors));
  els.envelope.appendChild(listSection("memory writes", data.memory_writes));
  const legacy = el("details", "env-section");
  legacy.appendChild(el("summary", null, "Legacy evidence score"));
  legacy.appendChild(el("p", "review-note",
    `${data.confidence == null ? "Unavailable" : data.confidence.toFixed(2)} — tool-result heuristic. A file read can score 1.00; this does not measure answer or patch correctness. Source locations are harvested from tools, without checking the answer's claims.`));
  els.envelope.appendChild(legacy);

}

// ---- API --------------------------------------------------------------

async function createSession() {
  if (busy || pendingRequest != null || pendingReversal != null || reversalRecoveryBlocked) return;
  const body = { user_id: userId };
  const workspace = els.workspace.value.trim();
  if (workspace) body.workspace_root = workspace;
  if (els.checkpointTurns.checked) body.checkpoint_turns = true;

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
    selectReviewTarget(null);
    sessionId = data.session_id;
    sessionUsable = true;
    uncertain = false;
    cancelRequested = false;
    els.sessionPill.title = sessionId;
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

// Entry point for a turn. Prefer SSE (live tool cards); fall back to POST.
function sendMessage(message) {
  if (pendingReversal || reversalRecoveryBlocked) return false;
  let request;
  try {
    request = {
      user_id: userId, session_id: sessionId, message, request_id: crypto.randomUUID(),
    };
    if (els.provider.value) request.provider = els.provider.value;
    // Commit exact retry input locally before starting any server work.
    localStorage.setItem(pendingStorageKey, JSON.stringify(request));
  } catch (err) {
    addErrorMessage(`Could not retain submission: ${err}. No request was sent.`);
    return false;
  }
  pendingRequest = request;
  dispatchRequest(request);
  return true;
}

function dispatchRequest(request, reviewOnly = false) {
  if (pendingReversal || reversalRecoveryBlocked) return;
  if (request.kind === "continuation") { postContinuation({ ...request }); return; }
  selectReviewTarget(null);
  request = { ...request };
  const reviewAfter = reviewOnly || !sessionUsable;
  if (!reviewAfter) addUserMessage(request.message);
  else els.messages.appendChild(el("p", "review-note", "Retrying the retained submission…"));
  if (typeof window.EventSource === "function") {
    streamTurn(request, reviewAfter);
  } else {
    postTurn(request, reviewAfter);
  }
}

// Live path: open an EventSource and draw tool cards as events arrive.
function streamTurn(request, reviewAfter) {
  const block = el("div", "turn-block");
  const tools = el("div", "tools");
  block.appendChild(tools);
  const spinner = addSpinner();
  block.appendChild(spinner);
  els.messages.appendChild(block);
  scrollToBottom();
  setBusy(true);

  const params = new URLSearchParams(request);

  let es;
  try {
    es = new EventSource(`/chat/stream?${params.toString()}`);
  } catch (err) {
    block.remove();
    postTurn(request, reviewAfter);
    return;
  }

  const pending = [];
  let done = false;

  const finishWith = (node) => {
    spinner.remove();
    if (node) block.appendChild(node);
    es.close();
    setBusy(false);
    scrollToBottom();
  };

  es.addEventListener("tool_start", (e) => {
    const card = runningToolCard(JSON.parse(e.data));
    tools.appendChild(card);
    pending.push(card);
    scrollToBottom();
  });

  es.addEventListener("tool_end", (e) => {
    const data = JSON.parse(e.data);
    const card = pending.find((c) => c._tool === data.tool && !c._done);
    if (card) completeToolCard(card, data);
    scrollToBottom();
  });

  es.addEventListener("turn_done", (e) => {
    done = true;
    const data = JSON.parse(e.data).response;
    sessionUsable = !reviewAfter && data.completion_status !== "cancelled";
    cancelRequested = false;
    uncertain = false;
    clearPending();
    finishWith(el("div", "msg assistant", data.answer || "(no answer)"));
    renderEnvelope(data);
  });

  // Fires for both server-sent `error` events (with .data) and connection drops.
  es.addEventListener("error", (e) => {
    if (e.data) {
      done = true;
      let detail = e.data;
      try {
        detail = JSON.parse(e.data).detail;
      } catch (_) {
        /* keep raw */
      }
      sessionUsable = false;
      cancelRequested = false;
      uncertain = true;
      finishWith(el("div", "msg error", `Turn failed: ${detail}. Inspect saved review and create a new session.`));
      return;
    }
    if (done) return; // clean close after turn_done
    es.close();
    uncertain = true;
    sessionUsable = false;
    finishWith(el("div", "msg error",
      "Stream disconnected. The server run may still be active. Check run reads saved state; Retry submission uses the retained key. Nothing is retried automatically."));
  });
}

// Fallback path: single POST /chat, render the full turn when it returns.
async function postTurn(request, reviewAfter) {
  const body = request;

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
      uncertain = true;
      sessionUsable = false;
      addErrorMessage(`Turn failed (${resp.status}): ${detail}. Check run before new work.`);
      return;
    }
    const data = await resp.json();
    sessionUsable = !reviewAfter && data.completion_status !== "cancelled";
    cancelRequested = false;
    uncertain = false;
    clearPending();
    addTurn(data);
    renderEnvelope(data);
  } catch (err) {
    spinner.remove();
    uncertain = true;
    sessionUsable = false;
    addErrorMessage(`Network error: ${err}. The server run may still be active; use Check run, Retry submission, or Cancel turn.`);
  } finally {
    setBusy(false);
  }
}

async function cancelTurn() {
  cancelRequested = true;
  setBusy(busy);
  try {
    const target = pendingRequest?.kind === "continuation"
      ? pendingRequest.fork_session_id : pendingRequest?.session_id || sessionId;
    if (!target) { cancelRequested = false; return; }
    const resp = await fetch(`/sessions/${encodeURIComponent(target)}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      cancelRequested = false;
      addErrorMessage(`Cancellation not confirmed: ${data.detail}`);
    } else if (uncertain) {
      addErrorMessage("Cancellation requested. Inspect saved session review after owned work settles.");
    }
  } catch (err) {
    cancelRequested = false;
    addErrorMessage(`Cancellation not confirmed: ${err}`);
  } finally {
    setBusy(busy);
  }
}

// ---- saved session reviews: observations, separate from explicit work ----

function selectReviewTarget(record) {
  // Object identity prevents a delayed response from rendering on another selection.
  reviewTarget = record ? { sessionId: record.request.session_id, runId: record.run_id } : null;
  els.sessionReview.hidden = reviewTarget == null;
  els.runProgress.hidden = reviewTarget == null;
  els.progressResults.replaceChildren();
  els.reviewResults.replaceChildren();
  els.reviewMessage.value = "";
  els.reviewProvider.value = "";
  continuationTurnId = null;
  els.continuationMessage.value = "";
  els.continuationAck.checked = false;
  els.continuationTarget.textContent = "Load the latest saved turn to pin this request.";
  reversalPairAvailable = !!record?.before_checkpoint_id && !!record?.after_checkpoint_id && record.status !== "running";
  els.reversalPath.value = "";
  els.reversalAck.checked = false;
  els.reversalTarget.textContent = record
    ? `Selected run: ${record.run_id}; ${reversalPairAvailable ? "capture IDs recorded; file eligibility is checked on submission" : "no available finalized capture pair"}.`
    : "No run selected. Retained reversal inspection is separate from run selection.";
  if (reviewTarget) {
    els.progressTarget.textContent = `Run: ${reviewTarget.runId}; lifecycle when selected: ${record.status || "unknown"}`;
    els.reviewTarget.textContent = `Session: ${reviewTarget.sessionId}; selected run: ${reviewTarget.runId}`;
  }
}

function toolProgressCard(data) {
  const card = el("section", "session-review-card");
  card.appendChild(el("h3", null, `Metadata coverage: ${data.coverage || "unknown"}`));
  card.appendChild(el("p", "review-note", `Run: ${data.run_id}; latest saved observation: ${data.updated_at || "unrecorded"}`));
  const events = Array.isArray(data.events) ? data.events : null;
  card.appendChild(el("p", null, `Observed events: ${data.observed_events ?? "unknown"}; saved prefix: ${events?.length ?? "unknown"}`));
  if (data.coverage === "not_recorded") {
    card.appendChild(el("p", null, "No journal was recorded for this run. Workspace edits may exist."));
  } else if (data.coverage === "recorded" && events?.length === 0) {
    card.appendChild(el("p", null, "No tool observations saved. This does not establish no inference or no workspace changes."));
  }
  if (data.events_truncated) {
    card.appendChild(el("p", null, "Saved prefix truncated; later observations are counted but not listed. This is an incomplete event history."));
  }
  if (events?.length) {
    const details = el("details", "saved-tool-events");
    details.open = events.length <= 6;
    details.appendChild(el("summary", null, `Saved event prefix (${events.length} records)`));
    const list = el("ol");
    for (const event of events) {
      const row = el("li");
      row.appendChild(el("div", null, `${event.sequence}: ${event.type} — ${event.tool}`));
      row.appendChild(el("div", "review-note", `Observed: ${event.observed_at || "unknown"}`));
      if (event.type === "tool_end") {
        const error = event.error_observed == null ? "unknown" : event.error_observed ? "yes" : "no";
        row.appendChild(el("div", null, `Error observed: ${error}; tool latency: ${event.latency_ms == null ? "unreported" : event.latency_ms.toFixed(1) + "ms"}`));
      }
      list.appendChild(row);
    }
    details.appendChild(list);
    card.appendChild(details);
  }
  card.appendChild(el("p", "review-note", data.scope_note || "Metadata only. Start precedes tool gates; end/error presence is not successful verification or patch acceptance. A missing end does not prove a tool is still running. No replay or permission to resume."));
  return card;
}

function checkSuggestionsCard(data, sessionId) {
  const statuses = ["inspected", "partial", "unavailable", "busy", "no_workspace"];
  const strings = value => Array.isArray(value) && value.length > 0 && value.every(item => typeof item === "string");
  if (!data || data.session_id !== sessionId || !statuses.includes(data.status) ||
      data.read_only !== true || data.commands_executed !== false || data.configuration_changed !== false ||
      !Array.isArray(data.suggestions) || !Array.isArray(data.issues) || typeof data.scope_note !== "string" ||
      (data.reason != null && typeof data.reason !== "string") ||
      !data.suggestions.every(item => item && strings(item.argv) && strings(item.evidence)) ||
      !data.issues.every(item => item && typeof item.path === "string" && typeof item.code === "string")) {
    throw new Error("Inconsistent check suggestions response");
  }
  const card = el("section", "session-review-card");
  card.appendChild(el("h3", null, `Check suggestions: ${data.status}`));
  card.appendChild(el("p", "review-note", "Read-only static hints for the current workspace. No command ran and no configuration changed. Inspect setup before execution."));
  if (data.reason) card.appendChild(el("p", null, data.reason));
  for (const suggestion of data.suggestions) {
    card.appendChild(el("pre", null, JSON.stringify(suggestion.argv)));
    card.appendChild(el("p", "review-note", `Evidence: ${suggestion.evidence.join(", ")}`));
  }
  if (!data.suggestions.length) {
    card.appendChild(el("p", null, "No suggestions returned within the supported root-manifest scope."));
  }
  for (const issue of data.issues) card.appendChild(el("p", null, `${issue.path}: ${issue.code}`));
  card.appendChild(el("p", "review-note", data.scope_note));
  return card;
}

function currentProfileSection(selection) {
  const section = el("div", "current-profile-review");
  const identity = selection?.identity;
  const valid = selection?.status === "selected" && identity?.schema_version === 1 &&
    ["name", "provider", "model", "definition_sha256"].every(key => typeof identity[key] === "string") &&
    /^[a-f0-9]{64}$/.test(identity.definition_sha256);
  if (selection == null) {
    section.appendChild(el("p", null, "Current model profile: not checked or legacy response."));
  } else if (selection.status === "direct_settings" && selection.identity == null) {
    section.appendChild(el("p", null, "Current model profile: direct settings (no named profile selected)."));
  } else if (valid) {
    section.appendChild(el("p", null, `Current model profile: ${identity.name}`));
    section.appendChild(el("p", null, `Configured target: ${identity.provider} / ${identity.model}`));
    const details = el("details");
    details.appendChild(el("summary", null, "Selected definition identity"));
    details.appendChild(el("pre", null, identity.definition_sha256));
    section.appendChild(details);
  } else {
    section.appendChild(el("p", null, "Current model profile: unknown or inconsistent."));
  }
  section.appendChild(el("p", "review-note", "Current operator configuration only; no switching, inference or settings changes. Not backend/capability evidence or continuation permission."));
  return section;
}

function sessionReviewCard(kind, data, input = {}) {
  const card = el("section", "session-review-card");
  card.appendChild(el("h3", null, `${kind}: ${data.status || "unknown"}`));
  card.appendChild(el("p", "review-note", `Latest archived turn: ${data.turn_id || "unrecorded"}`));
  if (data.reason) card.appendChild(el("p", null, data.reason));
  if (kind === "workspace") {
    card.appendChild(el("p", null, `Regular-file mode coverage: ${data.mode_coverage || "not_recorded"}`));
    card.appendChild(el("p", "review-note", `Baseline: ${data.baseline_captured_at || "unrecorded"}; checked: ${data.checked_at || "unknown"}`));
    for (const name of ["added", "modified", "deleted"]) {
      card.appendChild(el("p", null, `${name}: total ${data[name + "_count"] ?? "unknown"}`));
      card.appendChild(listSection(name, data[name]));
    }
    if (data.paths_truncated) card.appendChild(el("p", null, "Path lists truncated; counts cover the full comparison."));
    card.appendChild(listSection("Excluded patterns", data.excluded_patterns));
    card.appendChild(ignoredEntrySection(data.ignored_entries));
  } else if (kind === "transcript") {
    card.appendChild(el("p", null, `Messages: ${data.message_count ?? "unknown"}; issues: ${data.issue_count ?? "unknown"}`));
    card.appendChild(listSection("Structural issues", (data.issues || []).map(issue => `message[${issue.message_index}]: ${issue.code}`)));
    if (data.issues_truncated) card.appendChild(el("p", null, "Issue list truncated; count covers the full transcript."));
  } else {
    card.appendChild(el("p", null, `Provider: ${data.provider || input.provider || "server default"}`));
    card.appendChild(currentProfileSection(data.current_model_profile));
    card.appendChild(el("p", "review-note", `Requested input: ${input.message ? "supplied for estimation only" : "omitted; history-only request"}. Estimates do not establish the model server's actual context fit.`));
    if (input.message) {
      const details = el("details");
      details.appendChild(el("summary", null, "Proposed input sent for this review"));
      details.appendChild(el("pre", null, input.message));
      card.appendChild(details);
    }
    const config = data.configuration;
    card.appendChild(el("p", null, `Configuration: ${config?.status || "not checked"}`));
    if (config?.reason) card.appendChild(el("p", null, config.reason));
    card.appendChild(listSection("Changed fields", config?.changed_fields));
    card.appendChild(listSection("Unknown identities", config?.unknown_fields));
    const context = data.context;
    card.appendChild(el("p", null, `Context: ${context?.status || "not checked"}`));
    if (context) {
      card.appendChild(el("p", null, `Estimate includes proposed input: ${context.includes_proposed_message == null ? "unknown" : context.includes_proposed_message ? "yes" : "no"}`));
      card.appendChild(el("p", null, `Estimated prompt tokens: ${context.estimated_prompt_tokens ?? "unknown"}`));
      if (context.context_limit_policy === "configured-min-v1") {
        card.appendChild(el("p", null, `Effective context limit: ${context.max_context_tokens ?? "unknown"} (0 = uncapped by harness)`));
        card.appendChild(el("p", "review-note", "Context policy: smaller positive configured limits."));
      } else {
        card.appendChild(el("p", null, `Effective context limit: unknown (policy unrecorded or unsupported). Reported limit: ${context.max_context_tokens ?? "unknown"}`));
      }
      card.appendChild(el("p", null, `Configured harness context cap: ${context.configured_max_context_tokens ?? "unrecorded"} (0 disables only this cap)`));
      card.appendChild(el("p", null, `Ollama requested window: ${context.ollama_num_ctx ?? "unrecorded (unset, not applicable or unknown)"}`));
      card.appendChild(el("p", null, `Configured per-request output cap: ${context.configured_request_output_cap ?? "unrecorded"} (0 disables only this cap)`));
      card.appendChild(el("p", null, `Prompt estimate bytes per token: ${context.prompt_estimate_bytes_per_token ?? "unrecorded"}`));
      card.appendChild(el("p", "review-note", "Manual byte heuristic; higher ratios estimate fewer tokens. Estimated fit is not backend-attested/tokenizer fit or permission to continue."));
      card.appendChild(el("p", null, `Request output allowance: ${context.request_output_allowance ?? (["estimated_fit", "unbounded_context"].includes(context.status) ? "uncapped by harness" : "unavailable")}`));
      if (context.reason) card.appendChild(el("p", null, context.reason));
      card.appendChild(el("p", null, `Saved transcript: ${data.transcript?.status || "not checked"}`));
    }
  }
  card.appendChild(el("p", "review-note", data.scope_note || "Read-only observation; not permission to resume or revert."));
  return card;
}

async function requestSessionReview(kind) {
  if (busy || reviewInFlight || !reviewTarget) return;
  const target = reviewTarget;
  const input = { user_id: userId, message: els.reviewMessage.value };
  if (els.reviewProvider.value) input.provider = els.reviewProvider.value;
  const slot = el("div");
  // Clear just this review's previous observation before requesting a fresh one.
  const host = kind === "tool-progress" ? els.progressResults : els.reviewResults;
  const old = host.querySelector(`[data-review="${kind}"]`);
  if (old) old.remove();
  slot.dataset.review = kind;
  slot.appendChild(el("p", "review-note", `Reading ${kind} review…`));
  host.appendChild(slot);
  reviewInFlight = true;
  setBusy(busy);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const path = kind === "tool-progress"
      ? `/runs/${encodeURIComponent(target.runId)}/tool-progress`
      : `/sessions/${encodeURIComponent(target.sessionId)}/${kind === "check-suggestions" ? kind : kind + "-review"}`;
    const response = kind === "configuration"
      ? await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(input), signal: controller.signal })
      : await fetch(`${path}?${new URLSearchParams({ user_id: userId })}`, { signal: controller.signal });
    if (!response.ok) throw new Error(`Review unavailable (HTTP ${response.status}).`);
    const data = await response.json();
    if (kind === "tool-progress" && data.run_id !== target.runId) {
      throw new Error("Returned journal does not match the selected run");
    }
    if (reviewTarget === target) {
      slot.replaceChildren(kind === "tool-progress"
        ? toolProgressCard(data) : kind === "check-suggestions"
          ? checkSuggestionsCard(data, target.sessionId) : sessionReviewCard(kind, data, input));
    }
  } catch (error) {
    if (kind === "check-suggestions") {
      if (reviewTarget === target) slot.replaceChildren(el("p", "review-note",
        controller.signal.aborted
          ? "Check suggestions wait timed out; the server reader may still be settling. No command ran or configuration changed. No automatic retry; inspect again explicitly."
          : "Check suggestions unavailable or inconsistent. No command ran or configuration changed. No automatic retry; inspect again explicitly."));
      return;
    }
    const reason = controller.signal.aborted
      ? kind === "tool-progress" ? "Tool progress lookup timed out"
        : "Review wait timed out; server comparison may still be settling" : String(error);
    if (reviewTarget === target) slot.replaceChildren(el("p", "review-note", `${kind}: ${reason}. No work was submitted; retry this review explicitly.`));
  } finally {
    clearTimeout(timeout);
    reviewInFlight = false;
    setBusy(busy);
  }
}

// ---- explicit continuation: one POST, exact retained route/input ---------

async function prepareContinuation() {
  if (busy || reviewInFlight || !reviewTarget) return;
  const target = reviewTarget;
  continuationTurnId = null;
  els.continuationAck.checked = false;
  reviewInFlight = true;
  setBusy(busy);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const query = new URLSearchParams({ user_id: userId });
    const response = await fetch(`/sessions/${encodeURIComponent(target.sessionId)}?${query}`, { signal: controller.signal });
    if (!response.ok) throw new Error(`Saved history unavailable (HTTP ${response.status})`);
    const archive = await response.json();
    if (archive.session?.session_id !== target.sessionId) throw new Error("Mismatched saved session");
    const turns = archive.session.turns;
    const last = Array.isArray(turns) ? turns[turns.length - 1] : null;
    if (!last?.turn_id || !last.finished_at) throw new Error("No finalized saved turn to pin");
    if (reviewTarget === target) {
      continuationTurnId = last.turn_id;
      els.continuationTarget.textContent = `Source: ${target.sessionId}; pinned latest turn: ${last.turn_id}. Loading history grants no edit permission.`;
    }
  } catch (_) {
    if (reviewTarget === target) els.continuationTarget.textContent = "Saved turn unavailable. No task was submitted; explicitly reload the saved turn.";
  } finally {
    clearTimeout(timeout);
    reviewInFlight = false;
    setBusy(busy);
  }
}

function submitContinuation() {
  if (busy || reviewInFlight || pendingRequest || pendingReversal || reversalRecoveryBlocked || !reviewTarget || !continuationTurnId ||
      !els.continuationAck.checked || !els.continuationMessage.value.trim()) return;
  let request;
  try {
    request = {
      kind: "continuation", user_id: userId, session_id: reviewTarget.sessionId,
      expected_turn_id: continuationTurnId, message: els.continuationMessage.value.trim(),
      request_id: crypto.randomUUID(), acknowledge_uncaptured_state: true,
    };
    if (els.reviewProvider.value) request.provider = els.reviewProvider.value;
    localStorage.setItem(pendingStorageKey, JSON.stringify(request));
  } catch (_) {
    addErrorMessage("Could not retain exact continuation input. No task was submitted.");
    return;
  }
  pendingRequest = request;
  postContinuation(request);
}

async function postContinuation(request) {
  setBusy(true);
  sessionUsable = false;
  uncertain = true;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const { kind, session_id: source, fork_session_id, ...body } = request;
    const response = await fetch(`/sessions/${encodeURIComponent(source)}/continue`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Continuation submission HTTP ${response.status}`);
    const record = await response.json();
    const actual = record.request;
    if (!record.run_id || !actual?.session_id || actual.session_id === source ||
        actual.user_id !== request.user_id || actual.request_id !== request.request_id ||
        actual.message !== request.message || (actual.provider ?? null) !== (request.provider ?? null) ||
        actual.continuation_source?.session_id !== source ||
        actual.continuation_source?.turn_id !== request.expected_turn_id ||
        actual.continuation_source?.acknowledge_uncaptured_state !== true) {
      throw new Error("Continuation response identity mismatch");
    }
    showRun(record);
    addErrorMessage(`Continuation request ${request.request_id} submitted to fork ${actual.session_id}. Read saved run status; guards and task completion may still be pending. No polling or retry is automatic.`);
  } catch (_) {
    addErrorMessage(`Continuation request ${request.request_id} was not confirmed. Inspect this key using Check run before explicitly retrying the exact retained input. No automatic retry was made.`);
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
}

// ---- explicit one-file reversal: isolated retained route, never auto-run ----

function canonicalReversalPath(path) {
  return typeof path === "string" && path.length > 0 && path.length <= 4096 &&
    !path.startsWith("/") && !path.includes("\0") &&
    path.split("/").every(part => part && part !== "." && part !== "..");
}

function matchingReversal(record, request) {
  const actual = record?.request;
  return !!record?.revert_id && ["pending", "applied", "rejected", "uncertain", "interrupted"].includes(record.status) &&
    actual?.user_id === request.user_id && actual?.run_id === request.run_id &&
    actual?.path === request.path && actual?.request_id === request.request_id;
}

function showReversal(record, request) {
  if (!matchingReversal(record, request)) throw new Error("Reversal response identity mismatch");
  reversalStatus = record.status;
  const card = el("section", "session-review-card");
  card.appendChild(el("h3", null, `Saved reversal: ${record.status}`));
  card.appendChild(el("p", null, `Run: ${request.run_id}; file: ${request.path}; request: ${request.request_id}`));
  card.appendChild(el("p", "review-note", `Journal: ${record.revert_id}; finished: ${record.finished_at || "unrecorded"}`));
  if (record.reason) card.appendChild(el("p", null, record.reason));
  card.appendChild(el("p", "review-note", record.scope_note || "Observed single-file reversal only; inspect the workspace and rerun checks. No automatic reconciliation, restore or continuation."));
  card.appendChild(el("p", null, record.status === "applied"
    ? "Source session retired. Saved checks/diffs remain historical; inspect current files before new work."
    : record.status === "rejected" ? "Saved request rejected; no reversal was applied by this request."
    : "File outcome needs inspection. No automatic execution, retry or reconciliation will occur."));
  els.reversalResults.replaceChildren(card);
}

function submitReversal() {
  if (busy || reviewInFlight || pendingRequest || pendingReversal || reversalRecoveryBlocked || !reviewTarget ||
      !reversalPairAvailable || !els.reversalAck.checked || !canonicalReversalPath(els.reversalPath.value)) return;
  let request;
  try {
    request = { user_id: userId, run_id: reviewTarget.runId,
      path: els.reversalPath.value, request_id: crypto.randomUUID() };
    localStorage.setItem(reversalStorageKey, JSON.stringify(request));
  } catch (_) {
    addErrorMessage("Could not retain exact reversal input. No file action was submitted.");
    return;
  }
  pendingReversal = request;
  reversalStatus = null;
  els.reversalAck.checked = false;
  postReversal(request);
}

async function postReversal(request) {
  if (busy || reviewInFlight || pendingRequest) return;
  setBusy(true);
  sessionUsable = false;
  reversalStatus = null;
  els.reversalResults.replaceChildren(el("p", "review-note", "Submitting the retained single-file request…"));
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const { run_id, ...body } = request;
    const response = await fetch(`/runs/${encodeURIComponent(run_id)}/revert-file`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body), signal: controller.signal,
    });
    // 409/503 can contain a saved rejected/uncertain record; validate identity first.
    showReversal(await response.json(), request);
  } catch (_) {
    els.reversalResults.replaceChildren(el("p", "review-note", "Reversal delivery not confirmed. Inspect the retained key and current file before explicitly retrying this exact request. Server work may still be settling; no automatic retry was made."));
  } finally {
    clearTimeout(timeout);
    setBusy(false);
  }
}

async function inspectReversal() {
  if (busy || reviewInFlight || !pendingReversal) return;
  const request = pendingReversal;
  reviewInFlight = true;
  reversalStatus = null;
  els.reversalResults.replaceChildren(el("p", "review-note", "Reading saved reversal outcome…"));
  setBusy(busy);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const query = new URLSearchParams({ user_id: request.user_id, request_id: request.request_id });
    const response = await fetch(`/file-reverts?${query}`, { signal: controller.signal });
    if (!response.ok) throw new Error("Outcome lookup unavailable");
    const record = await response.json();
    if (record == null) {
      els.reversalResults.replaceChildren(el("p", "review-note", "No saved journal found for this key. This lookup submitted no file action; a delayed request may still arrive. Inspect current files before any explicit retry."));
    } else showReversal(record, request);
  } catch (_) {
    els.reversalResults.replaceChildren(el("p", "review-note", "Saved reversal outcome unavailable or mismatched. Nothing was submitted by this lookup; retry inspection explicitly."));
  } finally {
    clearTimeout(timeout);
    reviewInFlight = false;
    setBusy(busy);
  }
}

function restoreReversal() {
  try {
    const request = JSON.parse(localStorage.getItem(reversalStorageKey));
    if (!request) return;
    if (request.user_id !== userId || typeof request.run_id !== "string" || !request.run_id ||
        typeof request.request_id !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(request.request_id) || !canonicalReversalPath(request.path)) {
      throw new Error("Invalid retained reversal");
    }
    pendingReversal = request;
    sessionUsable = false;
    addErrorMessage("Retained reversal restored for inspection. Reload submitted no file action. Forgetting a key does not reconcile its outcome.");
  } catch (_) {
    reversalRecoveryBlocked = true;
    els.reversalRequest.textContent = "Retained reversal could not be read safely. Inspect saved requests/current files before explicitly forgetting this browser record; new work is blocked.";
    addErrorMessage("Could not restore retained reversal. No file action submitted; inspect saved requests before new work.");
  }
}

// ---- read-only recovery/history ---------------------------------------

function clearPending() {
  pendingRequest = null;
  try { localStorage.removeItem(pendingStorageKey); } catch (_) { /* review remains available */ }
}

async function getRuns(params = {}) {
  const query = new URLSearchParams({ user_id: userId, ...params });
  const resp = await fetch(`/runs?${query}`);
  if (!resp.ok) throw new Error(`Run lookup failed (${resp.status})`);
  return resp.json();
}

function showRun(record) {
  // Viewing history never makes a saved session executable.
  selectReviewTarget(record);
  if (pendingRequest?.request_id === record.request.request_id && pendingRequest.kind === "continuation") {
    pendingRequest.fork_session_id = record.request.session_id;
    try { localStorage.setItem(pendingStorageKey, JSON.stringify(pendingRequest)); }
    catch (_) { addErrorMessage("Could not retain fork identity. Inspect the request key in history after reload."); }
    uncertain = record.status === "running";
  }
  sessionId = record.request.session_id;
  sessionUsable = false;
  els.sessionPill.textContent = `review ${sessionId.slice(0, 8)}`;
  els.sessionPill.title = sessionId;
  els.messages.replaceChildren();
  els.envelope.replaceChildren();
  addUserMessage(record.request.message);
  addErrorMessage(`Saved run ${record.run_id}: ${record.status}. Review only; inspect the workspace before a new session.`);
  if (record.response) {
    addTurn(record.response);
    renderEnvelope(record.response, false);
  } else {
    addErrorMessage(record.error || "No saved final response yet; workspace edits may exist.");
  }
  els.envelope.prepend(listSection("run identity", [
    record.run_id, `lifecycle: ${record.status}`, `request: ${record.request.request_id}`,
    `model: ${record.configured_model || "unrecorded"}`,
  ]));
  if (pendingRequest?.request_id === record.request.request_id && record.status !== "running") {
    clearPending();
    uncertain = false;
    cancelRequested = false;
  }
  setBusy(false);
}

async function inspectRun(runId) {
  if (busy) return;
  setBusy(true);
  try {
    const query = new URLSearchParams({ user_id: userId });
    const resp = await fetch(`/runs/${encodeURIComponent(runId)}?${query}`);
    if (!resp.ok) throw new Error(`Run lookup failed (${resp.status})`);
    showRun(await resp.json());
  } catch (err) {
    addErrorMessage(String(err));
  } finally { setBusy(false); }
}

async function loadHistory() {
  if (busy) return;
  setBusy(true);
  try {
    const records = await getRuns({ limit: 20, offset: historyOffset });
    els.history.replaceChildren();
    for (const record of records) {
      const completion = record.response?.completion_status || "unknown";
      const checks = record.response?.verification_status || "unknown";
      const button = el("button", "history-run", `${record.request.message.slice(0, 80)} — ${record.status}; completion: ${completion}; checks: ${checks}`);
      button.title = record.run_id;
      button.addEventListener("click", () => inspectRun(record.run_id));
      els.history.appendChild(button);
    }
    if (!records.length) els.history.appendChild(el("p", "review-note", "No saved runs on this page."));
    els.nextRuns.dataset.more = records.length === 20 ? "yes" : "";
  } catch (err) { addErrorMessage(String(err)); }
  finally { setBusy(false); }
}

async function checkPending() {
  if (busy || !pendingRequest) return;
  setBusy(true);
  try {
    const records = await getRuns({ request_id: pendingRequest.request_id });
    if (records.length) showRun(records[0]);
    else addErrorMessage("No saved run found. Explicit Retry submission will reuse the retained input and key.");
  } catch (err) { addErrorMessage(String(err)); }
  finally { setBusy(false); }
}

function restorePending() {
  try {
    const request = JSON.parse(localStorage.getItem(pendingStorageKey));
    if (!request) return;
    if (request.user_id !== userId || typeof request.session_id !== "string" ||
        typeof request.message !== "string" || typeof request.request_id !== "string") {
      throw new Error("Invalid retained submission");
    }
    if (request.kind != null && request.kind !== "continuation") throw new Error("Unknown retained route");
    if (request.kind === "continuation" &&
        (typeof request.expected_turn_id !== "string" || !request.expected_turn_id ||
         request.acknowledge_uncaptured_state !== true ||
         (request.fork_session_id != null && typeof request.fork_session_id !== "string"))) {
      throw new Error("Invalid retained continuation");
    }
    pendingRequest = request;
    sessionId = request.fork_session_id || request.session_id;
    uncertain = true;
    sessionUsable = false;
    els.sessionPill.textContent = `review ${sessionId.slice(0, 8)}`;
    addUserMessage(request.message);
    addErrorMessage("Retained submission restored for review. Check run reads saved state; nothing has been submitted on reload.");
  } catch (err) {
    addErrorMessage(`Could not restore retained submission: ${err}`);
  }
}

// ---- events -----------------------------------------------------------

els.prepareContinuation.addEventListener("click", prepareContinuation);
els.submitReversal.addEventListener("click", submitReversal);
els.reversalPath.addEventListener("input", () => setBusy(busy));
els.reversalAck.addEventListener("change", () => setBusy(busy));
els.checkReversal.addEventListener("click", inspectReversal);
els.retryReversal.addEventListener("click", () => {
  if (!pendingReversal || reversalStatus != null) return;
  postReversal({ ...pendingReversal });
});
els.forgetReversal.addEventListener("click", () => {
  if (busy || reviewInFlight) return;
  try { localStorage.removeItem(reversalStorageKey); }
  catch (_) { addErrorMessage("Could not forget retained reversal; inspect its outcome before new work."); return; }
  pendingReversal = null;
  reversalRecoveryBlocked = false;
  reversalStatus = null;
  els.reversalResults.replaceChildren();
  els.reversalAck.checked = false;
  sessionUsable = false;
  addErrorMessage("Local reversal key forgotten. No server work was cancelled or reconciled; inspect files and saved checks before new work.");
  setBusy(busy);
});
els.submitContinuation.addEventListener("click", submitContinuation);
els.continuationMessage.addEventListener("input", () => setBusy(busy));
els.continuationAck.addEventListener("change", () => setBusy(busy));
els.inspectToolProgress.addEventListener("click", () => requestSessionReview("tool-progress"));
els.reviewWorkspace.addEventListener("click", () => requestSessionReview("workspace"));
els.reviewTranscript.addEventListener("click", () => requestSessionReview("transcript"));
els.reviewConfiguration.addEventListener("click", () => requestSessionReview("configuration"));
els.checkSuggestions.addEventListener("click", () => requestSessionReview("check-suggestions"));
els.refreshRuns.addEventListener("click", () => { historyOffset = 0; loadHistory(); });
els.previousRuns.addEventListener("click", () => { historyOffset = Math.max(0, historyOffset - 20); loadHistory(); });
els.nextRuns.addEventListener("click", () => { historyOffset += 20; loadHistory(); });
els.checkRun.addEventListener("click", checkPending);
els.retryRun.addEventListener("click", () => {
  if (busy || !pendingRequest) return;
  // Restored/recovered history remains read-only after the result arrives.
  dispatchRequest(pendingRequest);
});
els.forgetRun.addEventListener("click", () => {
  if (busy) return;
  clearPending();
  uncertain = false;
  sessionUsable = false;
  addErrorMessage("Local pending submission forgotten. Server work was not cancelled; use history to inspect saved runs before new work.");
  setBusy(false);
});
els.cancelTurn.addEventListener("click", cancelTurn);

els.newSession.addEventListener("click", createSession);

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || pendingReversal || reversalRecoveryBlocked || sessionId == null || !sessionUsable) return;
  const message = els.input.value.trim();
  if (!message) return;
  if (sendMessage(message)) els.input.value = "";
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

restorePending();
restoreReversal();
setBusy(false);
