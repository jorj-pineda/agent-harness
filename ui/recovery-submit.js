// One explicit recovery POST with retained exact intent and manual outcome recovery.
"use strict";
(() => {
  const ids = ["recovery-ack-state", "recovery-ack-context", "recovery-ack-budget", "submit-recovery",
    "recovery-submit-status", "recovery-retained", "recovery-retained-identity", "check-recovery",
    "retry-recovery", "cancel-recovery", "forget-recovery", "recovery-submit-results"];
  const ui = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
  const acks = ["recovery-ack-state", "recovery-ack-context", "recovery-ack-budget"];
  const key = "panel_pending_recovery_v1";
  let pending = null, blocked = false, operating = false, lastRecord = null, cancelRequested = false;
  const bytes = text => new TextEncoder().encode(text).length;
  const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
  const id = value => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value);
  const digest = value => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
  const approvalFields = ["expected_record_sha256", "spans", "message", "expected_messages_sha256",
    "expected_archive_sha256", "expected_tool_specs_sha256", "expected_request_output_allowance",
    "acknowledge_uncaptured_state", "acknowledge_omitted_context", "acknowledge_new_turn_budget"];

  function node(tag, text) { const value = document.createElement(tag); if (text != null) value.textContent = text; return value; }
  function status(text) { ui["recovery-submit-status"].textContent = text; }
  function approval(body) { return Object.fromEntries(approvalFields.map(field => [field, body[field]])); }
  function otherWork() { return busy || reviewInFlight || pendingRequest != null || pendingReversal != null || reversalRecoveryBlocked; }
  function eligible(candidate) {
    return candidate?.draft.configuration.status === "matching" && candidate.draft.context.status === "estimated_fit" &&
      Number.isSafeInteger(candidate.draft.context.request_output_allowance) && candidate.draft.context.request_output_allowance > 0 &&
      (candidate.draft.provider !== "ollama" || Number.isSafeInteger(candidate.draft.context.ollama_num_ctx));
  }
  function invalidate() { for (const field of acks) ui[field].checked = false; refresh(); }
  function refresh() {
    const locked = operating || otherWork();
    ui["submit-recovery"].disabled = locked || !!pending || blocked ||
      !eligible(globalThis.recoveryDraftPanel?.reviewed()) || !acks.every(field => ui[field].checked);
    for (const field of acks) ui[field].disabled = locked || !!pending || blocked;
    ui["recovery-retained"].hidden = !pending && !blocked;
    ui["check-recovery"].disabled = locked || !pending || blocked;
    ui["retry-recovery"].disabled = locked || !pending || blocked || (lastRecord && lastRecord.status !== "running");
    ui["cancel-recovery"].disabled = locked || blocked || lastRecord?.status !== "running" || cancelRequested;
    ui["forget-recovery"].disabled = locked;
    ui["recovery-retained-identity"].textContent = pending
      ? `Retained source: ${pending.source_run_id}; request: ${pending.body.request_id}. Exact plaintext proposal and pins are saved locally; new edits are blocked until released.`
      : "Retained recovery cannot be read safely. New edits are blocked; inspect server runs before forgetting this local record.";
  }

  function validateIntent(value) {
    if (!object(value) || Object.keys(value).sort().join() !== "body,schema_version,source_run_id,source_session_id" ||
        value.schema_version !== 1 || !id(value.source_run_id) || !id(value.source_session_id) || !object(value.body)) throw new Error("invalid-intent");
    const body = value.body;
    if (Object.keys(body).sort().join() !== [...approvalFields, "user_id", "request_id", "provider"].sort().join() ||
        body.user_id !== userId || !id(body.request_id) || (body.provider !== null && (typeof body.provider !== "string" || !body.provider)) ||
        typeof body.message !== "string" || !body.message.trim() || bytes(body.message) > 4096 ||
        !Number.isSafeInteger(body.expected_request_output_allowance) || body.expected_request_output_allowance < 1 || body.expected_request_output_allowance > 1000000 ||
        !["expected_record_sha256", "expected_archive_sha256", "expected_messages_sha256", "expected_tool_specs_sha256"].every(k => digest(body[k])) ||
        !["acknowledge_uncaptured_state", "acknowledge_omitted_context", "acknowledge_new_turn_budget"].every(k => body[k] === true) ||
        !Array.isArray(body.spans) || body.spans.length > 4) throw new Error("invalid-body");
    for (let i = 0; i < body.spans.length; i++) {
      const span = body.spans[i];
      if (!object(span) || Object.keys(span).sort().join() !== "end_line,expected_sha256,path,start_line" ||
          typeof span.path !== "string" || !span.path || span.path.length > 1024 || !digest(span.expected_sha256) ||
          !Number.isSafeInteger(span.start_line) || !Number.isSafeInteger(span.end_line) ||
          span.start_line < 1 || span.end_line > 1048576 || span.end_line < span.start_line || span.end_line - span.start_line + 1 > 200) throw new Error("invalid-span");
      for (const old of body.spans.slice(0, i)) if (old.path === span.path &&
          (old.expected_sha256 !== span.expected_sha256 || Math.max(old.start_line, span.start_line) <= Math.min(old.end_line, span.end_line))) throw new Error("conflicting-span");
    }
    if (bytes(JSON.stringify(value)) > 65536) throw new Error("oversized-intent");
    return value;
  }

  async function hash(text, algorithm = "SHA-256", prefix = []) {
    const result = await crypto.subtle.digest(algorithm, new Uint8Array([...prefix, ...new TextEncoder().encode(text)]));
    return [...new Uint8Array(result)].map(byte => byte.toString(16).padStart(2, "0")).join("");
  }
  async function forkId(intent) {
    // Match Python's UUIDv5 URL namespace and json.dumps(list) default ASCII/separators.
    const quote = text => JSON.stringify(text).replace(/[\u007f-\uffff]/g, c => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
    const name = "[" + ["agent-harness-recovery", userId, intent.source_run_id, intent.body.request_id].map(quote).join(", ") + "]";
    const namespace = "6ba7b8119dad11d180b400c04fd430c8".match(/../g).map(byte => parseInt(byte, 16));
    const hex = (await hash(name, "SHA-1", namespace)).slice(0, 32).split("");
    hex[12] = "5"; hex[16] = ((parseInt(hex[16], 16) & 3) | 8).toString(16);
    return hex.join("");
  }
  async function validateRecord(record, intent) {
    const body = intent.body, request = record?.request, origin = request?.recovery_source;
    const expected = JSON.stringify(approval(body));
    if (!object(record) || !id(record.run_id) || record.run_id === intent.source_run_id ||
        !["running", "finished", "cancelled", "failed", "interrupted"].includes(record.status) || !object(request) || !object(origin) ||
        request.user_id !== userId || request.request_id !== body.request_id || request.provider !== body.provider ||
        request.message !== body.message || request.session_id !== await forkId(intent) || request.session_id === intent.source_session_id ||
        origin.schema_version !== 1 || origin.draft_version !== "recovery-draft-v1" || origin.budget_scope !== "separate-configured-turn" ||
        origin.source_run_id !== intent.source_run_id || origin.source_session_id !== intent.source_session_id || !id(origin.source_turn_id) ||
        origin.source_record_sha256 !== body.expected_record_sha256 || origin.source_archive_sha256 !== body.expected_archive_sha256 ||
        origin.messages_sha256 !== body.expected_messages_sha256 || origin.tool_specs_sha256 !== body.expected_tool_specs_sha256 ||
        origin.request_output_allowance !== body.expected_request_output_allowance || origin.approval_json !== expected ||
        origin.approval_sha256 !== await hash(expected) ||
        request.continuation_source?.session_id !== intent.source_session_id || request.continuation_source?.turn_id !== origin.source_turn_id ||
        request.continuation_source?.acknowledge_uncaptured_state !== true) throw new Error("mismatched-record");
    const response = record.response;
    if ((["finished", "cancelled"].includes(record.status) && !object(response)) ||
        (response !== null && (!object(response) || response.run_id !== record.run_id ||
          !["completed", "incomplete", "blocked", "budget_exhausted", "cancelled"].includes(response.completion_status) ||
          !["not_run", "passed", "failed", "unavailable", "stale"].includes(response.verification_status)))) throw new Error("invalid-final");
    return record;
  }

  function display(record) {
    lastRecord = record;
    const card = node("section");
    card.appendChild(node("p", `Completion: ${record.response?.completion_status ?? "unknown"}; checks: ${record.response?.verification_status ?? "unknown"}; lifecycle: ${record.status}`));
    card.appendChild(node("p", `Run: ${record.run_id}; fork: ${record.request.session_id}; request: ${record.request.request_id}`));
    card.appendChild(node("p", "Submission is not passed admission or task acceptance. Separate configured turn budget; source evidence is unchanged. Inspect partial work and unknown usage before new work."));
    const details = node("details"); details.appendChild(node("summary", "Saved run and immutable recovery provenance"));
    details.appendChild(node("pre", JSON.stringify(record, null, 2))); card.appendChild(details);
    const open = node("button", "Open saved run review"); open.addEventListener("click", () => inspectRun(record.run_id)); card.appendChild(open);
    ui["recovery-submit-results"].replaceChildren(card);
  }
  async function read(path, options, signal) {
    const response = await fetch(path, { ...options, signal });
    if (!response.ok) throw new Error("unconfirmed-response");
    const text = await response.text();
    if (bytes(text) > 4 * 1024 * 1024) throw new Error("oversized-response");
    return JSON.parse(text);
  }
  async function action(kind) {
    if (operating || otherWork() || blocked || !pending || (kind === "cancel" &&
        (lastRecord?.status !== "running" || cancelRequested))) return;
    const intent = pending;
    try {
      if (localStorage.getItem(key) !== JSON.stringify(intent)) throw new Error("retained-intent-changed");
    } catch (_) { blocked = true; status("Retained input changed or cannot be read. No request sent; inspect saved runs before forgetting it."); setBusy(busy); return; }
    operating = true; setBusy(true); sessionUsable = false;
    const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const post = body => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (kind === "cancel") {
        const result = await read(`/sessions/${encodeURIComponent(lastRecord.request.session_id)}/cancel`, post({ user_id: userId }), controller.signal);
        if (result.status !== "requested") throw new Error("unconfirmed-cancel");
        cancelRequested = true; status("Cancellation requested; owned work may still be settling. Check saved recovery explicitly for final evidence.");
      } else {
        const data = kind === "post"
          ? await read(`/runs/${encodeURIComponent(intent.source_run_id)}/recover`, post(intent.body), controller.signal)
          : await read(`/runs?${new URLSearchParams({ user_id: userId, request_id: intent.body.request_id })}`, {}, controller.signal);
        if (kind === "check" && (!Array.isArray(data) || data.length > 1)) throw new Error("ambiguous-key");
        if (kind === "check" && data.length === 0) { status("No saved run found for this key. Inspect before explicitly retrying the exact retained request; nothing was replayed."); return; }
        const record = await validateRecord(kind === "check" ? data[0] : data, intent);
        if (pending !== intent) return;
        display(record);
        status("Saved recovery identity confirmed. Check status and current files before explicitly releasing this local intent. No automatic polling or retry.");
      }
    } catch (_) {
      status("Recovery delivery/outcome is unconfirmed or the response is inconsistent. Server work may exist. Check this retained key before an explicit exact retry; no automatic retry.");
    } finally { clearTimeout(timer); operating = false; setBusy(false); }
  }

  function submit() {
    const candidate = globalThis.recoveryDraftPanel?.reviewed();
    if (operating || otherWork() || blocked || pending || !eligible(candidate) || !acks.every(field => ui[field].checked)) return;
    const data = candidate.draft;
    try {
      const intent = { schema_version: 1, source_run_id: candidate.source_run_id, source_session_id: candidate.source_session_id,
        body: { user_id: userId, request_id: crypto.randomUUID(), provider: candidate.input.provider ?? null,
          expected_record_sha256: data.source_record_sha256, spans: candidate.input.spans, message: candidate.input.message,
          expected_messages_sha256: data.messages_sha256, expected_archive_sha256: data.source_archive_sha256,
          expected_tool_specs_sha256: data.tool_specs_sha256, expected_request_output_allowance: data.context.request_output_allowance,
          acknowledge_uncaptured_state: true, acknowledge_omitted_context: true, acknowledge_new_turn_budget: true } };
      validateIntent(intent);
      const encoded = JSON.stringify(intent); localStorage.setItem(key, encoded);
      if (localStorage.getItem(key) !== encoded) throw new Error("intent-not-retained");
      pending = JSON.parse(encoded); lastRecord = null; cancelRequested = false;
      globalThis.recoveryDraftPanel?.invalidate?.(); invalidate();
    } catch (_) { blocked = true; status("Exact recovery input could not be retained safely. No submission sent; inspect local/server state before releasing this record."); setBusy(busy); return; }
    return action("post");
  }
  function forget() {
    if (operating || busy || reviewInFlight) return;
    try { localStorage.removeItem(key); if (localStorage.getItem(key) !== null) throw new Error("intent-still-retained"); }
    catch (_) { status("Could not release local recovery intent; new edits remain blocked."); return; }
    pending = null; blocked = false; lastRecord = null; cancelRequested = false; sessionUsable = false;
    globalThis.recoveryDraftPanel?.invalidate?.(); invalidate();
    status("Local recovery intent forgotten. No server work cancelled or files reconciled. Inspect saved runs/current files before creating a new session."); setBusy(busy);
  }
  ui["submit-recovery"].addEventListener("click", submit);
  ui["check-recovery"].addEventListener("click", () => action("check"));
  ui["retry-recovery"].addEventListener("click", () => { if (!lastRecord || lastRecord.status === "running") return action("post"); });
  ui["cancel-recovery"].addEventListener("click", () => action("cancel"));
  ui["forget-recovery"].addEventListener("click", forget);
  for (const field of acks) ui[field].addEventListener("change", refresh);
  globalThis.recoverySubmitPanel = { invalidate, refresh, blocksWork: () => !!pending || blocked };
  try {
    const raw = localStorage.getItem(key);
    if (raw !== null) {
      if (bytes(raw) > 65536) throw new Error("oversized-intent");
      const parsed = JSON.parse(raw);
      if (JSON.stringify(parsed) !== raw) throw new Error("noncanonical-intent");
      pending = validateIntent(parsed); sessionUsable = false;
      status("Retained recovery restored for review only. Reload sent nothing. Check the saved key before explicit retry or release.");
    }
  } catch (_) { blocked = true; sessionUsable = false; status("Retained recovery unavailable or inconsistent; new edits are blocked. Inspect server runs before forgetting this local record."); }
  invalidate(); setBusy(busy);
})();
