// Owner memory review and explicit reviewed mutations. Shares only panel owner
// identity; no session creation, chat submission, polling or local note storage.
"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const namespace = byId("memory-namespace");
  const workspace = byId("memory-workspace");
  const refresh = byId("refresh-memory");
  const older = byId("older-memory");
  const status = byId("memory-status");
  const results = byId("memory-results");
  byId("memory-owner").textContent = `Owner: ${userId}`;
  let generation = 0;
  let nextCursor = null;
  let mutating = false;
  let detailEpoch = 0;
  let mutationNeedsReview = false;
  const controllers = new Set();

  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text != null) result.textContent = text;
    if (className) result.className = className;
    return result;
  }

  function clearReview() {
    if (mutating) return;
    generation += 1;
    for (const controller of controllers) controller.abort();
    controllers.clear();
    nextCursor = null;
    results.replaceChildren();
    older.disabled = true;
    refresh.disabled = false;
    workspace.disabled = namespace.value !== "workspace";
    status.textContent = "Filters changed. Refresh to inspect this namespace.";
  }

  async function read(path) {
    const controller = new AbortController();
    controllers.add(controller);
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(path, { signal: controller.signal });
      if (!response.ok) {
        if (response.status === 413) throw new Error("legacy-overflow");
        throw new Error("lookup-unavailable");
      }
      return await response.json();
    } finally {
      clearTimeout(timer);
      controllers.delete(controller);
    }
  }

  function validNote(note) {
    return note && Number.isSafeInteger(note.id) && note.id > 0 &&
      typeof note.preview === "string" && new TextEncoder().encode(note.preview).length <= 1024 &&
      typeof note.preview_truncated === "boolean" &&
      Number.isSafeInteger(note.fact_bytes) && note.fact_bytes >= 0 &&
      /^[0-9a-f]{64}$/.test(note.fact_sha256) &&
      (note.workspace_root === null || typeof note.workspace_root === "string");
  }

  function normalizedReplacement(text) {
    // Match the server's str.strip whitespace set; retain internal newlines and FEFF.
    const whitespace = "[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
    return text.replace(new RegExp(`^${whitespace}+|${whitespace}+$`, "gu"), "");
  }

  async function textHash(text) {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
  }

  function mutationControls() {
    namespace.disabled = mutating;
    workspace.disabled = mutating || namespace.value !== "workspace";
    refresh.disabled = mutating;
    older.disabled = mutating || nextCursor === null;
    for (const button of results.querySelectorAll("[data-memory-detail]")) {
      button.disabled = mutating || button.dataset.loading === "true";
    }
    for (const button of results.querySelectorAll("[data-memory-copy]")) {
      button.disabled = mutating || mutationNeedsReview || button.dataset.available !== "true";
    }
    for (const group of results.querySelectorAll(".memory-action")) {
      const ack = group.querySelector('input[type="checkbox"]');
      const destination = group.querySelector("[data-scope-destination]");
      const root = group.querySelector("[data-scope-root]");
      const locked = mutating || mutationNeedsReview;
      for (const input of group.querySelectorAll("input, select, textarea")) input.disabled = locked;
      if (root) root.disabled = locked || destination.value !== "workspace";
      const replacement = group.querySelector("[data-memory-replacement]");
      const bytes = replacement ? new TextEncoder().encode(normalizedReplacement(replacement.value)).length : 0;
      const ready = replacement ? bytes > 0 && bytes <= 4096 :
        !destination || destination.value === "global" ||
        (destination.value === "workspace" && root.value.length > 0);
      if (replacement) group.querySelector("[data-replacement-size]").textContent = `${bytes}/4,096 UTF-8 bytes after trimming`;
      group.querySelector("[data-memory-submit]").disabled = locked || !ack.checked || !ready;
    }
  }

  async function mutateNote(note, version, action, target = null) {
    if (mutating || mutationNeedsReview || version !== generation) return;
    if (action === "correct") {
      target = normalizedReplacement(target);
      if (!target || new TextEncoder().encode(target).length > 4096) return;
    }
    mutating = true;
    detailEpoch += 1;
    for (const controller of controllers) controller.abort();
    controllers.clear();
    for (const button of results.querySelectorAll("[data-memory-detail]")) button.dataset.loading = "false";
    for (const button of results.querySelectorAll("[data-memory-copy]")) button.dataset.available = "false";
    for (const host of results.querySelectorAll(".memory-detail-result")) host.replaceChildren();
    mutationControls();
    const label = action === "delete" ? "Deletion" : action === "correct" ? "Correction" : "Reclassification";
    status.textContent = `${label} of whole note ${note.id} pending…`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const payload = { user_id: userId, expected_fact_sha256: note.fact_sha256,
        expected_workspace_root: note.workspace_root };
      let replacementHash = null;
      if (action === "correct") {
        payload.replacement = target;
        replacementHash = await textHash(target);
      }
      if (action === "scope") {
        payload.workspace_root = target;
      }
      const response = await fetch(`/memory/${note.id}${action === "scope" ? "/scope" : ""}`, {
        method: action === "delete" ? "DELETE" : "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload), signal: controller.signal,
      });
      if (!response.ok) throw new Error("mutation-unconfirmed");
      const data = await response.json();
      if (action === "delete") {
        if (data.status !== "deleted" || data.fact_id !== note.id) throw new Error("delete-mismatch");
      } else if (action === "correct") {
        if (data.status !== "corrected" || !validNote(data.note) || data.note.id !== note.id ||
            data.note.workspace_root !== note.workspace_root || data.note.fact_sha256 !== replacementHash ||
            data.note.fact_bytes !== new TextEncoder().encode(target).length ||
            data.note.source_turn_id !== null || typeof data.note.updated_at !== "string") {
          throw new Error("correction-mismatch");
        }
      } else if (data.status !== "reclassified" || !validNote(data.note) ||
          data.note.id !== note.id || data.note.fact_sha256 !== note.fact_sha256 ||
          data.note.workspace_root !== target) {
        throw new Error("scope-mismatch");
      }
      mutating = false;
      clearReview();
      status.textContent = `${label} of whole note ${note.id} confirmed. Future memory use changed; saved history and sent prompts remain. Refresh to inspect current notes.`;
    } catch (_) {
      mutationNeedsReview = true;
      for (const ack of results.querySelectorAll('.memory-action input[type="checkbox"]')) ack.checked = false;
      status.textContent = `${label} of note ${note.id} was not confirmed; the outcome may be unknown. Refresh and inspect current notes before another mutation. No automatic retry.`;
    } finally {
      clearTimeout(timer);
      mutating = false;
      mutationControls();
    }
  }

  function card(note, version) {
    const item = node("article", null, "memory-note");
    item.appendChild(node("h3", `Note ${note.id}`));
    item.appendChild(node("p", `Workspace: ${note.workspace_root ?? "global"}`));
    item.appendChild(node("p", `${note.fact_bytes} UTF-8 bytes; preview ${note.preview_truncated ? "truncated" : "complete"}`));
    item.appendChild(node("p", `SHA-256: ${note.fact_sha256}`));
    item.appendChild(node("p", `Created: ${note.created_at ?? "unrecorded"}; corrected: ${note.updated_at ?? "unrecorded"}; source turn: ${note.source_turn_id ?? "unrecorded"}`));
    item.appendChild(node("pre", note.preview));
    const button = node("button", "Inspect complete note");
    button.dataset.memoryDetail = "";
    const detail = node("div", null, "memory-detail-result");
    let reviewedText = null;
    button.addEventListener("click", async () => {
      if (mutating || version !== generation || button.disabled) return;
      const epoch = detailEpoch;
      reviewedText = null;
      copy.dataset.available = "false";
      copy.disabled = true;
      button.dataset.loading = "true";
      button.disabled = true;
      detail.replaceChildren(node("p", "Reading complete note…"));
      try {
        const query = new URLSearchParams({ user_id: userId });
        const data = await read(`/memory/${note.id}?${query}`);
        if (version !== generation || epoch !== detailEpoch) return;
        if (!validNote(data.note) || data.note.id !== note.id ||
            typeof data.text !== "string" || new TextEncoder().encode(data.text).length > 4096 ||
            new TextEncoder().encode(data.text).length !== data.note.fact_bytes) {
          throw new Error("invalid-detail");
        }
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(data.text));
        if (version !== generation || epoch !== detailEpoch) return;
        const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
        if (hash !== data.note.fact_sha256) throw new Error("invalid-detail-hash");
        if (data.note.fact_sha256 !== note.fact_sha256 || data.note.workspace_root !== note.workspace_root) {
          detail.replaceChildren(node("p", "Note changed since this preview. Refresh notes before further review."));
        } else {
          detail.replaceChildren(node("pre", data.text), node("p", data.scope_note, "review-note"));
          reviewedText = data.text;
          copy.dataset.available = "true";
          mutationControls();
        }
      } catch (error) {
        if (version === generation && epoch === detailEpoch) {
          detail.replaceChildren(node("p", error.message === "legacy-overflow"
            ? "Legacy note exceeds the complete-text limit. Use the bounded preview above."
            : "Complete note unavailable. No work submitted; retry this lookup explicitly."));
        }
      } finally {
        if (version === generation && epoch === detailEpoch) {
          button.dataset.loading = "false";
          button.disabled = mutating;
        }
      }
    });
    const remove = node("details", null, "memory-delete memory-action");
    remove.appendChild(node("summary", "Delete this whole note"));
    const label = node("label");
    const ack = node("input");
    ack.type = "checkbox";
    label.append(ack, document.createTextNode(` Delete whole note ${note.id}, including text omitted from the preview, from future recall. Saved history remains.`));
    const submit = node("button", "Delete reviewed note");
    submit.dataset.memorySubmit = "";
    submit.disabled = true;
    ack.addEventListener("change", mutationControls);
    submit.addEventListener("click", () => { if (ack.checked) mutateNote(note, version, "delete"); });
    remove.append(label, submit);
    const scope = node("details", null, "memory-scope memory-action");
    scope.appendChild(node("summary", "Change this note's scope"));
    const destinationLabel = node("label", "Destination namespace");
    const destination = node("select");
    destination.dataset.scopeDestination = "";
    for (const [value, text] of [["", "Choose a destination"], ["global", "Global preference"], ["workspace", "Exact workspace convention"]]) {
      const option = node("option", text); option.value = value; destination.appendChild(option);
    }
    destinationLabel.appendChild(destination);
    const rootLabel = node("label", "Destination workspace path");
    const root = node("input"); root.type = "text"; root.maxLength = 4096;
    root.dataset.scopeRoot = ""; root.placeholder = "/absolute/project"; root.disabled = true;
    rootLabel.appendChild(root);
    const scopeAckLabel = node("label");
    const scopeAck = node("input"); scopeAck.type = "checkbox";
    scopeAckLabel.append(scopeAck, document.createTextNode(` Reclassify whole note ${note.id} from the displayed scope. Global notes can appear in all my workspaces; text and saved history stay unchanged.`));
    const move = node("button", "Reclassify reviewed note"); move.disabled = true;
    move.dataset.memorySubmit = "";
    for (const control of [destination, root, scopeAck]) control.addEventListener("input", mutationControls);
    destination.addEventListener("change", mutationControls);
    scopeAck.addEventListener("change", mutationControls);
    move.addEventListener("click", () => {
      if (!scopeAck.checked || move.disabled) return;
      mutateNote(note, version, "scope", destination.value === "global" ? null : root.value);
    });
    scope.append(destinationLabel, rootLabel, scopeAckLabel, move);
    const correction = node("details", null, "memory-correct memory-action");
    correction.appendChild(node("summary", "Correct this whole note"));
    correction.appendChild(node("p", "Enter the complete replacement, including text omitted from the preview. Outer whitespace is trimmed; internal newlines stay. Drafts are not saved on reload.", "review-note"));
    const copy = node("button", "Use inspected complete text");
    copy.dataset.memoryCopy = ""; copy.dataset.available = "false"; copy.disabled = true;
    const replacementLabel = node("label", "Whole-note replacement");
    const replacement = node("textarea"); replacement.rows = 4; replacement.maxLength = 8192;
    replacement.dataset.memoryReplacement = ""; replacement.autocomplete = "off";
    replacementLabel.appendChild(replacement);
    const size = node("p", "0/4,096 UTF-8 bytes after trimming", "review-note");
    size.dataset.replacementSize = "";
    const correctAckLabel = node("label");
    const correctAck = node("input"); correctAck.type = "checkbox";
    correctAckLabel.append(correctAck, document.createTextNode(` Replace whole note ${note.id}. Original source-turn attribution will be cleared; saved history stays unchanged.`));
    const save = node("button", "Correct reviewed note"); save.dataset.memorySubmit = ""; save.disabled = true;
    copy.addEventListener("click", () => {
      if (copy.disabled || reviewedText === null) return;
      replacement.value = reviewedText;
      correctAck.checked = false;
      mutationControls();
    });
    replacement.addEventListener("input", mutationControls);
    correctAck.addEventListener("change", mutationControls);
    save.addEventListener("click", () => {
      if (!correctAck.checked || save.disabled) return;
      mutateNote(note, version, "correct", replacement.value);
    });
    correction.append(copy, replacementLabel, size, correctAckLabel, save);
    item.append(button, detail, remove, scope, correction);
    return item;
  }

  async function load(beforeId = null) {
    if (mutating) return;
    clearReview();
    const version = generation;
    const scope = namespace.value;
    const root = workspace.value;
    if (scope === "workspace" && !root) {
      status.textContent = "Supply a canonical absolute memory workspace path.";
      return;
    }
    refresh.disabled = true;
    status.textContent = "Reading notes…";
    const query = new URLSearchParams({ user_id: userId, limit: "20", namespace: scope });
    if (scope === "workspace") query.set("workspace_root", root);
    if (beforeId != null) query.set("before_id", String(beforeId));
    try {
      const page = await read(`/memory?${query}`);
      if (version !== generation) return;
      if (!Array.isArray(page.notes) || page.notes.length > 20 || !page.notes.every(validNote) ||
          !(page.next_before_id === null || (Number.isSafeInteger(page.next_before_id) && page.next_before_id > 0)) ||
          typeof page.scope_note !== "string" ||
          page.notes.some((note) => (scope === "global" && note.workspace_root !== null) ||
            (scope === "workspace" && note.workspace_root !== root))) {
        throw new Error("invalid-page");
      }
      mutationNeedsReview = false;
      for (const note of page.notes) results.appendChild(card(note, version));
      nextCursor = page.next_before_id;
      older.disabled = nextCursor === null;
      status.textContent = `${page.notes.length ? "Current page loaded." : "No notes in this namespace."} ${page.scope_note}`;
    } catch (_) {
      if (version === generation) status.textContent = "Memory lookup unavailable. Check the namespace/path and retry explicitly. No work submitted.";
    } finally {
      if (version === generation) refresh.disabled = false;
    }
  }

  namespace.addEventListener("change", clearReview);
  workspace.addEventListener("input", clearReview);
  refresh.addEventListener("click", () => load());
  older.addEventListener("click", () => { if (nextCursor !== null) load(nextCursor); });
})();
