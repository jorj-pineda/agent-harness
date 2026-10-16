// Owner memory review and explicit hash-guarded deletion. Shares only panel owner
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
  let deleting = false;
  let deletionNeedsReview = false;
  const controllers = new Set();

  function node(tag, text, className) {
    const result = document.createElement(tag);
    if (text != null) result.textContent = text;
    if (className) result.className = className;
    return result;
  }

  function clearReview() {
    if (deleting) return;
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

  function deletionControls() {
    namespace.disabled = deleting;
    workspace.disabled = deleting || namespace.value !== "workspace";
    refresh.disabled = deleting;
    older.disabled = deleting || nextCursor === null;
    for (const group of results.querySelectorAll(".memory-delete")) {
      const ack = group.querySelector("input");
      ack.disabled = deleting || deletionNeedsReview;
      group.querySelector("button").disabled = deleting || deletionNeedsReview || !ack.checked;
    }
  }

  async function deleteNote(note, version) {
    if (deleting || deletionNeedsReview || version !== generation) return;
    deleting = true;
    deletionControls();
    status.textContent = `Deleting whole note ${note.id}…`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch(`/memory/${note.id}`, {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, expected_fact_sha256: note.fact_sha256 }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error("delete-unconfirmed");
      const data = await response.json();
      if (data.status !== "deleted" || data.fact_id !== note.id) throw new Error("delete-mismatch");
      deleting = false;
      clearReview();
      status.textContent = `Deleted whole note ${note.id} from future memory use. Saved history and already sent prompts remain. Refresh to inspect current notes.`;
    } catch (_) {
      deletionNeedsReview = true;
      for (const ack of results.querySelectorAll(".memory-delete input")) ack.checked = false;
      status.textContent = `Deletion of note ${note.id} was not confirmed; the outcome may be unknown. Refresh and inspect current notes before another attempt. No automatic retry.`;
    } finally {
      clearTimeout(timer);
      deleting = false;
      deletionControls();
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
    const detail = node("div");
    button.addEventListener("click", async () => {
      if (version !== generation || button.disabled) return;
      button.disabled = true;
      detail.replaceChildren(node("p", "Reading complete note…"));
      try {
        const query = new URLSearchParams({ user_id: userId });
        const data = await read(`/memory/${note.id}?${query}`);
        if (version !== generation) return;
        if (!validNote(data.note) || data.note.id !== note.id ||
            typeof data.text !== "string" || new TextEncoder().encode(data.text).length > 4096 ||
            new TextEncoder().encode(data.text).length !== data.note.fact_bytes) {
          throw new Error("invalid-detail");
        }
        const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(data.text));
        if (version !== generation) return;
        const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
        if (hash !== data.note.fact_sha256) throw new Error("invalid-detail-hash");
        if (data.note.fact_sha256 !== note.fact_sha256 || data.note.workspace_root !== note.workspace_root) {
          detail.replaceChildren(node("p", "Note changed since this preview. Refresh notes before further review."));
        } else {
          detail.replaceChildren(node("pre", data.text), node("p", data.scope_note, "review-note"));
        }
      } catch (error) {
        if (version === generation) {
          detail.replaceChildren(node("p", error.message === "legacy-overflow"
            ? "Legacy note exceeds the complete-text limit. Use the bounded preview above."
            : "Complete note unavailable. No work submitted; retry this lookup explicitly."));
        }
      } finally {
        if (version === generation) button.disabled = false;
      }
    });
    const remove = node("details", null, "memory-delete");
    remove.appendChild(node("summary", "Delete this whole note"));
    const label = node("label");
    const ack = node("input");
    ack.type = "checkbox";
    label.append(ack, document.createTextNode(` Delete whole note ${note.id}, including text omitted from the preview, from future recall. Saved history remains.`));
    const submit = node("button", "Delete reviewed note");
    submit.disabled = true;
    ack.addEventListener("change", deletionControls);
    submit.addEventListener("click", () => { if (ack.checked) deleteNote(note, version); });
    remove.append(label, submit);
    item.append(button, detail, remove);
    return item;
  }

  async function load(beforeId = null) {
    if (deleting) return;
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
      deletionNeedsReview = false;
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
