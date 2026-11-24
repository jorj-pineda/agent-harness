// Actual submission scripts and real loopback fetch; simulated DOM/storage, not a browser.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { webcrypto } = require('node:crypto');
const [root, packetPath, statePath, action] = process.argv.slice(2);
const data = JSON.parse(fs.readFileSync(packetPath, 'utf8'));
const storage = new Map(fs.existsSync(statePath)
  ? JSON.parse(fs.readFileSync(statePath, 'utf8')) : [['panel_user_id', 'dev']]);
const key = 'panel_pending_recovery_v1';
const reversalKey = 'panel_pending_reversal';
class Node {
  constructor(tag) {
    this.tag = tag; this.children = []; this.events = {}; this.value = ''; this._text = '';
    this.checked = false; this.dataset = {}; this.classList = { add() {}, remove() {} };
  }
  appendChild(child) { this.children.push(child); child.parent = this; return child; }
  prepend(child) { this.children.unshift(child); }
  replaceChildren(...children) { this.children = []; children.forEach(child => this.appendChild(child)); }
  remove() { if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  addEventListener(name, fn) { this.events[name] = fn; }
  async fire(name) { return this.events[name]?.({ preventDefault() {} }); }
  querySelector() { return null; }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [
      ...(selector === 'button' && child.tag === 'button' ? [child] : []),
      ...child.querySelectorAll(selector),
    ]);
  }
  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(child => child.textContent).join('\n'); }
  set innerHTML(_) { throw Error('unsafe HTML sink'); }
  focus() {}
}
const ids = new Map(), calls = [];
let droppedReversalStatus;
const get = id => {
  if (!ids.has(id)) ids.set(id, new Node('div'));
  return ids.get(id);
};
const context = vm.createContext({
  document: { getElementById: get, createElement: tag => new Node(tag) },
  localStorage: {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key),
  },
  crypto: { subtle: webcrypto.subtle, randomUUID: () => '11111111-1111-4111-8111-111111111111' },
  TextEncoder, AbortController, URLSearchParams, setTimeout, clearTimeout, window: {},
  fetch: async (path, options = {}) => {
    if (action.startsWith('reverse')) {
      if (path.endsWith('/revert-file')) {
        const { run_id, ...body } = JSON.parse(storage.get(reversalKey));
        assert.equal(path, `/runs/${run_id}/revert-file`);
        assert.deepEqual(JSON.parse(options.body), body);
      }
    } else if (action !== 'review' && !path.includes('/recovery-review?') && !path.endsWith('/recovery-draft')) {
      assert(storage.has(key), 'exact intent must be retained before execution or outcome lookup');
    }
    const url = new URL(path, data.url);
    assert.equal(url.origin, new URL(data.url).origin, 'only owned loopback API allowed');
    if (options.method === 'POST' && path.endsWith('/recover')) {
      assert.deepEqual(JSON.parse(options.body), JSON.parse(storage.get(key)).body);
    }
    calls.push({ path, method: options.method ?? 'GET', body: options.body ?? null });
    const response = await fetch(url, options);
    if (data.scenario === 'lost_response' && action === 'submit' && path.endsWith('/recover')) {
      await response.text(); // Server acknowledged; hide only this delivery from the client.
      throw Error('fixture response delivery loss');
    }
    if (data.scenario === 'lost_reversal_response' && action === 'reverse' && path.endsWith('/revert-file')) {
      await response.text();
      droppedReversalStatus = response.status;
      throw Error('fixture reversal response delivery loss');
    }
    return response;
  },
});
for (const name of ['app.js', 'recovery.js', 'recovery-submit.js']) {
  vm.runInContext(fs.readFileSync(root + '/' + name, 'utf8'), context);
}
(async () => {
  assert.equal(calls.length, 0, 'loading/restoring scripts must send nothing');
  if (action === 'reverse-check') {
    const retained = storage.get(reversalKey);
    const { request_id, user_id } = JSON.parse(retained);
    assert(get('new-session').disabled);
    assert(!get('check-reversal').disabled);
    await get('check-reversal').fire('click');
    assert.deepEqual(calls, [{
      path: `/file-reverts?${new URLSearchParams({ user_id, request_id })}`,
      method: 'GET', body: null,
    }]);
    const status = data.scenario === 'stale_reversal' ? 'rejected' : 'applied';
    assert.match(get('reversal-results').textContent, new RegExp(`Saved reversal: ${status}`));
    assert(get('retry-reversal').disabled);
    assert.equal(storage.get(reversalKey), retained);
    console.log(JSON.stringify({ calls, retained }));
    return;
  }
  if (action === 'reverse') {
    await vm.runInContext(`inspectRun(${JSON.stringify(data.source.run_id)})`, context);
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'GET');
    assert.equal(calls[0].path, `/runs/${data.source.run_id}?user_id=dev`);
    get('reversal-path').value = 'a.py';
    await get('reversal-path').fire('input');
    assert(get('submit-reversal').disabled);
    await get('submit-reversal').fire('click');
    assert.equal(calls.length, 1, 'unacknowledged submission sends nothing');
    assert(!storage.has(reversalKey));
    get('reversal-ack').checked = true;
    await get('reversal-ack').fire('change');
    assert(!get('submit-reversal').disabled);
    assert.equal(calls.length, 1, 'acknowledgement alone sends nothing');
    assert(!storage.has(reversalKey));
    if (data.scenario === 'stale_reversal') {
      fs.writeFileSync(data.target, 'later user edit to the reversal target\n');
      fs.chmodSync(data.target, 0o600);
    }
    await get('submit-reversal').fire('click');
    const deadline = Date.now() + 5000;
    while (vm.runInContext('busy', context) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.equal(vm.runInContext('busy', context), false);
    assert.equal(calls.length, 2);
    assert.equal(calls[1].method, 'POST');
    const result = get('reversal-results').textContent;
    if (data.scenario === 'stale_reversal') {
      assert.match(result, /Saved reversal: rejected/);
      assert.match(result, /no reversal was applied/);
    } else if (data.scenario === 'lost_reversal_response') {
      assert.equal(droppedReversalStatus, 200);
      assert.match(result, /Reversal delivery not confirmed/);
      assert(!get('check-reversal').disabled);
      assert(!get('retry-reversal').disabled);
    } else {
      assert.match(result, /Saved reversal: applied/);
    }
    assert(get('submit-reversal').disabled);
    assert(get('new-session').disabled);
    fs.writeFileSync(statePath, JSON.stringify([...storage]));
    console.log(JSON.stringify({ calls, retained: storage.get(reversalKey) }));
    return;
  }
  if (action === 'review') {
    context.recoveryDraftPanel.select(data.source);
    assert.equal(calls.length, 0, 'selection alone sends nothing');
    assert(get('preview-recovery').disabled);
    await get('load-recovery').fire('click');
    assert.deepEqual(calls, [{
      path: `/runs/${data.source.run_id}/recovery-review?user_id=dev`, method: 'GET', body: null,
    }]);
    const evidence = get('recovery-evidence').textContent;
    assert.match(evidence, /Source completion: incomplete; checks: not_run/);
    assert(get('submit-recovery').disabled);
    assert(!storage.has(key));
    console.log(JSON.stringify({ calls, text: evidence }));
    return;
  }
  if (action === 'submit') {
    context.recoveryDraftPanel.select(data.source);
    await get('load-recovery').fire('click');
    get('recovery-message').value = data.proposal.message;
    await get('recovery-message').fire('input');
    for (const span of data.proposal.spans) {
      await get('add-recovery-range').fire('click');
      const row = get('recovery-ranges').children.at(-1);
      const values = [span.path, span.expected_sha256, span.start_line, span.end_line];
      for (const [index, label] of row.children.filter(child => child.tag === 'label').entries()) {
        label.children[0].value = String(values[index]);
        await label.children[0].fire('input');
      }
    }
    assert(!get('preview-recovery').disabled);
    await get('preview-recovery').fire('click');
    assert.equal(calls.length, 2);
    assert.match(get('recovery-draft-results').textContent, /Context: estimated_fit; configuration: matching/);
    let reviewed = context.recoveryDraftPanel.reviewed();
    assert(reviewed);
    assert(!storage.has(key));
    assert(get('submit-recovery').disabled);
    const acknowledgements = ['state', 'context', 'budget'].map(name => get('recovery-ack-' + name));
    async function acknowledge() {
      for (const checkbox of acknowledgements) {
        checkbox.checked = true;
        await checkbox.fire('change');
      }
      assert(!get('submit-recovery').disabled);
    }
    await acknowledge();
    if (data.scenario === 'edited_proposal') {
      const previousHash = reviewed.draft.messages_sha256;
      get('recovery-message').value += ' Keep the staged user file intact.';
      await get('recovery-message').fire('input');
      assert.equal(context.recoveryDraftPanel.reviewed(), null);
      assert.equal(get('recovery-draft-results').textContent, '');
      assert(acknowledgements.every(checkbox => !checkbox.checked));
      assert(get('submit-recovery').disabled);
      await get('submit-recovery').fire('click');
      assert.equal(calls.length, 2);
      assert(!storage.has(key));
      await get('preview-recovery').fire('click');
      assert.equal(calls.length, 3);
      reviewed = context.recoveryDraftPanel.reviewed();
      assert(reviewed);
      assert.notEqual(reviewed.draft.messages_sha256, previousHash);
      assert(get('submit-recovery').disabled);
      assert(acknowledgements.every(checkbox => !checkbox.checked));
      await acknowledge();
    }
    await get('submit-recovery').fire('click');
    assert.equal(calls.length, data.scenario === 'edited_proposal' ? 4 : 3);
    const dispatch = calls.at(-1);
    assert.equal(dispatch.method, 'POST');
    const sent = JSON.parse(dispatch.body);
    assert.equal(sent.message, reviewed.input.message);
    assert.equal(sent.expected_messages_sha256, reviewed.draft.messages_sha256);
    assert.equal(sent.expected_tool_specs_sha256, reviewed.draft.tool_specs_sha256);
    assert.match(get('recovery-submit-status').textContent,
      data.scenario === 'lost_response' ? /outcome is unconfirmed/ : /identity confirmed/);
  } else if (action === 'retry') {
    assert(context.recoverySubmitPanel.blocksWork());
    await get('retry-recovery').fire('click');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'POST');
    assert.match(get('recovery-submit-results').textContent, /Completion: completed; checks: passed/);
  } else if (action === 'cancel') {
    assert(context.recoverySubmitPanel.blocksWork());
    await get('check-recovery').fire('click');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'GET');
    assert.match(get('recovery-submit-results').textContent, /lifecycle: running/);
    assert(!get('cancel-recovery').disabled);
    await get('cancel-recovery').fire('click');
    assert.equal(calls.length, 2);
    assert.equal(calls[1].method, 'POST');
    assert.match(get('recovery-submit-status').textContent, /Cancellation requested; owned work may still be settling/);
    await get('cancel-recovery').fire('click');
    assert.equal(calls.length, 2, 'acknowledged cancellation must not be resent');
  } else if (action === 'check') {
    assert(context.recoverySubmitPanel.blocksWork());
    assert.match(get('recovery-submit-status').textContent, /Reload sent nothing/);
    await get('check-recovery').fire('click');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'GET');
    assert.match(get('recovery-submit-results').textContent, data.scenario === 'cancelled'
      ? /Completion: cancelled; checks: not_run; lifecycle: cancelled/
      : /Completion: completed; checks: passed/);
  } else throw Error('unknown fixture action');
  assert(context.recoverySubmitPanel.blocksWork(), 'saved review must not enable new edits');
  assert(get('new-session').disabled);
  fs.writeFileSync(statePath, JSON.stringify([...storage]));
  console.log(JSON.stringify({ calls, retained: storage.get(key), text: get('recovery-submit-results').textContent }));
})().catch(error => { console.error(error); process.exitCode = 1; });
