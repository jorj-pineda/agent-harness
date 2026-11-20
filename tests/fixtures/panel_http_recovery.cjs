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
    assert(storage.has(key), 'exact intent must be retained before network');
    const url = new URL(path, data.url);
    assert.equal(url.origin, new URL(data.url).origin, 'only owned loopback API allowed');
    calls.push({ path, method: options.method ?? 'GET', body: options.body ?? null });
    return fetch(url, options);
  },
});
for (const name of ['app.js', 'recovery.js', 'recovery-submit.js']) {
  vm.runInContext(fs.readFileSync(root + '/' + name, 'utf8'), context);
}
// Deliberate selection seam: draft rendering is outside this submission test.
context.recoveryDraftPanel.reviewed = () => data.candidate;
(async () => {
  assert.equal(calls.length, 0, 'loading/restoring scripts must send nothing');
  if (action === 'submit') {
    for (const name of ['state', 'context', 'budget']) get('recovery-ack-' + name).checked = true;
    context.recoverySubmitPanel.refresh();
    assert(!get('submit-recovery').disabled);
    await get('submit-recovery').fire('click');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'POST');
    assert.match(get('recovery-submit-status').textContent, /identity confirmed/);
  } else if (action === 'check') {
    assert(context.recoverySubmitPanel.blocksWork());
    assert.match(get('recovery-submit-status').textContent, /Reload sent nothing/);
    await get('check-recovery').fire('click');
    assert.equal(calls.length, 1);
    assert.equal(calls[0].method, 'GET');
    assert.match(get('recovery-submit-results').textContent, /Completion: completed; checks: passed/);
  } else throw Error('unknown fixture action');
  assert(context.recoverySubmitPanel.blocksWork(), 'saved review must not enable new edits');
  assert(get('new-session').disabled);
  fs.writeFileSync(statePath, JSON.stringify([...storage]));
  console.log(JSON.stringify({ calls, retained: storage.get(key), text: get('recovery-submit-results').textContent }));
})().catch(error => { console.error(error); process.exitCode = 1; });
