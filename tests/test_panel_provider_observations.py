"""Exercise actual panel metadata format/render functions without a browser dependency."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel render contract")
def test_panel_observations_are_collapsed_bounded_sanitized_and_unknown_safe():
    source = Path(__file__).resolve().parents[1] / "ui/app.js"
    script = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[1], 'utf8');
class Node {
  constructor(tag) { this.tag = tag; this.children = []; this._text = ''; this.open = false; this.classList = {add() {}}; }
  appendChild(child) { this.children.push(child); return child; }
  set textContent(value) { this._text = String(value); }
  get textContent() { return this._text + this.children.map(x => x.textContent).join('\n'); }
}
const context = vm.createContext({document: {createElement: tag => new Node(tag)}});
vm.runInContext(source.slice(source.indexOf('function el('), source.indexOf('function clearEmptyState(')), context);
vm.runInContext(source.slice(source.indexOf('function listSection('), source.indexOf('function workspaceChangeItems(')), context);
// listSection delegates string conversion to the existing pretty helper.
context.pretty = x => typeof x === 'string' ? x : JSON.stringify(x);
const render = context.providerObservationSection;
const format = context.providerObservationText;
assert.equal(format(null), 'unrecorded');
assert.equal(format([]), 'unrecorded');
const observation = {duration_ms:0, prompt_usage_field:'present', completion_usage_field:'null', terminal_reason:'other', content_empty:false, tool_call_count:0};
assert.match(format(observation), /HTTP=0.0ms native_usage=present\/null/);
assert.match(format(observation), /zero_length_content=no tool_calls=0/);
const hostile = format({duration_ms:true, prompt_usage_field:'private', terminal_reason:'<img onerror=private>', content_empty:'false', tool_call_count:true});
assert(!hostile.includes('private')); assert(!hostile.includes('<img')); assert.match(hostile, /HTTP=unknown/);
for (const record of [{}, {budget_evidence:null}, {budget_evidence:{events:[null,2,{kind:'response'}]}}]) {
  const node = render(record); assert.equal(node.tag, 'details'); assert.equal(node.open, false);
  assert.match(node.textContent, /unrecorded/); assert.match(node.textContent, /prefix_truncated=unknown/);
}
const repeated = {kind:'response', iteration:0, call_observation:observation};
const node = render({budget_evidence:{events:Array(101).fill(repeated), observed_events:150, events_truncated:true}, provider_failure:{call_observation:{duration_ms:60000}}});
assert.equal((node.textContent.match(/request=1 response:/g) || []).length, 100);
assert.match(node.textContent, /saved_events=100 observed_events=150 prefix_truncated=yes/);
assert.match(node.textContent, /final failure \(separate record\): HTTP=60000.0ms/);
assert.match(node.textContent, /later observations may be absent/);
assert.match(node.textContent, /not valid usage or budget compliance/);
const failed = render({budget_evidence:{events:[{kind:'request_failure',iteration:2,call_observation:observation}],events_truncated:false},provider_failure:{call_observation:observation}});
assert.match(failed.textContent, /request=3 request_failure:/); assert(!failed.textContent.includes('separate record'));
console.log('panel render contract passed; no HTTP or workflow invocation');
"""
    result = subprocess.run(
        [shutil.which("node"), "-e", script, str(source)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
