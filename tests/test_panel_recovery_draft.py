"""Exercise actual panel request/render/invalidation paths against production-shaped data."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_cli_recovery_draft import fixture

SCRIPT = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');
const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
class Node {
  constructor(tag) { this.tag=tag; this.children=[]; this.events={}; this.value=''; this._text=''; this.disabled=false; }
  appendChild(child) { this.children.push(child); child.parent=this; return child; }
  replaceChildren(...children) { this.children=[]; for(const child of children) this.appendChild(child); }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this),1); }
  addEventListener(name,fn) { this.events[name]=fn; }
  async fire(name) { return this.events[name]?.(); }
  set textContent(value) { this._text=String(value); }
  get textContent() { return this._text+this.children.map(c=>c.textContent).join('\n'); }
  set innerHTML(_) { throw Error('Unsafe HTML sink'); }
}
const ids = new Map();
const byId = id => { if(!ids.has(id)) ids.set(id,new Node('div')); return ids.get(id); };
const calls=[]; let next; let network=0;
const context=vm.createContext({document:{getElementById:byId,createElement:tag=>new Node(tag)},
  userId:'dev',busy:false,reviewInFlight:false,TextEncoder,AbortController,URLSearchParams,
  crypto:webcrypto,setTimeout,clearTimeout,
  fetch:async (url,options={})=> { calls.push({url,options}); network++;
    if(typeof next==='function') return next();
    return {ok:true,text:async()=>JSON.stringify(next)};
  },
});
context.setBusy=value=> { context.busy=value; context.recoveryDraftPanel?.refresh(); };
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
const panel=context.recoveryDraftPanel;
const run={run_id:data.rid,request:{session_id:data.sid}};
const clone=value=>JSON.parse(JSON.stringify(value));
const sourceReview=JSON.parse(data.draft.messages[data.draft.evidence_message_index].content.split('\n').slice(1).join('\n')).review;
async function load() { next=sourceReview; await byId('load-recovery').fire('click'); }
async function prepare() {
  panel.select(run); await load();
  byId('recovery-message').value=data.message; await byId('recovery-message').fire('input');
  byId('recovery-provider').value='ollama'; await byId('recovery-provider').fire('change');
  for(const span of data.selection.spans) {
    await byId('add-recovery-range').fire('click');
    const row=byId('recovery-ranges').children.at(-1);
    row.children.filter(c=>c.tag==='label').forEach((label,i)=> {
      label.children[0].value=String(span[['path','expected_sha256','start_line','end_line'][i]]);
    });
  }
}
(async()=> {
  assert.equal(calls.length,0); panel.select(run); assert.equal(calls.length,0);
  assert.equal(byId('preview-recovery').disabled,true);
  await prepare();
  next=data.draft; await byId('preview-recovery').fire('click');
  const result=byId('recovery-draft-results');
  assert.match(result.textContent,/Source completion: budget_exhausted; checks: failed/);
  assert.match(result.textContent,/execution_supported=false/);
  assert.match(result.textContent,/hypothetical-new-turn/);
  assert.match(result.textContent,/omitted nonuser messages/);
  assert.match(result.textContent,/Full prospective messages/);
  const request=calls.at(-1);
  assert.equal(request.url,`/runs/${data.rid}/recovery-draft`);
  assert.equal(request.options.method,'POST');
  assert.deepEqual(JSON.parse(request.options.body),{user_id:'dev',...data.selection,message:data.message,provider:'ollama'});
  assert(calls.every(c=>c.url.includes('/recovery-review?')||c.url.endsWith('/recovery-draft')));
  // Any changed input clears the displayed draft, without a fetch or stored approval.
  const before=network;
  byId('recovery-message').value+='changed'; await byId('recovery-message').fire('input');
  assert.equal(result.textContent,''); assert.equal(network,before);
  // Corrupt envelopes must not render even when the HTTP status succeeds.
  for(const field of ['schema_version','execution_supported','messages_sha256','tool_specs_sha256','source_archive_sha256','budget_scope','context','configuration']) {
    await prepare(); next=clone(data.draft);
    next[field]=field.endsWith('sha256')?'bad':null;
    await byId('preview-recovery').fire('click');
    assert.equal(result.textContent,'',field);
    assert.match(byId('recovery-status').textContent,/unavailable, inconsistent or timed out/);
  }
  // Counts and hashes are checked on the actual supplied messages and embedded source.
  await prepare(); next=clone(data.draft); next.messages[1].content+='tampered';
  await byId('preview-recovery').fire('click'); assert.equal(result.textContent,'');
  // Local byte/range preconditions fail before any request.
  await prepare(); byId('recovery-message').value='é'.repeat(2049);
  const count=network; await byId('preview-recovery').fire('click'); assert.equal(network,count);
  // An old source/draft response cannot render after selecting another run.
  await prepare(); let resolve;
  next=()=>new Promise(r=>{resolve=r;});
  const pending=byId('preview-recovery').fire('click');
  panel.select({run_id:'other-run',request:{session_id:'other-session'}});
  resolve({ok:true,text:async()=>JSON.stringify(data.draft)}); await pending;
  assert.equal(result.textContent,''); assert.equal(byId('recovery-evidence').textContent,'');
  assert.match(byId('recovery-target').textContent,/other-run/);
  assert.equal(context.reviewInFlight,false);
  // Missing final evidence stays unknown and disables preview, without inventing zero counts.
  panel.select(run); next=clone(sourceReview); next.final_response_available=false;
  next.run_status='interrupted'; next.completion_status=null; next.verification_status=null;
  for(const key of ['unresolved_checks','recent_actions','diffs']) next[key]=[];
  for(const key of ['unresolved_check_count','action_count','diff_count','omitted_unresolved_checks','omitted_actions','omitted_diffs']) next[key]=null;
  await byId('load-recovery').fire('click');
  assert.match(byId('recovery-evidence').textContent,/completion: unknown; checks: unknown/);
  assert.equal(byId('preview-recovery').disabled,true);
  // No private network text is displayed and no automatic fallback/retry occurs.
  panel.select(run); const prior=network; next=()=>{throw Error('private-marker');};
  await byId('load-recovery').fire('click'); assert.equal(network,prior+1);
  assert(!byId('recovery-status').textContent.includes('private-marker'));
  console.log('Panel recovery request/render/invalidation contract passed; no execution route.');
})().catch(error=>{console.error(error);process.exit(1);});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel contract")
def test_panel_draft_request_render_invalidation_and_unknowns(tmp_path):
    rid, selection, message, _, draft = fixture(tmp_path)
    packet = tmp_path / "panel.json"
    evidence = json.loads(
        draft["messages"][draft["evidence_message_index"]]["content"].split("\n", 1)[1]
    )
    packet.write_text(
        json.dumps(
            {
                "rid": rid,
                "sid": evidence["spans"][0]["session_id"] if evidence["spans"] else "s1",
                "selection": selection,
                "message": message,
                "draft": draft,
            }
        )
    )
    script = tmp_path / "panel.cjs"
    script.write_text(SCRIPT)
    source = Path(__file__).resolve().parents[1] / "ui/recovery.js"
    result = subprocess.run(
        ["node", str(script), str(source), str(packet)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
