"""Actual panel submission scripts consume real API provenance and enforce manual recovery."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from providers.base import ToolCall
from tests.api.test_continuation import async_wait
from tests.api.test_recovery_execution import (
    RECOVERED,
    SETTINGS,
    known_provider,
    observed_response,
    source,
)
from tests.api.test_session_persistence import make_app

SCRIPT = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {webcrypto}=require('node:crypto');
const data=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const root=process.argv[2],storage=new Map([['panel_user_id','dev']]);
const key='panel_pending_recovery_v1';
class Node {
  constructor(tag) { this.tag=tag;this.children=[];this.events={};this.value='';this._text='';this.checked=false;
    this.dataset={};this.classList={add(){},remove(){}}; }
  appendChild(child){this.children.push(child);child.parent=this;return child;}
  prepend(child){this.children.unshift(child);}
  replaceChildren(...children){this.children=[];children.forEach(child=>this.appendChild(child));}
  remove(){if(this.parent)this.parent.children.splice(this.parent.children.indexOf(this),1);}
  addEventListener(name,fn){this.events[name]=fn;}
  async fire(name){return this.events[name]?.({preventDefault(){}});}
  querySelector(){return null;}
  querySelectorAll(selector){return this.children.flatMap(child=>[...(selector==='button'&&child.tag==='button'?[child]:[]),...child.querySelectorAll(selector)]);}
  set textContent(value){this._text=String(value);}
  get textContent(){return this._text+this.children.map(child=>child.textContent).join('\n');}
  set innerHTML(_){throw Error('unsafe HTML sink');}
  focus(){}
}
const clone=value=>JSON.parse(JSON.stringify(value));
let next,writeFailure=false;
function boot(){
  const ids=new Map(),calls=[];
  const get=id=>{if(!ids.has(id))ids.set(id,new Node('div'));return ids.get(id);};
  const context=vm.createContext({document:{getElementById:get,createElement:tag=>new Node(tag)},
    localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>{if(writeFailure)throw Error('storage denied');storage.set(k,v);},removeItem:k=>storage.delete(k)},
    crypto:{subtle:webcrypto.subtle,randomUUID:()=>data.running.request.request_id},TextEncoder,AbortController,URLSearchParams,
    setTimeout,clearTimeout,window:{},
    fetch:async(url,options={})=>{calls.push({url,options});assert(storage.has(key),'intent must persist before network');
      if(typeof next==='function')return next(url,options);
      return{ok:true,text:async()=>JSON.stringify(next)};
    },
  });
  vm.runInContext(fs.readFileSync(root+'/app.js','utf8'),context);
  assert(vm.runInContext('recoveryWorkBlocked()',context),'editing must stay closed before retained-state inspection loads');
  assert(get('new-session').disabled);
  for(const script of ['recovery.js','recovery-submit.js'])vm.runInContext(fs.readFileSync(root+'/'+script,'utf8'),context);
  let candidate=clone(data.candidate);
  context.recoveryDraftPanel.reviewed=()=>candidate;
  return{context,calls,get,setCandidate:value=>candidate=value};
}
const flags=panel=>{for(const key of ['state','context','budget'])panel.get('recovery-ack-'+key).checked=true;panel.context.recoverySubmitPanel.refresh();};
const clear=async panel=>panel.get('forget-recovery').fire('click');
const bypass=async panel=>{
  const before=panel.calls.length;
  await vm.runInContext('(async()=>{await createSession();sendMessage("bypass");dispatchRequest({});submitContinuation();await postContinuation({});submitReversal();await postReversal({});})()',panel.context);
  assert.equal(panel.calls.length,before,'retained intent must block other editing routes');
};
(async()=>{
  let p=boot();assert.equal(p.calls.length,0);
  // No candidate or missing acknowledgments cannot create intent or send a request.
  p.setCandidate(null);flags(p);await p.get('submit-recovery').fire('click');assert.equal(p.calls.length,0);
  p.setCandidate(data.candidate);p.context.recoverySubmitPanel.invalidate();
  await p.get('submit-recovery').fire('click');assert.equal(p.calls.length,0);
  flags(p);p.context.recoverySubmitPanel.invalidate();assert(!p.get('recovery-ack-context').checked);
  flags(p);next=data.running;await p.get('submit-recovery').fire('click');
  assert.equal(p.calls.length,1);assert.match(p.calls[0].url,/\/recover$/);
  const retained=storage.get(key),intent=JSON.parse(retained),body=p.calls[0].options.body;
  assert.equal(body,JSON.stringify(intent.body));assert.equal(intent.source_run_id,data.candidate.source_run_id);
  assert.equal(intent.body.message,data.candidate.input.message);
  assert.match(p.get('recovery-submit-results').textContent,/Completion: unknown; checks: unknown; lifecycle: running/);
  assert(p.context.recoverySubmitPanel.blocksWork());await bypass(p);
  // Cancellation targets only the validated fork and does not imply final cleanup.
  next={status:'requested'};await p.get('cancel-recovery').fire('click');
  assert.equal(p.calls.at(-1).url,`/sessions/${data.running.request.session_id}/cancel`);
  assert.deepEqual(JSON.parse(p.calls.at(-1).options.body),{user_id:'dev'});
  assert.match(p.get('recovery-submit-status').textContent,/still be settling/);
  // Reload does no HTTP; only explicit Check looks up the key, validates and renders final evidence.
  p=boot();assert.equal(p.calls.length,0);assert(p.context.recoverySubmitPanel.blocksWork());await bypass(p);
  next=[data.completed];await p.get('check-recovery').fire('click');
  assert.equal(p.calls.length,1);assert(!p.calls[0].options.method);
  assert.match(p.calls[0].url,/request_id=recover-once/);
  assert.match(p.get('recovery-submit-results').textContent,/Completion: completed; checks: passed/);
  assert.equal(storage.get(key),retained);await clear(p);assert(!storage.has(key));
  assert.equal(vm.runInContext('sessionUsable',p.context),false);
  // Lost POST remains uncertain; explicit retry keeps the original input despite changed form state.
  p=boot();flags(p);next=()=>{throw Error('private-marker');};await p.get('submit-recovery').fire('click');
  assert.equal(p.calls.length,1);assert(!p.get('recovery-submit-status').textContent.includes('private-marker'));
  const lostBody=p.calls[0].options.body;next=data.completed;
  p.setCandidate({...data.candidate,input:{...data.candidate.input,message:'unreviewed change'}});
  await p.get('retry-recovery').fire('click');assert.equal(p.calls.length,2);
  assert.equal(p.calls[1].options.body,lostBody);await clear(p);
  // Identity corruption never renders a confirmed result or triggers another request.
  for(const field of ['owner','key','fork','approval','hash','scope','source']){
    p=boot();flags(p);next=clone(data.completed);
    const r=next.request,o=r.recovery_source;
    if(field==='owner')r.user_id='other';if(field==='key')r.request_id='other';if(field==='fork')r.session_id='other';
    if(field==='approval')o.approval_json='{}';if(field==='hash')o.approval_sha256='0'.repeat(64);
    if(field==='scope')o.budget_scope='hypothetical-new-turn';if(field==='source')o.source_run_id='other';
    await p.get('submit-recovery').fire('click');assert.equal(p.calls.length,1);
    assert.equal(p.get('recovery-submit-results').textContent,'');
    assert.match(p.get('recovery-submit-status').textContent,/unconfirmed/);await clear(p);
  }
  // Readable but mismatched or unreadable storage freezes all new edits and cannot be retried.
  for(const raw of ['{broken',JSON.stringify({...intent,body:{...intent.body,user_id:'other'}}),retained.replace('"schema_version":1','"schema_version":true')]){
    storage.set(key,raw);p=boot();assert.equal(p.calls.length,0);await bypass(p);
    await p.get('retry-recovery').fire('click');assert.equal(p.calls.length,0);
    assert(p.context.recoverySubmitPanel.blocksWork());await clear(p);
  }
  // Failure to persist means no POST. A later storage race also refuses exact retry.
  p=boot();flags(p);writeFailure=true;await p.get('submit-recovery').fire('click');writeFailure=false;
  assert.equal(p.calls.length,0);await bypass(p);await clear(p);
  storage.set(key,retained);p=boot();storage.set(key,retained+' ');await p.get('retry-recovery').fire('click');
  assert.equal(p.calls.length,0);assert(p.context.recoverySubmitPanel.blocksWork());await clear(p);
  console.log('Actual app + panel guarded submission/restart/retry/cancel contract passed.');
})().catch(error=>{console.error(error);process.exit(1);});
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="Node needed for panel contract")
def test_panel_submission_uses_real_api_provenance_and_blocks_unsafe_recovery(tmp_path):
    data = source(tmp_path)
    provider = known_provider()
    provider.script(
        observed_response(
            tool_calls=[
                ToolCall(
                    id="edit", name="write_file", arguments={"path": "a.py", "content": RECOVERED}
                )
            ]
        ),
        observed_response(
            tool_calls=[
                ToolCall(id="check", name="run_command", arguments={"argv": ["pytest", "-q"]})
            ]
        ),
        observed_response(content="Changed and checked."),
    )
    with TestClient(make_app(tmp_path, provider, **SETTINGS)) as api:
        response = api.post(f"/runs/{data['rid']}/recover", json=data["approval"])
        assert response.status_code == 202, response.text
        running = response.json()
        api.portal.call(async_wait, api.app.state.components, running["run_id"])
        completed = api.get(f"/runs/{running['run_id']}", params={"user_id": "dev"}).json()
        assert completed["response"]["completion_status"] == "completed"
        assert completed["response"]["verification_status"] == "passed"
        assert (
            api.get(f"/sessions/{data['sid']}", params={"user_id": "dev"}).json() == data["archive"]
        )
    candidate = {
        "source_run_id": data["rid"],
        "source_session_id": data["sid"],
        "draft": data["draft"],
        "input": {
            "user_id": "dev",
            "expected_record_sha256": data["approval"]["expected_record_sha256"],
            "spans": data["approval"]["spans"],
            "message": data["approval"]["message"],
        },
    }
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps({"candidate": candidate, "running": running, "completed": completed})
    )
    script = tmp_path / "panel.cjs"
    script.write_text(SCRIPT)
    root = Path(__file__).resolve().parents[1] / "ui"
    result = subprocess.run(
        ["node", str(script), str(root), str(packet)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    assert len(provider.calls) == 3
    assert (data["root"] / "a.py").read_text() == RECOVERED
    assert (data["root"] / ".git/index").read_bytes() == data["index"]
