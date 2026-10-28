"""Actual client presentation stays bounded, unknown-safe and read-only."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cli.main import app, show_token_refusals, token_refusal_text
from providers.base import ToolCall
from tests.api.conftest import ScriptedProvider, make_response
from tests.api.test_session_persistence import make_app


def event(kind="pre_request_stop", **changes):
    return {
        "kind": kind,
        "iteration": 2,
        "request_block": {
            "format": "request-budget-block-v1",
            "budget": "total",
            "basis": "estimate",
            "limit_tokens": 32000,
            "available_tokens": 11882,
            "prompt_tokens": 11854,
            "minimum_output_tokens": 256,
            **changes,
        },
    }


def presentation_cases():
    before = event()
    after = event(
        "post_response_stop",
        basis="reported_prompt_lower_bound",
        available_tokens=8005,
        prompt_tokens=9719,
    )
    context = event(budget="context", available_tokens=32000, prompt_tokens=31900)
    cases = [
        (
            before,
            "Before request: total tokens; limit=32000; available before prompt=11882; estimated prompt=11854; minimum output=256.",
        ),
        (
            after,
            "After response: total tokens; limit=32000; available before prompt=8005; reported prompt lower bound=9719; minimum output=256.",
        ),
        (
            context,
            "Before request: context window; limit=32000; available before prompt=32000; estimated prompt=31900; minimum output=256.",
        ),
        ({"kind": "pre_request_stop"}, "unrecorded"),
    ]
    for changes in (
        {"format": "private"},
        {"budget": "<img onerror=private>"},
        {"basis": "private"},
        {"limit_tokens": True},
        {"available_tokens": "private"},
        {"prompt_tokens": -1},
        {"minimum_output_tokens": None},
        {"prompt_tokens": 2**53},
        {"limit_tokens": 0},
        {"prompt_tokens": 1},
        {"budget": "context"},
        {"available_tokens": 33000},
    ):
        cases.append((event(**changes), "unrecorded or unsupported"))
    for value in (None, [], "private", 5):
        damaged = event()
        damaged["request_block"] = value
        cases.append((damaged, "unrecorded" if value is None else "unrecorded or unsupported"))
    cases.append((event("post_response_stop"), "unrecorded or unsupported"))
    cases.append((event("response"), "unrecorded or unsupported"))
    for kind in (None, [], {}, 5):
        cases.append((event(kind), "unrecorded or unsupported"))
    cases.append(
        (
            event(available_tokens=0, prompt_tokens=0, minimum_output_tokens=1),
            "Before request: total tokens; limit=32000; available before prompt=0; estimated prompt=0; minimum output=1.",
        )
    )
    return cases


@pytest.mark.parametrize("value,expected", presentation_cases())
def test_cli_formats_only_supported_recorded_comparisons(value, expected):
    assert token_refusal_text(value) == expected


NODE_SCRIPT = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync(process.argv[1],'utf8');
const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class Node {
 constructor(tag){this.tag=tag;this.children=[];this._text='';this.open=false;this.classList={add(){}};}
 appendChild(child){this.children.push(child);return child;}
 prepend(child){this.children.unshift(child);}
 replaceChildren(...children){this.children=children;}
 set textContent(value){this._text=String(value);}
 get textContent(){return this._text+this.children.map(c=>c.textContent).join('\n');}
 set innerHTML(_){throw Error('Unsafe HTML sink');}
}
const context=vm.createContext({document:{createElement:tag=>new Node(tag)},
 fetch(){throw Error('Review rendering attempted HTTP');},
 localStorage:{setItem(){throw Error('Unexpected storage');}},
 pendingRequest:null, sessionId:null, sessionUsable:true,
 els:{sessionPolicy:new Node('p'),sessionPill:new Node('p'),messages:new Node('div'),envelope:new Node('div')},
 selectReviewTarget(){},addUserMessage(){},addErrorMessage(){},addTurn(){},renderEnvelope(){},setBusy(){},
});
vm.runInContext(source.slice(source.indexOf('function el('),source.indexOf('function clearEmptyState(')),context);
vm.runInContext(source.slice(source.indexOf('function listSection('),source.indexOf('function workspaceChangeItems(')),context);
vm.runInContext(source.slice(source.indexOf('function showRun('),source.indexOf('async function inspectRun(')),context);
context.pretty=x=>typeof x==='string'?x:JSON.stringify(x);
for(const [value,expected] of data.cases){assert.equal(context.tokenRefusalText(value),expected);}
for(const evidence of [null,{}, {events:[null,5,{}]}]){
 const node=context.tokenRefusalSection({budget_evidence:evidence});
 assert.equal(node.tag,'details');assert.equal(node.open,false);
 assert.match(node.textContent,/unrecorded/);assert.match(node.textContent,/prefix_truncated=unknown/);
}
const repeated=data.cases[0][0];
const node=context.tokenRefusalSection({budget_evidence:{events:Array(101).fill(repeated),events_truncated:true}});
assert.equal((node.textContent.match(/Before request:/g)||[]).length,100);
assert.match(node.textContent,/later refusals may be absent/);
assert(!node.textContent.includes('private'));
if(data.record){
 context.showRun(data.record);
 assert.equal(context.sessionUsable,false);
 assert(context.els.envelope.textContent.includes(data.expected));
 assert.match(context.els.envelope.textContent,/not next-request usage/);
}
console.log('actual panel functions passed; rendering invoked no HTTP or storage action');
"""


def render_panel(tmp_path, **data):
    if shutil.which("node") is None:
        pytest.skip("Node required for actual panel function checks")
    payload = tmp_path / "panel-input.json"
    payload.write_text(json.dumps({"cases": presentation_cases(), **data}))
    result = subprocess.run(
        [
            shutil.which("node"),
            "-e",
            NODE_SCRIPT,
            str(Path(__file__).resolve().parents[1] / "ui/app.js"),
            str(payload),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_panel_supported_unknown_malformed_and_bounded_metadata(tmp_path):
    render_panel(tmp_path)


@pytest.mark.parametrize("stage", ["estimate", "reported_prompt_lower_bound"])
def test_clients_review_real_saved_refusal_after_restart_without_dispatch(
    tmp_path, monkeypatch, stage
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "a.py"
    source.write_text("private-source" * 1000)
    initial = source.read_bytes()
    provider = ScriptedProvider()
    provider.script(
        make_response(
            tool_calls=[ToolCall(id="read", name="read_file", arguments={"path": "a.py"})],
            prompt_tokens=100 if stage == "estimate" else 10000,
            completion_tokens=5,
        )
    )
    limit = 7000 if stage == "estimate" else 15000
    with TestClient(make_app(tmp_path, provider, max_total_tokens_per_turn=limit)) as client:
        sid = client.post("/sessions", json={"user_id": "dev", "workspace_root": str(repo)}).json()[
            "session_id"
        ]
        result = client.post(
            "/chat", json={"user_id": "dev", "session_id": sid, "message": "Inspect"}
        ).json()
        endpoint = f"/runs/{result['run_id']}"
        saved = client.get(endpoint, params={"user_id": "dev"}).json()
        stop = saved["budget_evidence"]["events"][-1]
        assert stop["request_block"]["basis"] == stage
        assert len(provider.calls) == 1
    restarted = ScriptedProvider()
    with TestClient(make_app(tmp_path, restarted)) as client:
        gets = []

        def get(url, **kwargs):
            gets.append(url)
            assert url.endswith(endpoint)
            return client.get(endpoint, params=kwargs["params"])

        monkeypatch.setattr("cli._client.httpx.get", get)
        before = copy.deepcopy(client.get(endpoint, params={"user_id": "dev"}).json())
        response = CliRunner().invoke(app, ["run", result["run_id"], "--user-id", "dev"])
        assert response.exit_code == 0, response.output
        expected = token_refusal_text(stop)
        assert expected in response.output
        assert "Next-request token refusals (saved review only)" in response.output
        assert "not next-request usage" in response.output
        assert len(gets) == 1
        raw = CliRunner().invoke(app, ["run", result["run_id"], "--user-id", "dev", "--json"])
        assert raw.exit_code == 0 and json.loads(raw.output) == saved
        assert len(gets) == 2
        render_panel(tmp_path, record=saved, expected=expected)
        assert client.get(endpoint, params={"user_id": "dev"}).json() == before == saved
        assert restarted.calls == [] and source.read_bytes() == initial


def test_cli_prefix_bound_and_legacy_details_stay_unknown(capsys):
    show_token_refusals(
        {
            "budget_evidence": {
                "events": [event()] * 101,
                "events_truncated": True,
            }
        }
    )
    text = capsys.readouterr().out
    assert text.count("Before request:") == 100
    assert "prefix_truncated=yes" in text and "later refusals may be absent" in text
    for evidence in (
        None,
        {},
        {"events": [None, 5, {}]},
        {"events": [{"kind": "pre_request_stop"}]},
    ):
        show_token_refusals(
            {
                "budget_evidence": evidence,
                "execution_profile": {"settings": {"max_total_tokens_per_turn": 999}},
            }
        )
        text = capsys.readouterr().out
        assert "unrecorded" in text and "prefix_truncated=unknown" in text
        assert "999" not in text
