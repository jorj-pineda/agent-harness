import json

import pytest

from evals.native_summary import summarize


def save(directory, name, data):
    path = directory / name
    path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("status", [200, 500])
def test_summary_distinguishes_native_completion_from_http_error(tmp_path, status):
    save(tmp_path, "request-01.json", {"status_code": status})
    save(tmp_path, "request-01.request.bin", b"private request sentinel")
    save(
        tmp_path,
        "request-01.response.bin",
        {
            "done": True,
            "model": "gemma4:12b",
            "prompt_eval_count": 12,
            "eval_count": 3,
            "message": "private output sentinel",
        },
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = summarize(tmp_path)
    row = result.requests[0]
    assert row.model_completed == (status == 200)
    assert row.prompt_tokens == (12 if status == 200 else None)
    assert row.completion_tokens == (3 if status == 200 else None)
    assert row.elapsed_seconds is None
    assert "private" not in result.model_dump_json()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_failure_summary_has_unknown_completion_and_scoped_timing(tmp_path):
    save(
        tmp_path,
        "request-01.failure.json",
        {
            "error_type": "private-sentinel",
            "timing": {"scope": "native_request_to_evidence_record_v1", "elapsed_seconds": 2.5},
        },
    )
    save(tmp_path, "request-01.partial-response.bin", b"private partial")
    row = summarize(tmp_path).requests[0]
    assert row.model_completed is None and row.status_code is None
    assert row.failure_recorded and row.partial_response_recorded and row.elapsed_seconds == 2.5
    assert row.prompt_tokens is None and row.completion_tokens is None


@pytest.mark.parametrize(
    "body", [b"bad JSON", b"[]", b"\xff", b'{"done":false}', b'{"done":1,"model":"x"}']
)
def test_invalid_or_incomplete_body_is_not_model_completion(tmp_path, body):
    save(tmp_path, "request-01.json", {"status_code": 200})
    save(tmp_path, "request-01.response.bin", body)
    assert summarize(tmp_path).requests[0].model_completed is False


def test_missing_status_and_invalid_usage_stay_unknown(tmp_path):
    body = {"done": True, "model": "x", "prompt_eval_count": True, "eval_count": -1}
    save(tmp_path, "request-01.response.bin", body)
    assert summarize(tmp_path).requests[0].model_completed is None
    save(tmp_path, "request-01.json", {"status_code": 200})
    row = summarize(tmp_path).requests[0]
    assert (
        row.model_completed is True and row.prompt_tokens is None and row.completion_tokens is None
    )


@pytest.mark.parametrize(
    "fault", ["mode", "link", "oversize", "unknown", "ordinal", "timing", "status"]
)
def test_unsafe_or_invalid_evidence_refuses(tmp_path, fault):
    path = save(tmp_path, "request-01.json", {"status_code": 200})
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "link":
        path.unlink()
        path.symlink_to(tmp_path / "missing")
    elif fault == "oversize":
        path.write_bytes(b"x" * 2_000_001)
    elif fault == "unknown":
        path.rename(tmp_path / "unknown.json")
    elif fault == "ordinal":
        path.rename(tmp_path / "request-33.json")
    elif fault == "timing":
        path.write_text(
            json.dumps(
                {
                    "timing": {
                        "scope": "native_request_to_evidence_record_v1",
                        "elapsed_seconds": float("nan"),
                    }
                }
            )
        )
    else:
        path.write_text('{"status_code":true}')
    with pytest.raises((ValueError, OSError)):
        summarize(tmp_path)


def test_inventory_bound_refuses_before_reading_files(tmp_path, monkeypatch):
    import evals.native_summary as summary

    monkeypatch.setattr(summary, "MAX_REQUESTS", 1)
    for index in range(len(summary.SUFFIXES) + 1):
        (tmp_path / str(index)).touch()
    with pytest.raises(ValueError, match="inventory exceeds"):
        summarize(tmp_path)


def test_response_read_drift_is_not_misreported_as_invalid_json(tmp_path, monkeypatch):
    import evals.native_summary as summary
    from evals.heldout_guard import HeldoutDriftError

    save(tmp_path, "request-01.json", {"status_code": 200})
    save(tmp_path, "request-01.response.bin", {"done": True, "model": "x"})
    original = summary._read_regular

    def read(path, limit):
        if path.name.endswith("response.bin"):
            raise HeldoutDriftError("scripted drift")
        return original(path, limit)

    monkeypatch.setattr(summary, "_read_regular", read)
    with pytest.raises(HeldoutDriftError, match="scripted drift"):
        summarize(tmp_path)
