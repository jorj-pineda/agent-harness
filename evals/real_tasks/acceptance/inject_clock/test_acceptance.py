import deadlines
import pytest


def test_injected_clock_is_called_once_without_default_reads(monkeypatch):
    def forbidden():
        pytest.fail("Injected clock must not read the default clock")

    monkeypatch.setattr(deadlines.time, "time", forbidden)
    calls = []

    def clock():
        calls.append(None)
        return 40.25

    assert deadlines.deadline_after(2.5, clock=clock) == 42.75
    assert len(calls) == 1
    assert deadlines.is_expired(40.25, clock=clock) is True
    assert len(calls) == 2


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_clock_is_resolved_at_each_call(monkeypatch, explicit_none):
    readings = iter([100, 200, 300, 400])
    calls = []

    def clock():
        calls.append(None)
        return next(readings)

    monkeypatch.setattr(deadlines.time, "time", clock)
    kwargs = {"clock": None} if explicit_none else {}
    assert deadlines.deadline_after(-0.5, **kwargs) == 99.5
    assert deadlines.is_expired(201, **kwargs) is False
    assert deadlines.deadline_after(0, **kwargs) == 300
    assert deadlines.is_expired(400, **kwargs) is True
    assert len(calls) == 4


@pytest.mark.parametrize("deadline,expected", [(9.5, True), (10.5, True), (11.5, False)])
def test_inclusive_expiry_boundary(deadline, expected):
    assert deadlines.is_expired(deadline, clock=lambda: 10.5) is expected


@pytest.mark.parametrize("name", ["deadline_after", "is_expired"])
def test_clock_exceptions_propagate_without_retry(name):
    error = RuntimeError("clock unavailable")
    calls = []

    def clock():
        calls.append(None)
        raise error

    with pytest.raises(RuntimeError) as caught:
        getattr(deadlines, name)(1, clock=clock)
    assert caught.value is error and len(calls) == 1


@pytest.mark.parametrize("name", ["deadline_after", "is_expired"])
def test_clock_argument_is_keyword_only(name):
    with pytest.raises(TypeError):
        getattr(deadlines, name)(1, lambda: 0)
