from deadlines import deadline_after, is_expired


def test_injected_clock():
    assert deadline_after(5, clock=lambda: 100) == 105
    assert is_expired(99, clock=lambda: 100) is True
