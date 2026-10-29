import settings


def test_overlay_and_existing_lookup():
    assert settings.overlay_settings({"timeout": 10, "retries": 2}, {"timeout": 30}) == {
        "timeout": 30,
        "retries": 2,
    }
    assert settings.lookup_setting({}, "missing", 5) == 5
