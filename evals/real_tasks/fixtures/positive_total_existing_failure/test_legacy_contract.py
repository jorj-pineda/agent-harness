from legacy_settings import DATE_STYLE


def test_known_legacy_format():
    assert DATE_STYLE == "iso"
