import pytest
from settings import lookup_setting, overlay_settings


@pytest.mark.parametrize("override", [None, False, 0, "", [], {}])
def test_present_overrides_win_even_when_false(override):
    result = overlay_settings({"option": "default", "retained": 7}, {"option": override})
    assert result["option"] is override
    assert result["retained"] == 7
    assert set(result) == {"option", "retained"}


def test_union_shallow_identity_and_input_preservation():
    defaults = {"default_only": object(), "nested": {"old": 1}}
    overrides = {"override_only": [], "nested": {"new": 2}}
    before_defaults, before_overrides = defaults.copy(), overrides.copy()
    result = overlay_settings(defaults, overrides)
    assert result is not defaults and result is not overrides
    assert set(result) == {"default_only", "override_only", "nested"}
    assert result["default_only"] is defaults["default_only"]
    assert result["override_only"] is overrides["override_only"]
    assert result["nested"] is overrides["nested"]
    assert result["nested"] == {"new": 2}
    assert defaults["nested"] == {"old": 1}
    assert defaults == before_defaults and overrides == before_overrides
    result.clear()
    assert defaults == before_defaults and overrides == before_overrides


@pytest.mark.parametrize("same_input", [False, True])
def test_empty_and_aliased_inputs_still_return_new_dictionary(same_input):
    defaults = {}
    overrides = defaults if same_input else {}
    result = overlay_settings(defaults, overrides)
    assert result == {} and result is not defaults and result is not overrides
    shared = {"key": object()}
    result = overlay_settings(shared, shared)
    assert result == shared and result is not shared
    assert result["key"] is shared["key"]


def test_existing_lookup_preserves_presence_and_default_identity():
    marker = object()
    assert lookup_setting({"present": None}, "present", marker) is None
    assert lookup_setting({"present": False}, "present", marker) is False
    assert lookup_setting({}, "missing", marker) is marker
    assert lookup_setting({}, "missing") is None
