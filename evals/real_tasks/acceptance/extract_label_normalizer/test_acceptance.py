import labels
import pytest


@pytest.mark.parametrize("value", ["", " \t\n", " A B ", " Straße ", "İ", "ÉCOLE"])
def test_extracted_helper_and_formatting_preserve_normalization(value):
    assert labels.normalize_label(value) == value.strip().lower()
    assert labels.label_for(value) == f"[{value.strip().lower()}]"


def test_index_preserves_order_duplicates_empty_and_input():
    values = [" B ", "A", "b", "", " Straße ", "STRASSE", " \t "]
    original = values.copy()
    expected = {"b": 2, "a": 1, "": 6, "straße": 4, "strasse": 5}
    result = labels.index_labels(values)
    assert result == expected and list(result) == list(expected)
    assert labels.index_labels(iter(values)) == expected
    assert labels.index_labels([]) == {}
    assert values == original


@pytest.mark.parametrize("value", [None, 1, False])
def test_existing_attribute_errors_remain(value):
    for function in (labels.normalize_label, labels.label_for):
        with pytest.raises(AttributeError):
            function(value)
    with pytest.raises(AttributeError):
        labels.index_labels([value])


@pytest.mark.parametrize("value", [None, 1, False])
def test_non_iterable_index_input_keeps_type_error(value):
    with pytest.raises(TypeError):
        labels.index_labels(value)


def test_both_callers_use_current_module_helper(monkeypatch):
    calls = []

    def replacement(value):
        calls.append(value)
        return f"key:{value}"

    monkeypatch.setattr(labels, "normalize_label", replacement)
    assert labels.label_for(" X ") == "[key: X ]"
    assert labels.index_labels([" A ", "b", " A "]) == {"key: A ": 2, "key:b": 1}
    assert calls == [" X ", " A ", "b", " A "]
