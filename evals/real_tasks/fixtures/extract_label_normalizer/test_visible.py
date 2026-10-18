from labels import index_labels, label_for, normalize_label


def test_normalization_and_existing_callers():
    assert normalize_label("  Hello ") == "hello"
    assert label_for("  Hello ") == "[hello]"
    assert index_labels([" One ", "TWO", "one"]) == {"one": 2, "two": 1}
