from slug import slugify


def test_slug() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  A   B  ") == "a-b"
    assert slugify("One_two") == "one-two"
    assert slugify("Release 42") == "release-42"
    assert slugify("!! A...B !!") == "a-b"
    # The request requires ASCII, without selecting a transliteration policy.
    assert slugify("Café_42").isascii()
