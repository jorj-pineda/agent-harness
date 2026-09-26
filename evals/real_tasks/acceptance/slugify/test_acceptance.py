from slug import slugify


def test_slug() -> None:
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  A   B  ") == "a-b"
    assert slugify("One_two") == "one-two"
