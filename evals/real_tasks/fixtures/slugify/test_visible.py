from slug import slugify


def test_slug() -> None:
    assert slugify("Hello, World!") == "hello-world"
