import pytest
from filenames import split_extension


@pytest.mark.parametrize(
    "path,expected",
    [
        ("", ("", "")),
        ("file", ("file", "")),
        ("file.txt", ("file", ".txt")),
        ("archive.tar.gz", ("archive.tar", ".gz")),
        (".env", (".env", "")),
        ("...", ("...", "")),
        (".config.json", (".config", ".json")),
        ("name.", ("name", ".")),
        ("dir.name/file", ("dir.name/file", "")),
        ("dir.name/file.txt", ("dir.name/file", ".txt")),
        ("dir.name/", ("dir.name/", "")),
        ("../.env", ("../.env", "")),
        ("./A/../ÉCOLE.TXT", ("./A/../ÉCOLE", ".TXT")),
    ],
)
def test_final_component_rules_without_normalization(path, expected):
    result = split_extension(path)
    assert isinstance(result, tuple) and result == expected
    assert "".join(result) == path
