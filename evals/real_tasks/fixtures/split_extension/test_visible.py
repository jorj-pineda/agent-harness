from filenames import split_extension


def test_dotfile_and_regular_extension():
    assert split_extension(".env") == (".env", "")
    assert split_extension("file.txt") == ("file", ".txt")
