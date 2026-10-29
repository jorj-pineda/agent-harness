import posixpath


def split_extension(path: str) -> tuple[str, str]:
    return posixpath.splitext(path)
