def split_extension(path: str) -> tuple[str, str]:
    if "." not in path:
        return path, ""
    root, extension = path.rsplit(".", 1)
    return root, "." + extension
