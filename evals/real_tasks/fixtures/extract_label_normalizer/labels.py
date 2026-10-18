def label_for(value: str) -> str:
    normalized = value.strip().lower()
    return f"[{normalized}]"


def index_labels(values) -> dict[str, int]:
    result = {}
    for position, value in enumerate(values):
        normalized = value.strip().lower()
        result[normalized] = position
    return result
