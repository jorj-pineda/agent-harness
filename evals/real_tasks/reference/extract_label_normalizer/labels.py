def normalize_label(value: str) -> str:
    return value.strip().lower()


def label_for(value: str) -> str:
    return f"[{normalize_label(value)}]"


def index_labels(values) -> dict[str, int]:
    result = {}
    for position, value in enumerate(values):
        result[normalize_label(value)] = position
    return result
