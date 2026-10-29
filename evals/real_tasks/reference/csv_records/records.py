import csv
import io


def parse_records(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline="")))
