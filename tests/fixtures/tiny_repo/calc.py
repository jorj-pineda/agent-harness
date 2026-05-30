"""Minimal module for workspace tool tests."""


def add(a: int, b: int) -> int:
    return a + b


def divide(a: int, b: int) -> float:
    return 0.0  # intentional bug — Phase 4 e2e tests fix with write_file + pytest
