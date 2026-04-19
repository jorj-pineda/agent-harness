"""Dispatch table mapping provider names to `ChatProvider` instances.

The router is deliberately thin: it holds a `name -> ChatProvider` mapping
and resolves a request (or the default) to one of them. It does **not**
construct providers — callers pass already-wired instances in. That keeps
`harness/` free of any `providers.ollama` / `providers.anthropic` imports
(rule #2) and makes tests swap providers by building the mapping directly.

Fallback, retry, and embedder routing are explicitly out of scope here —
see the "Deferred" section in CLAUDE.md.
"""

from __future__ import annotations

from collections.abc import Mapping

from providers.base import ChatProvider


class ProviderNotFoundError(KeyError):
    """Raised when a requested provider name is not in the router's mapping."""


class ProviderRouter:
    """Resolve a provider name (or the default) to a `ChatProvider` instance."""

    def __init__(
        self,
        providers: Mapping[str, ChatProvider],
        *,
        default: str,
    ) -> None:
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider")
        if default not in providers:
            raise ValueError(
                f"default provider {default!r} is not in the provider mapping "
                f"(known: {sorted(providers)})"
            )
        self._providers: dict[str, ChatProvider] = dict(providers)
        self._default = default

    @property
    def default(self) -> str:
        return self._default

    def names(self) -> list[str]:
        """Registered provider names, sorted for stable output."""
        return sorted(self._providers)

    def resolve(self, name: str | None = None) -> ChatProvider:
        """Return the provider registered under `name`, or the default if `None`."""
        key = name if name is not None else self._default
        try:
            return self._providers[key]
        except KeyError as exc:
            raise ProviderNotFoundError(
                f"unknown provider {key!r} (known: {sorted(self._providers)})"
            ) from exc
