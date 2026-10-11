"""Build the configured LLM client.

``create_llm`` returns ``None`` for the *offline* provider. Every agent is
written so that it works without an LLM (deterministic analytics) and uses the
LLM to *enrich* its reasoning when one is available - so a missing key or an
API outage degrades quality, never availability.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from sentinelcrew.config import Settings
from sentinelcrew.llm.base import LLMClient, LLMError
from sentinelcrew.llm.http import Transport
from sentinelcrew.llm.providers import AnthropicClient, GeminiClient


def create_llm(settings: Settings, transport: Transport | None = None) -> LLMClient | None:
    provider = settings.resolved_provider()
    common: dict[str, Any] = {
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
        "timeout_s": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
        "transport": transport,
    }
    if provider == "offline":
        return None
    if provider == "gemini":
        return GeminiClient(settings.gemini_api_key, model=settings.gemini_model, **common)
    if provider == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, model=settings.anthropic_model, **common)
    raise LLMError(f"unsupported provider {provider}")  # pragma: no cover - guarded in config


def create_llm_safe(settings: Settings) -> tuple[LLMClient | None, str]:
    """Like :func:`create_llm` but never raises; returns (client, note)."""
    try:
        client = create_llm(settings)
    except (LLMError, ValueError) as exc:
        return None, f"LLM disabled ({exc}); running in offline deterministic mode"
    if client is None:
        return None, "offline deterministic mode"
    return client, f"{client.name}:{client.model}"


class ScriptedLLM(LLMClient):
    """Deterministic fake LLM for tests and dry-runs.

    ``responses`` maps a *marker* (a substring searched in the system prompt)
    to either a string, a dict (serialised to JSON) or a callable receiving
    the prompt. Unmatched calls return ``default``.
    """

    name = "scripted"
    model = "scripted-v1"

    def __init__(
        self,
        responses: dict[str, str | dict[str, Any] | Callable[[str], str]] | None = None,
        default: str | dict[str, Any] | None = None,
        fail: bool = False,
        max_retries: int = 1,
    ) -> None:
        super().__init__(max_retries=max_retries)
        self.responses = responses or {}
        self.default = default
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def _complete(self, system: str, prompt: str, json_mode: bool) -> str:
        self.calls.append((system, prompt))
        if self.fail:
            raise LLMError("scripted failure")
        for marker, response in self.responses.items():
            if marker in system:
                return self._render(response, prompt)
        if self.default is None:
            raise LLMError("no scripted response")
        return self._render(self.default, prompt)

    @staticmethod
    def _render(response: str | dict[str, Any] | Callable[[str], str], prompt: str) -> str:
        if callable(response):
            return response(prompt)
        if isinstance(response, dict):
            return json.dumps(response)
        return response
