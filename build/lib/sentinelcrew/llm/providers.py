"""Concrete LLM providers: Google Gemini and Anthropic Claude.

Both talk to the vendors' public REST APIs. Choose one with the
``LLM_PROVIDER`` environment variable (see :mod:`sentinelcrew.config`).
"""

from __future__ import annotations

from typing import Any

from sentinelcrew.llm.base import LLMClient, LLMError
from sentinelcrew.llm.http import Transport, TransportError, post_with_retry, urllib_transport


class GeminiClient(LLMClient):
    """Google Gemini via the Generative Language REST API (testing / demo)."""

    name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries)
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.transport: Transport = transport or urllib_transport

    def build_payload(self, system: str, prompt: str, json_mode: bool) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": self.max_tokens,
        }
        if json_mode:
            config["responseMimeType"] = "application/json"
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    def _complete(self, system: str, prompt: str, json_mode: bool) -> str:
        url = f"{self.BASE_URL}/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        try:
            data = post_with_retry(
                self.transport,
                url,
                headers,
                self.build_payload(system, prompt, json_mode),
                self.timeout_s,
                self.max_retries,
            )
        except TransportError as exc:
            raise LLMError(f"gemini request failed: {exc}") from exc
        return self.parse_response(data)

    @staticmethod
    def parse_response(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise LLMError(f"gemini returned no candidates ({feedback})")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            reason = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"gemini returned empty text (finishReason={reason})")
        return text


class AnthropicClient(LLMClient):
    """Anthropic Claude via the Messages API (production)."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(max_retries=max_retries)
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.transport: Transport = transport or urllib_transport

    def build_payload(self, system: str, prompt: str, json_mode: bool) -> dict[str, Any]:
        user_content = prompt
        if json_mode:
            user_content = f"{prompt}\n\nReturn only valid JSON, with no prose before or after it."
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
        }

    def _complete(self, system: str, prompt: str, json_mode: bool) -> str:
        headers = {"x-api-key": self.api_key, "anthropic-version": self.API_VERSION}
        try:
            data = post_with_retry(
                self.transport,
                self.URL,
                headers,
                self.build_payload(system, prompt, json_mode),
                self.timeout_s,
                self.max_retries,
            )
        except TransportError as exc:
            raise LLMError(f"anthropic request failed: {exc}") from exc
        return self.parse_response(data)

    @staticmethod
    def parse_response(data: dict[str, Any]) -> str:
        if data.get("type") == "error":
            raise LLMError(f"anthropic error: {data.get('error')}")
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text.strip():
            raise LLMError(f"anthropic returned empty text (stop_reason={data.get('stop_reason')})")
        return text
