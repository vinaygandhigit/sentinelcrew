"""Provider-agnostic LLM interface.

Every provider implements :meth:`LLMClient.complete` (text in, text out).
:meth:`LLMClient.complete_json` adds robust JSON extraction, schema
validation and a corrective retry so that agents always receive structured,
validated output or a clean :class:`LLMError` they can fall back from.
"""

from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sentinelcrew.validation import SchemaError, validate


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable answer."""


@dataclass
class LLMUsage:
    calls: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    input_chars: int = 0
    output_chars: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, latency_ms: float, prompt: str, output: str, ok: bool) -> None:
        with self._lock:
            self.calls += 1
            self.total_latency_ms += latency_ms
            self.input_chars += len(prompt)
            self.output_chars += len(output)
            if not ok:
                self.failures += 1

    def as_dict(self) -> dict[str, float]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "input_chars": self.input_chars,
            "output_chars": self.output_chars,
        }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response.

    Handles fenced code blocks, leading prose and trailing commentary.
    Raises ``ValueError`` when no valid JSON can be found.
    """
    if text is None:
        raise ValueError("empty response")
    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
        parsed = _scan_balanced(candidate)
        if parsed is not None:
            return parsed
    raise ValueError("no JSON object found in response")


def _scan_balanced(text: str) -> Any:
    """Find the first balanced ``{...}`` / ``[...]`` block that parses."""
    for start, char in enumerate(text):
        if char not in "{[":
            continue
        closing = "}" if char == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            c = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == char:
                depth += 1
            elif c == closing:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
    return None


class LLMClient(ABC):
    """Abstract base for all LLM providers."""

    name: str = "base"
    model: str = ""

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max(0, max_retries)
        self.usage = LLMUsage()

    @abstractmethod
    def _complete(self, system: str, prompt: str, json_mode: bool) -> str:
        """Provider specific call. Must raise :class:`LLMError` on failure."""

    def complete(self, system: str, prompt: str, json_mode: bool = False) -> str:
        start = time.perf_counter()
        output = ""
        ok = False
        try:
            output = self._complete(system, prompt, json_mode)
            ok = True
            return output
        finally:
            self.usage.record((time.perf_counter() - start) * 1000, system + prompt, output, ok)

    def complete_json(self, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a dict that is guaranteed to satisfy ``schema``.

        On malformed output the model is re-prompted with the validation
        error (up to ``max_retries`` times) before giving up.
        """
        schema_text = json.dumps(schema, indent=None)
        full_prompt = (
            f"{prompt}\n\nRespond ONLY with a single JSON object that matches this JSON schema:\n"
            f"{schema_text}"
        )
        last_error = "unknown error"
        attempt_prompt = full_prompt
        for _ in range(self.max_retries + 1):
            raw = self.complete(system, attempt_prompt, json_mode=True)
            try:
                data = extract_json(raw)
                validate(data, schema)
                return data  # type: ignore[no-any-return]
            except (ValueError, SchemaError) as exc:
                last_error = str(exc)
                attempt_prompt = (
                    f"{full_prompt}\n\nYour previous answer was invalid ({last_error}). "
                    "Return only corrected JSON."
                )
        raise LLMError(f"{self.name}: invalid structured output after retries: {last_error}")

    def describe(self) -> dict[str, str]:
        return {"provider": self.name, "model": self.model}
