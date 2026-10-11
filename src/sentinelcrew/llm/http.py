"""Tiny HTTP helper used by the LLM providers.

Using the standard library keeps the provider layer free of heavyweight SDK
dependencies and - more importantly - makes it trivial to inject a fake
transport in unit tests.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol


class TransportError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class Transport(Protocol):
    def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]: ...


RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def urllib_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("content-type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))  # type: ignore[no-any-return]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise TransportError(
            f"HTTP {exc.code}: {detail}", status=exc.code, retryable=exc.code in RETRYABLE_STATUS
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportError(f"network error: {exc}", retryable=True) from exc
    except json.JSONDecodeError as exc:
        raise TransportError(f"invalid JSON from server: {exc}") from exc


def post_with_retry(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    retries: int,
    backoff_s: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Call ``transport`` with exponential back-off on retryable failures."""
    attempt = 0
    while True:
        try:
            return transport(url, headers, payload, timeout)
        except TransportError as exc:
            if not exc.retryable or attempt >= retries:
                raise
            sleep(backoff_s * (2**attempt))
            attempt += 1
