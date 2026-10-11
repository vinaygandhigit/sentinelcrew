"""Parsers for the supported input formats: JSON, JSON-Lines, CSV and syslog."""

from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sentinelcrew.ingestion.normalizer import NormalizationError, normalize_alert, normalize_event
from sentinelcrew.models import Alert, LogEvent

# RFC-5424 style:  <34>1 2026-10-08T10:00:00Z host app - - - key=value key2="v 2"
_SYSLOG_5424 = re.compile(
    r"^<(?P<pri>\d{1,3})>\d\s+(?P<ts>\S+)\s+(?P<host>\S+)\s+(?P<app>\S+)\s+\S+\s+\S+\s+\S+\s*(?P<msg>.*)$"
)
# RFC-3164 style:  <34>Oct  8 10:00:00 host app: key=value
_SYSLOG_3164 = re.compile(
    r"^(?:<(?P<pri>\d{1,3})>)?(?P<ts>[A-Z][a-z]{2}\s+\d{1,2}\s\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<app>[^:\s]+):?\s*(?P<msg>.*)$"
)
_KV = re.compile(r'(\w[\w.]*)=("([^"]*)"|\S+)')


def parse_kv(text: str) -> dict[str, str]:
    """Parse ``key=value key2="quoted value"`` pairs."""
    result: dict[str, str] = {}
    for match in _KV.finditer(text):
        key = match.group(1)
        value = match.group(3) if match.group(3) is not None else match.group(2)
        result[key] = value
    return result


def parse_syslog_line(line: str, year: int | None = None) -> dict[str, Any]:
    """Parse one syslog line into a flat record (key/value payload merged)."""
    line = line.strip()
    match = _SYSLOG_5424.match(line)
    if match:
        timestamp: Any = match.group("ts")
    else:
        match = _SYSLOG_3164.match(line)
        if not match:
            raise NormalizationError(f"not a syslog line: {line[:60]!r}")
        year = year or datetime.now(timezone.utc).year
        parsed = datetime.strptime(f"{year} {match.group('ts')}", "%Y %b %d %H:%M:%S")
        timestamp = parsed.replace(tzinfo=timezone.utc).isoformat()
    record: dict[str, Any] = {
        "timestamp": timestamp,
        "host": match.group("host"),
        "source": match.group("app"),
        "message": match.group("msg"),
    }
    record.update(parse_kv(match.group("msg")))
    return record


def _records_from_text(text: str, fmt: str) -> list[dict[str, Any]]:
    fmt = fmt.lower()
    if fmt == "json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("alerts", "events", "logs", "records"):
                if isinstance(data.get(key), list):
                    return list(data[key])
            return [data]
        if isinstance(data, list):
            return list(data)
        raise NormalizationError("JSON must be an object or a list")
    if fmt in ("jsonl", "ndjson"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if fmt == "csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if fmt == "syslog":
        return [parse_syslog_line(line) for line in text.splitlines() if line.strip()]
    raise NormalizationError(f"unsupported format '{fmt}'")


def detect_format(text: str, filename: str | None = None) -> str:
    if filename:
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix in ("json", "jsonl", "ndjson", "csv"):
            return suffix
        if suffix in ("log", "syslog"):
            return "syslog"
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        first_line = stripped.splitlines()[0].strip() if stripped else ""
        if first_line.startswith("{") and first_line.endswith("}") and len(stripped.splitlines()) > 1:
            try:
                json.loads(stripped)
            except json.JSONDecodeError:
                return "jsonl"
        return "json"
    if stripped.startswith("<") or _SYSLOG_3164.match(stripped.splitlines()[0] if stripped else ""):
        return "syslog"
    return "csv"


def parse_records(text: str, fmt: str | None = None, filename: str | None = None) -> list[dict[str, Any]]:
    return _records_from_text(text, fmt or detect_format(text, filename))


def parse_alerts(
    text: str, fmt: str | None = None, filename: str | None = None
) -> tuple[list[Alert], list[str]]:
    """Parse and normalise alerts. Returns ``(alerts, errors)`` - bad rows are skipped."""
    return _normalize_many(parse_records(text, fmt, filename), normalize_alert)  # type: ignore[return-value]


def parse_events(
    text: str, fmt: str | None = None, filename: str | None = None
) -> tuple[list[LogEvent], list[str]]:
    """Parse and normalise log events. Returns ``(events, errors)``."""
    return _normalize_many(parse_records(text, fmt, filename), normalize_event)  # type: ignore[return-value]


def _normalize_many(records: Iterable[dict[str, Any]], fn: Any) -> tuple[list[Any], list[str]]:
    items: list[Any] = []
    errors: list[str] = []
    for index, record in enumerate(records):
        try:
            items.append(fn(record))
        except (NormalizationError, TypeError, ValueError) as exc:
            errors.append(f"record {index}: {exc}")
    return items, errors
