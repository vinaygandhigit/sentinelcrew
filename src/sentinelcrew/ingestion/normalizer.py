"""Normalise heterogeneous alert / log records into the canonical model.

Different tools name the same field differently (``src``, ``source_ip``,
``SourceIp``, ``client.ip`` ...). The normaliser maps known aliases to the
canonical field names, parses timestamps and coerces types.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Mapping

from sentinelcrew.models import Alert, LogEvent

ALIASES: dict[str, tuple[str, ...]] = {
    "alert_id": ("alert_id", "id", "alertid", "event_id", "uuid"),
    "timestamp": ("timestamp", "time", "@timestamp", "ts", "event_time", "created", "date"),
    "source": ("source", "product", "sensor", "detector", "log_source"),
    "rule_name": ("rule_name", "rule", "name", "signature", "title", "alert_name"),
    "description": ("description", "message", "msg", "summary", "details_text"),
    "severity_hint": ("severity", "severity_hint", "level", "priority"),
    "host": ("host", "hostname", "computer", "device", "computer_name", "workstation"),
    "user": ("user", "username", "user_name", "account", "target_user", "targetusername"),
    "src_ip": ("src_ip", "source_ip", "src", "sourceip", "client_ip", "ipaddress", "srcaddr",
               "source.ip", "client.ip", "src.ip"),
    "dst_ip": ("dst_ip", "destination_ip", "dst", "destip", "dest_ip", "dstaddr", "remote_ip",
               "destination.ip", "server.ip", "dst.ip"),
    "dst_port": ("dst_port", "destination_port", "dport", "dest_port", "port"),
    "process": ("process", "process_name", "image", "proc", "command_line", "commandline"),
    "file_hash": ("file_hash", "hash", "sha256", "md5", "sha1"),
    "domain": ("domain", "dns_query", "query", "hostname_requested", "url_domain", "fqdn"),
    "event_type": ("event_type", "type", "category", "action_type", "eventtype"),
    "bytes_out": ("bytes_out", "bytes_sent", "sent_bytes", "out_bytes", "bytes"),
    "status": ("status", "outcome", "result", "action"),
}

SEVERITY_WORDS = {
    "critical": "critical",
    "crit": "critical",
    "high": "high",
    "medium": "medium",
    "med": "medium",
    "moderate": "medium",
    "low": "low",
    "info": "info",
    "informational": "info",
}


class NormalizationError(ValueError):
    """Raised when a record cannot be normalised."""


def _flatten(record: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        name = f"{prefix}{key}".lower()
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        else:
            flat[name] = value
    return flat


def _lookup(flat: Mapping[str, Any], field: str) -> Any:
    for alias in ALIASES[field]:
        if alias in flat and flat[alias] not in (None, ""):
            return flat[alias]
    # other dotted keys such as "event.user" -> match on the last segment
    for key, value in flat.items():
        if "." in key and key.rsplit(".", 1)[-1] in ALIASES[field] and value not in (None, ""):
            return value
    return None


def parse_timestamp(value: Any) -> datetime:
    """Parse ISO-8601 strings, ``Z`` suffixes, epoch seconds / millis."""
    if value is None or value == "":
        raise NormalizationError("missing timestamp")
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = value / 1000 if value > 1e11 else value
        dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.replace(".", "", 1).isdigit():
            return parse_timestamp(float(text))
        text = text.replace("Z", "+00:00").replace(" ", "T", 1)
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise NormalizationError(f"unparseable timestamp {value!r}") from exc
    else:
        raise NormalizationError(f"unsupported timestamp type {type(value).__name__}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_severity(value: Any) -> str:
    if value is None:
        return "medium"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # common 0-10 / 1-5 numeric scales
        scale = value if value > 5 else value * 2
        if scale >= 9:
            return "critical"
        if scale >= 7:
            return "high"
        if scale >= 4:
            return "medium"
        if scale >= 2:
            return "low"
        return "info"
    return SEVERITY_WORDS.get(str(value).strip().lower(), "medium")


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _stable_id(prefix: str, record: Mapping[str, Any]) -> str:
    digest = hashlib.sha1(repr(sorted(record.items())).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def normalize_alert(record: Mapping[str, Any]) -> Alert:
    """Convert a raw alert dictionary into an :class:`Alert`."""
    if not isinstance(record, Mapping) or not record:
        raise NormalizationError("alert record must be a non-empty mapping")
    flat = _flatten(record)
    rule = _str(_lookup(flat, "rule_name"))
    if not rule:
        raise NormalizationError("alert has no rule/signature name")
    alert_id = _str(_lookup(flat, "alert_id")) or _stable_id("ALR", flat)
    return Alert(
        alert_id=alert_id,
        timestamp=parse_timestamp(_lookup(flat, "timestamp")),
        source=_str(_lookup(flat, "source")) or "unknown",
        rule_name=rule,
        description=_str(_lookup(flat, "description")) or "",
        severity_hint=normalize_severity(_lookup(flat, "severity_hint")),
        host=_str(_lookup(flat, "host")),
        user=_str(_lookup(flat, "user")),
        src_ip=_str(_lookup(flat, "src_ip")),
        dst_ip=_str(_lookup(flat, "dst_ip")),
        dst_port=_to_int(_lookup(flat, "dst_port")),
        process=_str(_lookup(flat, "process")),
        file_hash=_str(_lookup(flat, "file_hash")),
        domain=_str(_lookup(flat, "domain")),
        raw=dict(record),
    )


def normalize_event(record: Mapping[str, Any]) -> LogEvent:
    """Convert a raw telemetry record into a :class:`LogEvent`."""
    if not isinstance(record, Mapping) or not record:
        raise NormalizationError("log record must be a non-empty mapping")
    flat = _flatten(record)
    event_type = _str(_lookup(flat, "event_type"))
    if not event_type:
        raise NormalizationError("log record has no event_type")
    known = {alias for aliases in ALIASES.values() for alias in aliases}
    details = {k: v for k, v in record.items() if k.lower() not in known}
    return LogEvent(
        event_id=_str(_lookup(flat, "alert_id")) or _stable_id("EVT", flat),
        timestamp=parse_timestamp(_lookup(flat, "timestamp")),
        event_type=event_type.lower(),
        host=_str(_lookup(flat, "host")),
        user=_str(_lookup(flat, "user")),
        src_ip=_str(_lookup(flat, "src_ip")),
        dst_ip=_str(_lookup(flat, "dst_ip")),
        dst_port=_to_int(_lookup(flat, "dst_port")),
        process=_str(_lookup(flat, "process")),
        domain=_str(_lookup(flat, "domain")),
        bytes_out=_to_int(_lookup(flat, "bytes_out")) or 0,
        status=(_str(_lookup(flat, "status")) or "").lower() or None,
        details=details,
    )
