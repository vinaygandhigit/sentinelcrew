"""Ingestion layer: parse -> normalise."""

from sentinelcrew.ingestion.normalizer import (
    NormalizationError,
    normalize_alert,
    normalize_event,
    normalize_severity,
    parse_timestamp,
)
from sentinelcrew.ingestion.parsers import (
    detect_format,
    parse_alerts,
    parse_events,
    parse_kv,
    parse_records,
    parse_syslog_line,
)

__all__ = [
    "NormalizationError",
    "detect_format",
    "normalize_alert",
    "normalize_event",
    "normalize_severity",
    "parse_alerts",
    "parse_events",
    "parse_kv",
    "parse_records",
    "parse_syslog_line",
    "parse_timestamp",
]