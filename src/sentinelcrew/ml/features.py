"""Behavioural feature extraction for the Triage model.

Features are computed from the alert *plus* the surrounding telemetry
window so that the model scores behaviour, not just the alert label.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Iterable

from sentinelcrew.models import Alert, LogEvent

FEATURE_NAMES: tuple[str, ...] = (
    "failed_auth_count",
    "distinct_users_from_src",
    "success_after_failures",
    "off_hours",
    "external_src",
    "mb_out",
    "distinct_dst_ports",
    "beacon_regularity",
    "suspicious_process",
    "severity_hint",
    "asset_criticality",
)

SEVERITY_VALUE = {"info": 0.0, "low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}

SUSPICIOUS_PROCESS_MARKERS = (
    "powershell -enc",
    " -enc ",
    "-w hidden",
    "powershell.exe -e",
    "-encodedcommand",
    "rundll32",
    "regsvr32",
    "mshta",
    "certutil -urlcache",
    "bitsadmin",
    "mimikatz",
    "procdump",
    "vssadmin delete",
    "wmic shadowcopy",
    "psexec",
    "nc.exe",
    "cobalt",
    "beacon",
)

MIN_FAILURES_FOR_COMPROMISE = 5
BEACON_MAX_BYTES = 100_000

FAILED_STATUSES = {"failure", "failed", "deny", "denied", "fail"}
SUCCESS_STATUSES = {"success", "succeeded", "ok", "allowed", "allow"}


def _is_external(ip: str | None) -> bool:
    from sentinelcrew.tools.threat_intel import classify_indicator, is_internal_ip

    return bool(ip) and classify_indicator(ip) == "ip" and not is_internal_ip(ip)  # type: ignore[arg-type]


def is_off_hours(ts: datetime, start_hour: int = 7, end_hour: int = 20) -> bool:
    """Weekend or outside 07:00-20:00 (UTC by default)."""
    return ts.weekday() >= 5 or not (start_hour <= ts.hour < end_hour)


def beacon_regularity(timestamps: Iterable[datetime]) -> float:
    """1.0 for perfectly periodic connections, ~0 for random ones.

    Uses ``1 - coefficient_of_variation`` of the inter-arrival times, which is
    the classic C2-beacon heuristic. Needs at least 4 connections.
    """
    times = sorted(timestamps)
    if len(times) < 4:
        return 0.0
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    mean = statistics.fmean(gaps)
    if mean <= 0:
        return 0.0
    cv = statistics.pstdev(gaps) / mean
    return max(0.0, min(1.0, 1.0 - cv))


def is_suspicious_process(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in SUSPICIOUS_PROCESS_MARKERS)


def extract_features(
    alert: Alert, events: list[LogEvent], asset_criticality: float = 0.5
) -> dict[str, float]:
    auth = [e for e in events if e.event_type == "auth"]
    failed = [e for e in auth if (e.status or "") in FAILED_STATUSES]
    succeeded = [e for e in auth if (e.status or "") in SUCCESS_STATUSES]
    first_failure = min((e.timestamp for e in failed), default=None)
    success_after = any(first_failure and e.timestamp > first_failure for e in succeeded)

    src_users = {e.user for e in auth if alert.src_ip and e.src_ip == alert.src_ip and e.user}

    network = [e for e in events if e.event_type in ("network", "dns", "proxy")]
    mb_out = sum(e.bytes_out for e in events) / 1_000_000
    ports = {e.dst_port for e in network if e.dst_port}

    by_dest: dict[str, list[datetime]] = {}
    for e in network:
        if e.bytes_out > BEACON_MAX_BYTES:
            continue  # beacons are small check-ins; bulk transfers are scored separately
        key = e.dst_ip or e.domain
        if key:
            by_dest.setdefault(key, []).append(e.timestamp)
    regularity = max((beacon_regularity(ts) for ts in by_dest.values()), default=0.0)

    process_texts = [alert.process] + [e.process for e in events if e.event_type == "process"]
    suspicious = any(is_suspicious_process(p) for p in process_texts)

    external = _is_external(alert.src_ip) or any(_is_external(e.src_ip) for e in failed + succeeded)

    return {
        "failed_auth_count": float(len(failed)),
        "distinct_users_from_src": float(len(src_users)),
        "success_after_failures": 1.0 if success_after and len(failed) >= MIN_FAILURES_FOR_COMPROMISE else 0.0,
        "off_hours": 1.0 if is_off_hours(alert.timestamp) else 0.0,
        "external_src": 1.0 if external else 0.0,
        "mb_out": round(mb_out, 3),
        "distinct_dst_ports": float(len(ports)),
        "beacon_regularity": round(regularity, 3),
        "suspicious_process": 1.0 if suspicious else 0.0,
        "severity_hint": SEVERITY_VALUE.get(alert.severity_hint, 0.5),
        "asset_criticality": float(asset_criticality),
    }


def to_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]
