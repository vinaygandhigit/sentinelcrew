"""Threat-intelligence lookups against a local intel feed.

The MVP ships with a curated local feed (``data/intel/feed.json``) so that it
works fully offline and without third-party accounts. The ``ThreatIntelSource``
protocol makes it easy to plug in external feeds (e.g. MISP, OTX) later.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from sentinelcrew.models import IndicatorVerdict

_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HASH_RE = re.compile(r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b")
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.I)
# Restricting to known TLDs avoids false IOCs such as usernames ("j.mehta") or files ("x.dll").
KNOWN_TLDS = {
    "com", "net", "org", "io", "co", "xyz", "info", "biz", "us", "in", "uk", "de", "ru", "cn", "top",
    "site", "online", "app", "dev", "cloud", "gov", "edu", "me", "tk", "cc", "ws", "su", "ir", "kp",
    "club", "live", "shop", "link", "click", "pw", "ly", "to", "ai",
}


def classify_indicator(value: str) -> str:
    """Return ``ip``, ``hash``, ``domain`` or ``unknown``."""
    value = value.strip()
    try:
        ipaddress.ip_address(value)
        return "ip"
    except ValueError:
        pass
    if _HASH_RE.fullmatch(value):
        return "hash"
    if _DOMAIN_RE.fullmatch(value) and value.rsplit(".", 1)[-1].lower() in KNOWN_TLDS:
        return "domain"
    return "unknown"


def is_internal_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local


def extract_indicators(texts: Iterable[str]) -> dict[str, set[str]]:
    """Regex-extract IOCs from free text."""
    found: dict[str, set[str]] = {"ip": set(), "domain": set(), "hash": set()}
    for text in texts:
        if not text:
            continue
        for ip in _IP_RE.findall(text):
            if classify_indicator(ip) == "ip":
                found["ip"].add(ip)
        for h in _HASH_RE.findall(text):
            found["hash"].add(h.lower())
        for d in _DOMAIN_RE.findall(text):
            if classify_indicator(d) == "domain" and not _IP_RE.fullmatch(d):
                found["domain"].add(d.lower())
    return found


class ThreatIntelSource(Protocol):
    name: str

    def lookup(self, indicator: str, kind: str) -> IndicatorVerdict | None: ...


@dataclass
class LocalIntelFeed:
    """Lookup table loaded from JSON: ``{"indicators": [...], "allowlist": [...]}``."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    allowlist: set[str] = field(default_factory=set)
    name: str = "local-feed"

    @classmethod
    def from_file(cls, path: Path) -> "LocalIntelFeed":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocalIntelFeed":
        entries = {str(e["indicator"]).lower(): e for e in data.get("indicators", [])}
        allow = {str(a).lower() for a in data.get("allowlist", [])}
        return cls(entries=entries, allowlist=allow)

    def is_allowlisted(self, indicator: str) -> bool:
        value = indicator.lower()
        if value in self.allowlist:
            return True
        return any(value.endswith("." + a) for a in self.allowlist if "." in a)

    def lookup(self, indicator: str, kind: str) -> IndicatorVerdict | None:
        value = indicator.lower()
        if self.is_allowlisted(value):
            return IndicatorVerdict(
                indicator=indicator,
                kind=kind,
                reputation=0.0,
                malicious=False,
                tags=["allowlisted"],
                sources=[self.name],
                context="Indicator is on the organisation allowlist.",
            )
        entry = self.entries.get(value)
        if entry is None and kind == "domain":
            # parent-domain match (sub.evil.com -> evil.com)
            parts = value.split(".")
            for i in range(1, len(parts) - 1):
                entry = self.entries.get(".".join(parts[i:]))
                if entry:
                    break
        if entry is None:
            return None
        reputation = float(entry.get("reputation", 0.5))
        return IndicatorVerdict(
            indicator=indicator,
            kind=kind,
            reputation=max(0.0, min(1.0, reputation)),
            malicious=reputation >= 0.7,
            tags=list(entry.get("tags", [])),
            sources=[entry.get("source", self.name)],
            context=entry.get("context", ""),
        )


class ThreatIntelService:
    """Aggregates one or more intel sources and adds internal-IP context."""

    def __init__(self, sources: list[ThreatIntelSource]) -> None:
        self.sources = sources

    def lookup(self, indicator: str, kind: str | None = None) -> IndicatorVerdict:
        kind = kind or classify_indicator(indicator)
        if kind == "ip" and is_internal_ip(indicator):
            return IndicatorVerdict(
                indicator=indicator,
                kind=kind,
                reputation=0.0,
                malicious=False,
                tags=["internal"],
                sources=["rfc1918"],
                context="Private / internal address space.",
            )
        hits = [v for v in (s.lookup(indicator, kind) for s in self.sources) if v]
        if not hits:
            return IndicatorVerdict(
                indicator=indicator,
                kind=kind,
                reputation=0.1 if kind == "ip" else 0.15,
                malicious=False,
                tags=["unknown"],
                sources=[],
                context="No intelligence on this indicator (unknown / rare).",
            )
        if any("allowlisted" in h.tags for h in hits):
            return next(h for h in hits if "allowlisted" in h.tags)
        best = max(hits, key=lambda h: h.reputation)
        best.sources = sorted({s for h in hits for s in h.sources})
        best.tags = sorted({t for h in hits for t in h.tags})
        return best

    def is_allowlisted(self, indicator: str) -> bool:
        return any(
            getattr(source, "is_allowlisted", lambda _x: False)(indicator) for source in self.sources
        )