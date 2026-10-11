"""Organisational context: asset inventory, identities and change calendar.

The Critic agent uses this context to test *benign explanations* - e.g. "is
this source an authorised vulnerability scanner?" or "was there an approved
maintenance window?".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sentinelcrew.ingestion.normalizer import parse_timestamp

CRITICALITY_SCORE = {"critical": 1.0, "high": 0.75, "medium": 0.5, "low": 0.25}


@dataclass
class Asset:
    hostname: str
    ip: str | None = None
    role: str = "workstation"
    owner: str = ""
    criticality: str = "medium"
    tags: list[str] = field(default_factory=list)

    @property
    def criticality_score(self) -> float:
        return CRITICALITY_SCORE.get(self.criticality, 0.5)


@dataclass
class Identity:
    username: str
    department: str = ""
    privileged: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class ChangeWindow:
    change_id: str
    start: datetime
    end: datetime
    description: str
    assets: list[str] = field(default_factory=list)  # hostnames or IPs

    def covers(self, when: datetime, entity: str | None) -> bool:
        if not (self.start <= when <= self.end):
            return False
        if not self.assets:
            return True
        return bool(entity) and entity.lower() in {a.lower() for a in self.assets}


class OrgContext:
    def __init__(
        self,
        assets: list[Asset] | None = None,
        identities: list[Identity] | None = None,
        changes: list[ChangeWindow] | None = None,
    ) -> None:
        self.assets = assets or []
        self.identities = {i.username.lower(): i for i in (identities or [])}
        self.changes = changes or []
        self._by_host = {a.hostname.lower(): a for a in self.assets}
        self._by_ip = {a.ip: a for a in self.assets if a.ip}

    @classmethod
    def from_file(cls, path: Path) -> "OrgContext":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrgContext":
        assets = [Asset(**a) for a in data.get("assets", [])]
        identities = [Identity(**i) for i in data.get("identities", [])]
        changes = [
            ChangeWindow(
                change_id=c["change_id"],
                start=parse_timestamp(c["start"]),
                end=parse_timestamp(c["end"]),
                description=c.get("description", ""),
                assets=list(c.get("assets", [])),
            )
            for c in data.get("changes", [])
        ]
        return cls(assets, identities, changes)

    def asset(self, host_or_ip: str | None) -> Asset | None:
        if not host_or_ip:
            return None
        return self._by_host.get(host_or_ip.lower()) or self._by_ip.get(host_or_ip)

    def identity(self, username: str | None) -> Identity | None:
        if not username:
            return None
        return self.identities.get(username.lower())

    def active_changes(self, when: datetime, *entities: str | None) -> list[ChangeWindow]:
        hits: list[ChangeWindow] = []
        for change in self.changes:
            if any(change.covers(when, e) for e in entities if e) or (
                not change.assets and change.covers(when, None)
            ):
                hits.append(change)
        return hits

    def has_tag(self, host_or_ip: str | None, tag: str) -> bool:
        asset = self.asset(host_or_ip)
        return bool(asset and tag in asset.tags)
