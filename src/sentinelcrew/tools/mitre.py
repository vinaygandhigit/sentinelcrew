"""MITRE ATT&CK technique catalogue and keyword mapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Technique:
    technique_id: str
    name: str
    tactic: str
    keywords: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, str]:
        return {"id": self.technique_id, "name": self.name, "tactic": self.tactic}


class MitreMapper:
    def __init__(self, techniques: Iterable[Technique]) -> None:
        self.techniques = {t.technique_id: t for t in techniques}

    @classmethod
    def from_file(cls, path: Path) -> "MitreMapper":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MitreMapper":
        return cls(
            Technique(
                technique_id=t["id"],
                name=t["name"],
                tactic=t["tactic"],
                keywords=tuple(k.lower() for k in t.get("keywords", [])),
            )
            for t in data.get("techniques", [])
        )

    def get(self, technique_id: str) -> Technique | None:
        return self.techniques.get(technique_id)

    def map_text(self, text: str, limit: int = 5) -> list[Technique]:
        """Rank techniques by the number of keyword hits in ``text``."""
        lowered = (text or "").lower()
        scored: list[tuple[int, Technique]] = []
        for technique in self.techniques.values():
            hits = sum(1 for k in technique.keywords if k in lowered)
            if hits:
                scored.append((hits, technique))
        scored.sort(key=lambda item: (-item[0], item[1].technique_id))
        return [t for _, t in scored[:limit]]

    def describe(self, ids: Iterable[str]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []
        seen: set[str] = set()
        for technique_id in ids:
            if not technique_id or technique_id in seen:
                continue
            seen.add(technique_id)
            technique = self.get(technique_id)
            if technique:
                result.append(technique.as_dict())
            else:
                result.append({"id": technique_id, "name": "Unknown technique", "tactic": "unknown"})
        return result

    def primary_tactic(self, ids: Iterable[str]) -> str:
        for technique_id in ids:
            technique = self.get(technique_id)
            if technique:
                return technique.tactic
        return "unknown"
