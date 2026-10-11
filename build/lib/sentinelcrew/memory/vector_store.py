"""Case memory: a lightweight TF-IDF vector store with cosine similarity.

Gives the crew institutional memory ("this looks like CASE-0007, which was a
confirmed credential-stuffing attack and was contained by blocking the IP").
The :class:`Embedder` protocol allows swapping TF-IDF for neural embeddings
(e.g. Gemini / Voyage embeddings) without touching the agents.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]{1,}")
_STOP = {
    "the", "and", "for", "with", "from", "this", "that", "was", "were", "are", "has", "have",
    "into", "via", "onto", "but", "not", "any", "all", "out", "its", "per", "after", "over",
}


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall((text or "").lower()):
        token = token.strip(".-_")
        if len(token) > 1 and token not in _STOP:
            tokens.append(token)
    return tokens


class Embedder(Protocol):
    def fit(self, documents: list[str]) -> None: ...
    def embed(self, text: str) -> dict[str, float]: ...


class TfidfEmbedder:
    """Sparse TF-IDF embedding (dict-of-weights, L2 normalised)."""

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.n_docs = 0

    def fit(self, documents: list[str]) -> None:
        self.n_docs = len(documents)
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        self.idf = {t: math.log((1 + self.n_docs) / (1 + c)) + 1.0 for t, c in df.items()}

    def embed(self, text: str) -> dict[str, float]:
        counts = Counter(tokenize(text))
        if not counts:
            return {}
        default_idf = math.log(1 + self.n_docs) + 1.0
        vec = {t: (1 + math.log(c)) * self.idf.get(t, default_idf) for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(t, 0.0) for t, w in a.items())


@dataclass
class MemoryRecord:
    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: dict[str, float] = field(default_factory=dict)


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.records: dict[str, MemoryRecord] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record_id: str, text: str, metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.records[record_id] = MemoryRecord(record_id, text, metadata or {})
            self._reindex()

    def add_many(self, items: list[tuple[str, str, dict[str, Any]]]) -> None:
        with self._lock:
            for record_id, text, metadata in items:
                self.records[record_id] = MemoryRecord(record_id, text, metadata)
            self._reindex()

    def _reindex(self) -> None:
        self.embedder.fit([r.text for r in self.records.values()])
        for record in self.records.values():
            record.vector = self.embedder.embed(record.text)

    def search(
        self, text: str, k: int = 3, min_score: float = 0.05, exclude: set[str] | None = None
    ) -> list[tuple[MemoryRecord, float]]:
        exclude = exclude or set()
        with self._lock:
            query = self.embedder.embed(text)
            scored = [
                (record, cosine(query, record.vector))
                for record in self.records.values()
                if record.record_id not in exclude
            ]
        scored = [(r, s) for r, s in scored if s >= min_score]
        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        return scored[:k]
