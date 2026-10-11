"""In-memory, indexed log store used by the Investigator's ``query_logs`` tool.

Events are indexed by every pivotable entity (user, host, src_ip, dst_ip,
domain) so that correlation queries are O(matches) rather than O(all logs).
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable

from sentinelcrew.models import LogEvent

ENTITY_FIELDS = ("user", "host", "src_ip", "dst_ip", "domain")


class LogStore:
    def __init__(self, events: Iterable[LogEvent] = ()) -> None:
        self._events: list[LogEvent] = []
        self._by_id: dict[str, LogEvent] = {}
        self._index: dict[tuple[str, str], list[LogEvent]] = defaultdict(list)
        self._lock = threading.Lock()
        self.add_many(events)

    def __len__(self) -> int:
        return len(self._events)

    def add(self, event: LogEvent) -> None:
        self.add_many([event])

    def add_many(self, events: Iterable[LogEvent]) -> int:
        added = 0
        with self._lock:
            for event in events:
                if event.event_id in self._by_id:
                    continue
                self._by_id[event.event_id] = event
                self._insert_sorted(self._events, event)
                for field in ENTITY_FIELDS:
                    value = getattr(event, field)
                    if value:
                        self._insert_sorted(self._index[(field, value.lower())], event)
                added += 1
        return added

    @staticmethod
    def _insert_sorted(bucket: list[LogEvent], event: LogEvent) -> None:
        keys = [e.timestamp for e in bucket]
        bucket.insert(bisect.bisect_right(keys, event.timestamp), event)

    def get(self, event_id: str) -> LogEvent | None:
        return self._by_id.get(event_id)

    def all(self) -> list[LogEvent]:
        return list(self._events)

    def query(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        event_types: Iterable[str] | None = None,
        limit: int | None = None,
        **entities: str | None,
    ) -> list[LogEvent]:
        """Return events in ``[start, end]`` matching ANY of the given entities.

        Example: ``query(start, end, user="j.mehta", src_ip="1.2.3.4")`` returns
        events involving the user OR the IP - which is what a pivot needs.
        """
        wanted = {k: v for k, v in entities.items() if v}
        unknown = set(wanted) - set(ENTITY_FIELDS)
        if unknown:
            raise ValueError(f"unknown entity field(s): {sorted(unknown)}")
        if wanted:
            seen: dict[str, LogEvent] = {}
            for field, value in wanted.items():
                for event in self._index.get((field, value.lower()), []):
                    seen[event.event_id] = event
            candidates = sorted(seen.values(), key=lambda e: (e.timestamp, e.event_id))
        else:
            candidates = list(self._events)
        types = {t.lower() for t in event_types} if event_types else None
        result = [
            e
            for e in candidates
            if (start is None or e.timestamp >= start)
            and (end is None or e.timestamp <= end)
            and (types is None or e.event_type in types)
        ]
        return result[:limit] if limit else result

    def window(self, center: datetime, minutes: int, **entities: str | None) -> list[LogEvent]:
        delta = timedelta(minutes=minutes)
        return self.query(start=center - delta, end=center + delta, **entities)
