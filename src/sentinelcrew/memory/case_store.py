"""SQLite persistence for cases, actions and analyst decisions."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sentinelcrew.models import Case

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id      TEXT PRIMARY KEY,
    alert_id     TEXT NOT NULL,
    title        TEXT NOT NULL,
    status       TEXT NOT NULL,
    verdict      TEXT NOT NULL,
    priority     TEXT NOT NULL,
    confidence   REAL NOT NULL,
    initial_confidence REAL,
    duration_ms  REAL NOT NULL,
    llm_provider TEXT,
    created_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_created ON cases(created_at);
CREATE TABLE IF NOT EXISTS decisions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id    TEXT NOT NULL,
    action_id  TEXT NOT NULL,
    decision   TEXT NOT NULL,
    analyst    TEXT NOT NULL,
    comment    TEXT,
    decided_at TEXT NOT NULL
);
"""


class CaseStore:
    """Thread-safe SQLite store. Use ``":memory:"`` for tests."""

    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._tx() as cur:
            cur.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                cur.close()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def save(self, case: Case) -> None:
        payload = case.to_dict()
        initial = case.initial_assessment.confidence if case.initial_assessment else None
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO cases (case_id, alert_id, title, status, verdict, priority, confidence,
                       initial_confidence, duration_ms, llm_provider, created_at, payload)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(case_id) DO UPDATE SET
                       status=excluded.status, verdict=excluded.verdict, priority=excluded.priority,
                       confidence=excluded.confidence, payload=excluded.payload""",
                (
                    case.case_id,
                    case.alert.alert_id,
                    case.alert.title(),
                    case.status.value,
                    case.verdict.value,
                    case.priority.value,
                    case.confidence,
                    initial,
                    case.duration_ms,
                    case.llm_provider,
                    case.created_at.isoformat(),
                    json.dumps(payload),
                ),
            )

    def get(self, case_id: str) -> dict[str, Any] | None:
        with self._tx() as cur:
            row = cur.execute("SELECT payload FROM cases WHERE case_id=?", (case_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._tx() as cur:
            rows = cur.execute(
                """SELECT case_id, alert_id, title, status, verdict, priority, confidence,
                          initial_confidence, duration_ms, llm_provider, created_at
                   FROM cases ORDER BY created_at DESC, case_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._tx() as cur:
            return int(cur.execute("SELECT COUNT(*) FROM cases").fetchone()[0])

    def record_decision(
        self, case_id: str, action_id: str, decision: str, analyst: str, comment: str, when: str
    ) -> None:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO decisions (case_id, action_id, decision, analyst, comment, decided_at)"
                " VALUES (?,?,?,?,?,?)",
                (case_id, action_id, decision, analyst, comment, when),
            )

    def decisions(self, case_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM decisions"
        params: tuple[Any, ...] = ()
        if case_id:
            query += " WHERE case_id=?"
            params = (case_id,)
        with self._tx() as cur:
            return [dict(r) for r in cur.execute(query + " ORDER BY id", params).fetchall()]

    def update_payload(self, case_id: str, payload: dict[str, Any]) -> None:
        with self._tx() as cur:
            cur.execute(
                "UPDATE cases SET payload=?, status=? WHERE case_id=?",
                (json.dumps(payload), payload.get("status"), case_id),
            )

    def metrics(self) -> dict[str, Any]:
        with self._tx() as cur:
            rows = cur.execute(
                "SELECT verdict, status, confidence, initial_confidence, duration_ms FROM cases"
            ).fetchall()
            pending = cur.execute("SELECT payload FROM cases WHERE status='awaiting_approval'").fetchall()
        total = len(rows)
        by_verdict: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for row in rows:
            by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + 1
            by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        pending_actions = 0
        for row in pending:
            payload = json.loads(row["payload"])
            pending_actions += sum(
                1 for a in payload.get("actions", []) if a.get("status") == "pending_approval"
            )
        auto_closed = by_verdict.get("false_positive", 0) + by_verdict.get("noise", 0)
        shifts = [
            abs(r["confidence"] - r["initial_confidence"])
            for r in rows
            if r["initial_confidence"] is not None
        ]
        return {
            "total_cases": total,
            "by_verdict": by_verdict,
            "by_status": by_status,
            "auto_closed": auto_closed,
            "auto_close_rate": round(auto_closed / total, 3) if total else 0.0,
            "mean_time_to_triage_ms": round(sum(r["duration_ms"] for r in rows) / total, 1) if total else 0.0,
            "mean_confidence": round(sum(r["confidence"] for r in rows) / total, 3) if total else 0.0,
            "mean_debate_shift": round(sum(shifts) / len(shifts), 3) if shifts else 0.0,
            "pending_approvals": pending_actions,
        }
