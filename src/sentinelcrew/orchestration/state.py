"""Shared, thread-safe state that flows through the orchestration graph."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from sentinelcrew.models import (
    AgentMessage,
    Alert,
    Assessment,
    DebateRound,
    IntelReport,
    Investigation,
    ResponseAction,
    SimilarCase,
    TriageResult,
)

MessageListener = Callable[[AgentMessage], None]


@dataclass
class CaseState:
    case_id: str
    alert: Alert
    triage: TriageResult | None = None
    investigation: Investigation | None = None
    intel: IntelReport | None = None
    similar_cases: list[SimilarCase] = field(default_factory=list)
    initial_assessment: Assessment | None = None
    assessment: Assessment | None = None
    debate: list[DebateRound] = field(default_factory=list)
    critic_adjustment: float = 0.0
    checks_done: set[str] = field(default_factory=set)
    debate_converged: bool = False
    actions: list[ResponseAction] = field(default_factory=list)
    report_markdown: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    listeners: list[MessageListener] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def emit(self, agent: str, kind: str, content: str, **data: Any) -> AgentMessage:
        message = AgentMessage(agent=agent, kind=kind, content=content, data=data)
        with self._lock:
            self.messages.append(message)
            listeners = list(self.listeners)
        for listener in listeners:
            try:
                listener(message)
            except Exception:  # noqa: BLE001 - a broken UI callback must never kill a case
                pass
        return message

    def record_error(self, where: str, error: str) -> None:
        with self._lock:
            self.errors.append(f"{where}: {error}")

    @property
    def debate_rounds(self) -> int:
        return len(self.debate)
