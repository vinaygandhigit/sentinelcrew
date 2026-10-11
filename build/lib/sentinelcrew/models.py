"""Core domain model shared by every agent.

All objects are plain dataclasses so that they are cheap to create, easy to
test and trivially serialisable to JSON (see :func:`to_jsonable`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Final classification of a case."""

    TRUE_POSITIVE = "true_positive"
    NEEDS_REVIEW = "needs_review"
    FALSE_POSITIVE = "false_positive"
    NOISE = "noise"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ActionStatus(str, Enum):
    EXECUTED = "executed"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"


class CaseStatus(str, Enum):
    CLOSED_NOISE = "closed_noise"
    CLOSED_FALSE_POSITIVE = "closed_false_positive"
    AWAITING_APPROVAL = "awaiting_approval"
    CONTAINED = "contained"
    UNDER_REVIEW = "under_review"


class ChallengeStatus(str, Enum):
    OPEN = "open"
    UPHELD = "upheld"  # the counter-evidence exists -> lowers confidence
    REFUTED = "refuted"  # the counter-evidence does not exist
    UNVERIFIABLE = "unverifiable"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Alert:
    """A normalised security alert coming from any detection source."""

    alert_id: str
    timestamp: datetime
    source: str
    rule_name: str
    description: str = ""
    severity_hint: str = "medium"
    host: str | None = None
    user: str | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    process: str | None = None
    file_hash: str | None = None
    domain: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def entities(self) -> dict[str, str]:
        """Return the non-empty pivotable entities of the alert."""
        candidates = {
            "host": self.host,
            "user": self.user,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "domain": self.domain,
            "file_hash": self.file_hash,
            "process": self.process,
        }
        return {k: v for k, v in candidates.items() if v}

    def title(self) -> str:
        subject = self.user or self.host or self.src_ip or "unknown entity"
        return f"{self.rule_name} ({subject})"


@dataclass
class LogEvent:
    """A normalised telemetry event (authentication, network, process, ...)."""

    event_id: str
    timestamp: datetime
    event_type: str
    host: str | None = None
    user: str | None = None
    src_ip: str | None = None
    dst_ip: str | None = None
    dst_port: int | None = None
    process: str | None = None
    domain: str | None = None
    bytes_out: int = 0
    status: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def short(self) -> str:
        parts = [self.timestamp.strftime("%Y-%m-%d %H:%M:%S"), self.event_type]
        for label, value in (
            ("user", self.user),
            ("host", self.host),
            ("src", self.src_ip),
            ("dst", self.dst_ip),
            ("port", self.dst_port),
            ("proc", self.process),
            ("domain", self.domain),
        ):
            if value not in (None, ""):
                parts.append(f"{label}={value}")
        if self.bytes_out:
            parts.append(f"bytes_out={self.bytes_out}")
        return " ".join(str(p) for p in parts)


@dataclass
class TriageResult:
    score: float
    priority: Priority
    category: str
    techniques: list[str]
    features: dict[str, float]
    anomaly_score: float
    rule_score: float
    reasons: list[str]
    is_noise: bool


@dataclass
class Finding:
    """A piece of evidence produced by deterministic analytics."""

    title: str
    detail: str
    weight: float  # 0..1 - how strongly the finding indicates malicious activity
    technique_id: str | None = None
    evidence: list[str] = field(default_factory=list)  # event ids
    source: str = "analytics"


@dataclass
class Investigation:
    timeline: list[LogEvent]
    findings: list[Finding]
    indicators: dict[str, list[str]]
    hypothesis: str
    attack_stage: str = "unknown"
    open_questions: list[str] = field(default_factory=list)
    llm_used: bool = False


@dataclass
class IndicatorVerdict:
    indicator: str
    kind: str  # ip | domain | hash
    reputation: float  # 0 (clean) .. 1 (known bad)
    malicious: bool
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class IntelReport:
    results: list[IndicatorVerdict]
    techniques: list[dict[str, str]]
    max_reputation: float
    summary: str


@dataclass
class SimilarCase:
    case_id: str
    similarity: float
    title: str
    verdict: str
    summary: str = ""
    actions: list[str] = field(default_factory=list)


@dataclass
class Assessment:
    confidence: float
    verdict: Verdict
    components: dict[str, float] = field(default_factory=dict)
    rationale: list[str] = field(default_factory=list)


@dataclass
class Challenge:
    challenge_id: str
    check: str
    claim: str
    round: int
    status: ChallengeStatus = ChallengeStatus.OPEN
    evidence: str = ""
    impact: float = 0.0
    proposed_by: str = "heuristic"


@dataclass
class DebateRound:
    round: int
    critic_commentary: str
    challenges: list[Challenge]
    confidence_before: float
    confidence_after: float


@dataclass
class ResponseAction:
    action_id: str
    kind: str
    target: str
    risk: RiskLevel
    status: ActionStatus
    rationale: str
    result: str = ""
    decided_by: str = "policy"
    decided_at: datetime | None = None


@dataclass
class AgentMessage:
    """A single entry of the visible 'agent conversation'."""

    agent: str
    kind: str  # thought | tool | result | challenge | decision | error
    content: str
    timestamp: datetime = field(default_factory=utcnow)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Case:
    case_id: str
    alert: Alert
    status: CaseStatus
    verdict: Verdict
    confidence: float
    priority: Priority
    triage: TriageResult
    investigation: Investigation | None = None
    intel: IntelReport | None = None
    similar_cases: list[SimilarCase] = field(default_factory=list)
    initial_assessment: Assessment | None = None
    debate: list[DebateRound] = field(default_factory=list)
    final_assessment: Assessment | None = None
    techniques: list[dict[str, str]] = field(default_factory=list)
    actions: list[ResponseAction] = field(default_factory=list)
    report_markdown: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    llm_provider: str = "offline"
    created_at: datetime = field(default_factory=utcnow)
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, enums and datetimes to JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        return obj.item()
    return obj
