"""Application service - the single entry point used by the CLI, API and UI.

Responsibilities: build the toolkit from the data directory, run alerts
through the workflow, persist cases, update case memory, and handle the
human-in-the-loop approval of pending actions.
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from sentinelcrew.agents.base import Toolkit
from sentinelcrew.agents.memory_agent import case_fingerprint
from sentinelcrew.config import Settings
from sentinelcrew.ingestion.parsers import parse_alerts, parse_events
from sentinelcrew.llm.base import LLMClient
from sentinelcrew.llm.factory import create_llm_safe
from sentinelcrew.memory.case_store import CaseStore
from sentinelcrew.memory.vector_store import VectorStore
from sentinelcrew.ml.isolation_forest import IsolationForest
from sentinelcrew.ml.train import train_model
from sentinelcrew.models import (
    ActionStatus,
    Alert,
    AgentMessage,
    Case,
    CaseStatus,
    LogEvent,
    Priority,
    TriageResult,
    Verdict,
    utcnow,
)
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.orchestration.workflow import SOCWorkflow
from sentinelcrew.tools.actions import ActionExecutor, ActionNotAllowed, SimulatedEnvironment
from sentinelcrew.tools.context import OrgContext
from sentinelcrew.tools.log_store import LogStore
from sentinelcrew.tools.mitre import MitreMapper
from sentinelcrew.tools.threat_intel import LocalIntelFeed, ThreatIntelService


class ServiceError(RuntimeError):
    pass


@dataclass
class Scenario:
    scenario_id: str
    title: str
    description: str
    expected_verdict: str
    alert: Alert
    events: list[LogEvent]


def load_scenario(path: Path) -> Scenario:
    meta = json.loads((path / "scenario.json").read_text(encoding="utf-8"))
    alerts, errors = parse_alerts(json.dumps(meta["alert"]), fmt="json")
    if errors or not alerts:
        raise ServiceError(f"invalid alert in scenario {path.name}: {errors}")
    events: list[LogEvent] = []
    logs = path / "logs.jsonl"
    if logs.exists():
        events, errors = parse_events(logs.read_text(encoding="utf-8"), fmt="jsonl")
        if errors:
            raise ServiceError(f"invalid logs in scenario {path.name}: {errors[:3]}")
    return Scenario(
        scenario_id=path.name,
        title=meta.get("title", path.name),
        description=meta.get("description", ""),
        expected_verdict=meta.get("expected_verdict", ""),
        alert=alerts[0],
        events=events,
    )


def derive_status(verdict: Verdict, actions: list[Any]) -> CaseStatus:
    statuses = {a.status if hasattr(a, "status") else ActionStatus(a["status"]) for a in actions}
    if verdict == Verdict.NOISE:
        return CaseStatus.CLOSED_NOISE
    if verdict == Verdict.FALSE_POSITIVE:
        return CaseStatus.CLOSED_FALSE_POSITIVE
    if ActionStatus.PENDING_APPROVAL in statuses:
        return CaseStatus.AWAITING_APPROVAL
    if verdict == Verdict.TRUE_POSITIVE:
        return CaseStatus.CONTAINED
    return CaseStatus.UNDER_REVIEW


class SentinelService:
    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMClient | None | str = "auto",
        store: CaseStore | None = None,
        model: IsolationForest | None = None,
        load_history: bool = True,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if llm == "auto":
            self.llm, self.llm_note = create_llm_safe(self.settings)
        else:
            self.llm = llm  # type: ignore[assignment]
            self.llm_note = f"{llm.name}:{llm.model}" if llm else "offline deterministic mode"
        data = self.settings.data_dir

        self.environment = SimulatedEnvironment()
        self.toolkit = Toolkit(
            logs=LogStore(),
            intel=ThreatIntelService([LocalIntelFeed.from_file(data / "intel" / "feed.json")]),
            mitre=MitreMapper.from_file(data / "mitre" / "techniques.json"),
            context=OrgContext.from_file(data / "context" / "org.json"),
            executor=ActionExecutor(self.environment),
            memory=VectorStore(),
            model=model or self._load_model(),
        )
        self.store = store or CaseStore(self.settings.db_path)
        self.workflow = SOCWorkflow(self.toolkit, self.settings, self.llm)
        self._ids = itertools.count(self.store.count() + 1)
        self._lock = threading.Lock()
        self._load_baseline_logs()
        if load_history:
            self._load_history()

    # ---- bootstrapping -------------------------------------------------
    def _load_model(self) -> IsolationForest:
        path = self.settings.model_path
        if path.exists():
            try:
                return IsolationForest.load(path)
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        model = train_model()
        try:
            model.save(path)
        except OSError:
            pass
        return model

    def _load_baseline_logs(self) -> None:
        path = self.settings.data_dir / "logs" / "baseline.jsonl"
        if path.exists():
            events, _ = parse_events(path.read_text(encoding="utf-8"), fmt="jsonl")
            self.toolkit.logs.add_many(events)

    def _load_history(self) -> None:
        path = self.settings.data_dir / "memory" / "historical_cases.json"
        if not path.exists():
            return
        items = json.loads(path.read_text(encoding="utf-8"))
        self.toolkit.memory.add_many(
            [
                (
                    c["case_id"],
                    c["fingerprint"],
                    {"title": c["title"], "verdict": c["verdict"], "summary": c.get("summary", ""),
                     "actions": c.get("actions", [])},
                )
                for c in items
            ]
        )

    # ---- scenarios -----------------------------------------------------
    def scenarios(self) -> list[Scenario]:
        root = self.settings.data_dir / "scenarios"
        return [load_scenario(p) for p in sorted(root.iterdir()) if (p / "scenario.json").exists()]

    def scenario(self, scenario_id: str) -> Scenario:
        path = self.settings.data_dir / "scenarios" / scenario_id
        if not (path / "scenario.json").exists():
            raise ServiceError(f"unknown scenario '{scenario_id}'")
        return load_scenario(path)

    def run_scenario(
        self, scenario_id: str, listener: Callable[[AgentMessage], None] | None = None
    ) -> Case:
        scenario = self.scenario(scenario_id)
        self.ingest_events(scenario.events)
        case = self.process_alert(scenario.alert, listener=listener)
        case.metadata["scenario"] = scenario_id
        case.metadata["expected_verdict"] = scenario.expected_verdict
        self.store.save(case)
        return case

    # ---- core ----------------------------------------------------------
    def ingest_events(self, events: list[LogEvent]) -> int:
        return self.toolkit.logs.add_many(events)

    def ingest_text(self, text: str, kind: str = "alerts", fmt: str | None = None) -> tuple[list[Any], list[str]]:
        if kind == "events":
            events, errors = parse_events(text, fmt)
            self.ingest_events(events)
            return events, errors
        return parse_alerts(text, fmt)

    def new_case_id(self) -> str:
        with self._lock:
            while True:
                case_id = f"CASE-{next(self._ids):04d}"
                if self.store.get(case_id) is None:
                    return case_id

    def process_alert(self, alert: Alert, listener: Callable[[AgentMessage], None] | None = None) -> Case:
        case_id = self.new_case_id()
        state = CaseState(case_id=case_id, alert=alert)
        if listener:
            state.listeners.append(listener)
        started = time.perf_counter()
        self.workflow.run(state)
        duration = (time.perf_counter() - started) * 1000
        case = self._to_case(state, duration)
        self.store.save(case)
        self._memorise(state, case)
        return case

    def _to_case(self, state: CaseState, duration_ms: float) -> Case:
        triage = state.triage or TriageResult(0.0, Priority.INFO, "unknown", [], {}, 0.0, 0.0, ["triage failed"], False)
        assessment = state.assessment
        verdict = assessment.verdict if assessment else Verdict.NEEDS_REVIEW
        confidence = assessment.confidence if assessment else 0.5
        techniques = state.intel.techniques if state.intel else []
        return Case(
            case_id=state.case_id,
            alert=state.alert,
            status=derive_status(verdict, state.actions),
            verdict=verdict,
            confidence=confidence,
            priority=triage.priority,
            triage=triage,
            investigation=state.investigation,
            intel=state.intel,
            similar_cases=state.similar_cases,
            initial_assessment=state.initial_assessment,
            debate=state.debate,
            final_assessment=assessment,
            techniques=techniques,
            actions=state.actions,
            report_markdown=state.report_markdown,
            messages=state.messages,
            llm_provider=self.llm.name if self.llm else "offline",
            duration_ms=round(duration_ms, 1),
            metadata={"errors": state.errors, "trace": state.trace},
        )

    def _memorise(self, state: CaseState, case: Case) -> None:
        if case.verdict == Verdict.NOISE:
            return
        self.toolkit.memory.add(
            case.case_id,
            case_fingerprint(state),
            {
                "title": case.alert.title(),
                "verdict": case.verdict.value,
                "summary": state.investigation.hypothesis if state.investigation else "",
                "actions": [f"{a.kind}:{a.target}" for a in case.actions if a.status == ActionStatus.EXECUTED],
            },
        )

    # ---- human in the loop --------------------------------------------
    def decide_action(
        self, case_id: str, action_id: str, approve: bool, analyst: str = "analyst", comment: str = ""
    ) -> dict[str, Any]:
        payload = self.store.get(case_id)
        if payload is None:
            raise ServiceError(f"unknown case {case_id}")
        action = next((a for a in payload["actions"] if a["action_id"] == action_id), None)
        if action is None:
            raise ServiceError(f"unknown action {action_id}")
        if action["status"] != ActionStatus.PENDING_APPROVAL.value:
            raise ServiceError(f"action {action_id} is not pending (status={action['status']})")
        now = utcnow().isoformat()
        if approve:
            try:
                action["result"] = self.toolkit.executor.execute(action["kind"], action["target"])
                action["status"] = ActionStatus.APPROVED.value
            except ActionNotAllowed as exc:
                action["result"] = str(exc)
                action["status"] = ActionStatus.BLOCKED.value
        else:
            action["status"] = ActionStatus.REJECTED.value
            action["result"] = f"rejected by {analyst}" + (f": {comment}" if comment else "")
        action["decided_by"] = analyst
        action["decided_at"] = now
        payload["status"] = derive_status(Verdict(payload["verdict"]), payload["actions"]).value
        payload.setdefault("messages", []).append(
            {"agent": f"Human:{analyst}", "kind": "decision",
             "content": f"{'APPROVED' if approve else 'REJECTED'} {action['kind']}({action['target']})"
                        + (f" - {comment}" if comment else ""),
             "timestamp": now, "data": {}}
        )
        self.store.update_payload(case_id, payload)
        self.store.record_decision(case_id, action_id, "approve" if approve else "reject", analyst, comment, now)
        return payload

    def metrics(self) -> dict[str, Any]:
        m = self.store.metrics()
        m["llm"] = self.llm_note
        m["environment"] = self.environment.snapshot()
        m["memory_size"] = len(self.toolkit.memory)
        m["log_events"] = len(self.toolkit.logs)
        if self.llm:
            m["llm_usage"] = self.llm.usage.as_dict()
        return m

    def save_report(self, case_id: str, out_dir: Path | None = None) -> Path:
        payload = self.store.get(case_id)
        if payload is None:
            raise ServiceError(f"unknown case {case_id}")
        out_dir = out_dir or self.settings.reports_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{case_id}.md"
        path.write_text(payload["report_markdown"], encoding="utf-8")
        return path