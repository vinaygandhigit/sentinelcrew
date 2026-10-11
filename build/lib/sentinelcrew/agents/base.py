"""Agent base class and the shared tool-kit.

Design principle - *LLM proposes, tools verify*:
every numeric decision (scores, confidence, verdicts, action risk) is made by
deterministic, unit-tested code. The LLM is used for what it is good at -
hypothesis generation, choosing which challenges to raise, narrative and
summarisation - and its output is schema-validated and constrained to
allow-listed vocabularies. If the LLM is unavailable, every agent falls back
to deterministic reasoning, so the crew never stops working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from sentinelcrew.config import Settings
from sentinelcrew.llm.base import LLMClient, LLMError
from sentinelcrew.memory.vector_store import VectorStore
from sentinelcrew.ml.isolation_forest import IsolationForest
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.tools.actions import ActionExecutor
from sentinelcrew.tools.context import OrgContext
from sentinelcrew.tools.log_store import LogStore
from sentinelcrew.tools.mitre import MitreMapper
from sentinelcrew.tools.threat_intel import ThreatIntelService

SOC_PERSONA = (
    "You are a member of SentinelCrew, an autonomous Security Operations Center team. "
    "Be precise, evidence-driven and concise. Never invent log events, indicators or "
    "facts that are not present in the provided context."
)


@dataclass
class Toolkit:
    """Everything an agent may use. Injected, so tests can supply fakes."""

    logs: LogStore
    intel: ThreatIntelService
    mitre: MitreMapper
    context: OrgContext
    executor: ActionExecutor
    memory: VectorStore
    model: IsolationForest
    extras: dict[str, Any] = field(default_factory=dict)


class Agent:
    name = "agent"
    role = ""

    def __init__(self, toolkit: Toolkit, settings: Settings, llm: LLMClient | None = None) -> None:
        self.tools = toolkit
        self.settings = settings
        self.llm = llm

    def __call__(self, state: CaseState) -> None:
        self.run(state)

    def run(self, state: CaseState) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- helpers -------------------------------------------------------
    def say(self, state: CaseState, kind: str, content: str, **data: Any) -> None:
        state.emit(self.name, kind, content, **data)

    def ask_json(
        self, state: CaseState, task: str, context: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Ask the LLM for schema-valid JSON; return ``None`` on any failure."""
        if self.llm is None:
            return None
        system = f"{SOC_PERSONA}\nROLE: {self.name} - {self.role}"
        prompt = f"TASK:\n{task}\n\nCONTEXT (JSON):\n{json.dumps(context, default=str, indent=1)}"
        try:
            result = self.llm.complete_json(system, prompt, schema)
        except LLMError as exc:
            state.record_error(self.name, str(exc))
            self.say(state, "error", f"LLM unavailable, using deterministic reasoning ({exc})")
            return None
        self.say(state, "tool", f"LLM reasoning completed via {self.llm.name}")
        return result
