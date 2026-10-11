"""Case-Memory agent: retrieves similar past incidents from the vector store."""

from __future__ import annotations

from sentinelcrew.agents.base import Agent
from sentinelcrew.models import SimilarCase
from sentinelcrew.orchestration.state import CaseState

MIN_SIMILARITY = 0.12


def case_fingerprint(state: CaseState) -> str:
    """Text used to embed a case (also used when the case is memorised)."""
    alert = state.alert
    parts = [alert.rule_name, alert.description, alert.source, alert.process or "", alert.domain or ""]
    if state.triage:
        parts.append(state.triage.category)
        parts.extend(state.triage.techniques)
        parts.extend(state.triage.reasons[:-1])
    if state.investigation:
        parts.extend(f.title for f in state.investigation.findings)
    return " ".join(p for p in parts if p)


class MemoryAgent(Agent):
    name = "CaseMemory"
    role = "Finds similar past incidents and what was done about them."

    def run(self, state: CaseState) -> None:
        query = case_fingerprint(state)
        hits = self.tools.memory.search(query, k=3, min_score=MIN_SIMILARITY, exclude={state.case_id})
        state.similar_cases = [
            SimilarCase(
                case_id=record.record_id,
                similarity=round(score, 3),
                title=record.metadata.get("title", record.record_id),
                verdict=record.metadata.get("verdict", "unknown"),
                summary=record.metadata.get("summary", ""),
                actions=list(record.metadata.get("actions", [])),
            )
            for record, score in hits
        ]
        if not state.similar_cases:
            self.say(state, "result", "No sufficiently similar past cases in memory.")
            return
        for case in state.similar_cases:
            self.say(
                state,
                "result",
                f"Similar to {case.case_id} ({case.similarity:.0%}): '{case.title}' -> {case.verdict}"
                + (f"; actions: {', '.join(case.actions)}" if case.actions else ""),
            )
