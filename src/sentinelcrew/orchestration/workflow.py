"""The Orchestrator: wires the agent team into the SOC workflow graph.

::

    triage --noise--> close_noise --> report --> END
       |
     escalate
       v
    analysis  (parallel: Investigator || Threat-Intel)
       v
    memory  -> assess -> critic --(continue)--> critic   (debate loop, max N)
                            |
                          (done)
                            v
                       respond -> report -> END
"""

from __future__ import annotations

from sentinelcrew.agents.base import Toolkit
from sentinelcrew.agents.critic import CriticAgent, fuse_state
from sentinelcrew.agents.investigator import InvestigatorAgent
from sentinelcrew.agents.memory_agent import MemoryAgent
from sentinelcrew.agents.reporter import ReporterAgent
from sentinelcrew.agents.responder import ResponderAgent
from sentinelcrew.agents.threat_intel_agent import ThreatIntelAgent
from sentinelcrew.agents.triage import TriageAgent
from sentinelcrew.config import Settings
from sentinelcrew.llm.base import LLMClient
from sentinelcrew.models import Assessment, Verdict
from sentinelcrew.orchestration.graph import END, StateGraph
from sentinelcrew.orchestration.state import CaseState

ORCHESTRATOR = "Orchestrator"


class SOCWorkflow:
    def __init__(self, toolkit: Toolkit, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm
        self.triage = TriageAgent(toolkit, settings, llm)
        self.investigator = InvestigatorAgent(toolkit, settings, llm)
        self.intel = ThreatIntelAgent(toolkit, settings, llm)
        self.memory = MemoryAgent(toolkit, settings, llm)
        self.critic = CriticAgent(toolkit, settings, llm)
        self.responder = ResponderAgent(toolkit, settings, llm)
        self.reporter = ReporterAgent(toolkit, settings, llm)
        self.graph = self.build()

    # ---- nodes owned by the orchestrator --------------------------------
    def close_noise(self, state: CaseState) -> None:
        score = state.triage.score if state.triage else 0.0
        state.assessment = Assessment(
            confidence=round(score, 4),
            verdict=Verdict.NOISE,
            components={"triage": score},
            rationale=["closed at triage: below noise threshold and no rule fired"],
        )
        state.initial_assessment = state.assessment
        state.emit(ORCHESTRATOR, "decision", "Closed as noise - no further agent time spent.")

    def assess(self, state: CaseState) -> None:
        state.assessment = fuse_state(state, self.settings)
        state.initial_assessment = state.assessment
        a = state.assessment
        state.emit(
            ORCHESTRATOR,
            "decision",
            f"Initial fused confidence {a.confidence:.2f} -> leaning {a.verdict.value}. Handing to Critic.",
            components=a.components,
        )

    # ---- routers --------------------------------------------------------
    def route_after_triage(self, state: CaseState) -> str:
        return "noise" if state.triage and state.triage.is_noise else "escalate"

    def route_debate(self, state: CaseState) -> str:
        if state.debate_converged or state.debate_rounds >= self.settings.debate_max_rounds:
            return "done"
        return "continue"

    def build(self) -> StateGraph[CaseState]:
        graph: StateGraph[CaseState] = StateGraph(max_workers=4)
        graph.add_node("triage", self.triage)
        graph.add_node("close_noise", self.close_noise)
        graph.add_parallel_node("analysis", {"investigator": self.investigator, "threat_intel": self.intel})
        graph.add_node("memory", self.memory)
        graph.add_node("assess", self.assess)
        graph.add_node("critic", self.critic)
        graph.add_node("respond", self.responder)
        graph.add_node("report", self.reporter)

        graph.set_entry_point("triage")
        graph.add_conditional_edges("triage", self.route_after_triage, {"noise": "close_noise", "escalate": "analysis"})
        graph.add_edge("close_noise", "report")
        graph.add_edge("analysis", "memory")
        graph.add_edge("memory", "assess")
        graph.add_edge("assess", "critic")
        graph.add_conditional_edges("critic", self.route_debate, {"continue": "critic", "done": "respond"})
        graph.add_edge("respond", "report")
        graph.add_edge("report", END)
        graph.validate()
        return graph

    def _on_error(self, node: str, exc: Exception, state: CaseState) -> None:
        state.record_error(node, repr(exc))
        state.emit(ORCHESTRATOR, "error", f"Node '{node}' failed: {exc}. Continuing with degraded data.")

    def run(self, state: CaseState) -> CaseState:
        state.emit(ORCHESTRATOR, "thought", f"Opened {state.case_id} for alert {state.alert.alert_id}.")
        max_steps = 10 + self.settings.debate_max_rounds
        return self.graph.run(state, max_steps=max_steps, trace=state.trace, on_error=self._on_error)
