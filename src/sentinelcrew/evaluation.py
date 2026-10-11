"""Measure the value of the Critic debate ("debate-before-escalation").

Runs every scenario twice - once with the Critic disabled (single-pass
verdict) and once with the full debate - and compares accuracy, false
positives that would have been escalated, and analyst workload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentinelcrew.config import Settings
from sentinelcrew.memory.case_store import CaseStore
from sentinelcrew.models import Verdict
from sentinelcrew.service import SentinelService

ESCALATED = {Verdict.TRUE_POSITIVE.value, Verdict.NEEDS_REVIEW.value}
BENIGN = {Verdict.FALSE_POSITIVE.value, Verdict.NOISE.value}


@dataclass
class EvalRow:
    scenario: str
    expected: str
    without_critic: str
    with_critic: str
    confidence_before: float
    confidence_after: float


def _matches(expected: str, got: str) -> bool:
    if expected in BENIGN:
        return got in BENIGN
    return got == expected


def evaluate(settings: Settings | None = None) -> dict[str, Any]:
    settings = (settings or Settings.from_env()).with_overrides(llm_provider="offline")
    service = SentinelService(settings, llm=None, store=CaseStore(":memory:"))
    rows: list[EvalRow] = []
    for scenario in service.scenarios():
        case = service.run_scenario(scenario.scenario_id)
        before = case.initial_assessment.verdict.value if case.initial_assessment else case.verdict.value
        rows.append(
            EvalRow(
                scenario=scenario.scenario_id,
                expected=scenario.expected_verdict,
                without_critic=before,
                with_critic=case.verdict.value,
                confidence_before=case.initial_assessment.confidence if case.initial_assessment else case.confidence,
                confidence_after=case.confidence,
            )
        )
    n = len(rows) or 1

    def summary(attr: str) -> dict[str, Any]:
        verdicts = [getattr(r, attr) for r in rows]
        correct = sum(_matches(r.expected, v) for r, v in zip(rows, verdicts))
        false_escalations = sum(1 for r, v in zip(rows, verdicts) if r.expected in BENIGN and v in ESCALATED)
        missed = sum(1 for r, v in zip(rows, verdicts) if r.expected == "true_positive" and v != "true_positive")
        analyst_queue = sum(1 for v in verdicts if v in ESCALATED)
        return {
            "accuracy": round(correct / n, 3),
            "false_escalations": false_escalations,
            "missed_incidents": missed,
            "analyst_queue": analyst_queue,
        }

    return {"rows": [r.__dict__ for r in rows], "without_critic": summary("without_critic"),
            "with_critic": summary("with_critic")}


def format_table(result: dict[str, Any]) -> str:
    lines = [
        f"{'scenario':32} {'expected':15} {'no critic':15} {'with critic':15} {'conf':>11}",
        "-" * 92,
    ]
    for r in result["rows"]:
        lines.append(
            f"{r['scenario']:32} {r['expected']:15} {r['without_critic']:15} {r['with_critic']:15} "
            f"{r['confidence_before']:.2f}->{r['confidence_after']:.2f}"
        )
    lines.append("")
    for label in ("without_critic", "with_critic"):
        s = result[label]
        lines.append(
            f"{label:15} accuracy={s['accuracy']:.0%} false_escalations={s['false_escalations']} "
            f"missed={s['missed_incidents']} analyst_queue={s['analyst_queue']}"
        )
    return "\n".join(lines)
