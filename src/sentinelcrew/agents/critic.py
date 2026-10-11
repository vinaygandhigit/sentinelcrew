"""Critic agent - *debate before escalation*.

The Critic plays devil's advocate. Each debate round it raises challenges
("could this be the authorised vulnerability scanner?", "was there an
approved change window?") drawn from a catalogue of **verifiable** checks.
Each challenge is then resolved against real data by a tool, never by
opinion:

* ``UPHELD``       - the benign explanation is supported -> confidence goes down
* ``REFUTED``      - the benign explanation is disproved -> small boost
* ``UNVERIFIABLE`` - not enough data, no change

With an LLM, the model chooses which challenges to raise (constrained to the
catalogue via a JSON-schema ``enum``) and writes the commentary - but the
resolution and its numeric impact stay deterministic. The loop stops when
no new challenges remain, the confidence stops moving, or after
``DEBATE_MAX_ROUNDS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sentinelcrew.agents.base import Agent, Toolkit
from sentinelcrew.ml.features import FAILED_STATUSES
from sentinelcrew.models import Challenge, ChallengeStatus, DebateRound
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.scoring import BENIGN_VERDICTS, MALICIOUS_VERDICTS, clamp, fuse
from sentinelcrew.tools.threat_intel import is_internal_ip

Outcome = tuple[ChallengeStatus, str, float]

MAX_NEGATIVE_ADJUSTMENT = -0.55
MAX_POSITIVE_ADJUSTMENT = 0.10
CHECKS_PER_ROUND = 3
CONVERGENCE_EPSILON = 0.005


@dataclass(frozen=True)
class CriticCheck:
    check_id: str
    question: str
    applies: Callable[[CaseState, Toolkit], bool]
    evaluate: Callable[[CaseState, Toolkit], Outcome]
    priority: int = 50


def _entities(state: CaseState) -> list[str]:
    alert = state.alert
    values = [alert.host, alert.src_ip, alert.dst_ip]
    return [v for v in values if v]


def _techniques(state: CaseState) -> set[str]:
    found = {f.technique_id for f in (state.investigation.findings if state.investigation else []) if f.technique_id}
    if state.triage:
        found.update(state.triage.techniques)
    return found


# ---- check implementations ---------------------------------------------
def _scanner_applies(state: CaseState, tools: Toolkit) -> bool:
    techniques = _techniques(state)
    text = f"{state.alert.rule_name} {state.alert.description}".lower()
    return "T1046" in techniques or "scan" in text or "recon" in text


def _scanner_eval(state: CaseState, tools: Toolkit) -> Outcome:
    for entity in _entities(state):
        asset = tools.context.asset(entity)
        if asset and "authorized-scanner" in asset.tags:
            return (
                ChallengeStatus.UPHELD,
                f"{entity} is {asset.hostname}, an authorised vulnerability scanner owned by {asset.owner or 'IT'}",
                -0.30,
            )
    return ChallengeStatus.REFUTED, "source is not a registered scanner in the asset inventory", 0.02


def _change_eval(state: CaseState, tools: Toolkit) -> Outcome:
    changes = tools.context.active_changes(state.alert.timestamp, *_entities(state))
    if changes:
        change = changes[0]
        return (
            ChallengeStatus.UPHELD,
            f"approved change {change.change_id} was active: {change.description}",
            -0.15,
        )
    return ChallengeStatus.REFUTED, "no approved change window covers these assets at that time", 0.0


def _allowlist_applies(state: CaseState, tools: Toolkit) -> bool:
    return bool(state.intel and state.intel.results)


def _allowlist_eval(state: CaseState, tools: Toolkit) -> Outcome:
    assert state.intel is not None
    malicious = [r for r in state.intel.results if r.malicious]
    allowed = [r for r in state.intel.results if "allowlisted" in r.tags]
    if malicious:
        names = ", ".join(r.indicator for r in malicious[:3])
        return ChallengeStatus.REFUTED, f"destination(s) are known-malicious, not sanctioned: {names}", 0.03
    if allowed and len(allowed) == len(state.intel.results):
        return ChallengeStatus.UPHELD, "every external indicator is on the organisation allowlist", -0.20
    return ChallengeStatus.UNVERIFIABLE, "indicators are neither allowlisted nor known-bad", 0.0


def _intel_eval(state: CaseState, tools: Toolkit) -> Outcome:
    if not state.intel or not state.intel.results:
        return ChallengeStatus.UPHELD, "no external indicators - no independent intel corroboration", -0.04
    rep = state.intel.max_reputation
    if rep >= 0.7:
        return ChallengeStatus.REFUTED, f"independent intel corroborates (max reputation {rep:.2f})", 0.03
    if rep >= 0.4:
        return ChallengeStatus.UNVERIFIABLE, f"intel is only suggestive (max reputation {rep:.2f})", 0.0
    return ChallengeStatus.UPHELD, "intel sources have nothing on these indicators", -0.04


def _single_signal_applies(state: CaseState, tools: Toolkit) -> bool:
    return bool(state.investigation and state.investigation.findings)


def _single_signal_eval(state: CaseState, tools: Toolkit) -> Outcome:
    assert state.investigation is not None
    strong = [f for f in state.investigation.findings if f.weight >= 0.4]
    tactics = {
        t.tactic for t in (tools.mitre.get(f.technique_id or "") for f in strong) if t
    }
    if len(strong) <= 1:
        return ChallengeStatus.UPHELD, f"only {len(strong)} strong finding - verdict rests on one signal", -0.06
    if len(tactics) >= 2:
        return (
            ChallengeStatus.REFUTED,
            f"{len(strong)} strong findings across {len(tactics)} ATT&CK tactics corroborate each other",
            0.03,
        )
    return ChallengeStatus.UNVERIFIABLE, f"{len(strong)} findings but all within a single tactic", 0.0


def _user_error_applies(state: CaseState, tools: Toolkit) -> bool:
    return "T1110" in _techniques(state) or "T1078" in _techniques(state)


def _user_error_eval(state: CaseState, tools: Toolkit) -> Outcome:
    assert state.investigation is not None
    failed = [
        e for e in state.investigation.timeline
        if e.event_type == "auth" and (e.status or "") in FAILED_STATUSES
    ]
    sources = {e.src_ip for e in failed if e.src_ip}
    external = [s for s in sources if not is_internal_ip(s)]
    if len(failed) < 10 and not external:
        return ChallengeStatus.UPHELD, f"only {len(failed)} internal failures - consistent with a mistyped password", -0.15
    return (
        ChallengeStatus.REFUTED,
        f"{len(failed)} failures" + (f" from external source {external[0]}" if external else "")
        + " - not a typo pattern",
        0.02,
    )


def _memory_applies(state: CaseState, tools: Toolkit) -> bool:
    return any(c.similarity >= 0.25 for c in state.similar_cases)


def _memory_eval(state: CaseState, tools: Toolkit) -> Outcome:
    best = max(state.similar_cases, key=lambda c: c.similarity)
    if best.verdict in BENIGN_VERDICTS:
        return ChallengeStatus.UPHELD, f"closest precedent {best.case_id} ({best.similarity:.0%}) was benign: {best.title}", -0.10
    if best.verdict in MALICIOUS_VERDICTS:
        return ChallengeStatus.REFUTED, f"closest precedent {best.case_id} ({best.similarity:.0%}) was a confirmed incident", 0.02
    return ChallengeStatus.UNVERIFIABLE, "precedent was inconclusive", 0.0


def _service_account_applies(state: CaseState, tools: Toolkit) -> bool:
    return tools.context.identity(state.alert.user) is not None


def _service_account_eval(state: CaseState, tools: Toolkit) -> Outcome:
    identity = tools.context.identity(state.alert.user)
    assert identity is not None
    if "service-account" in identity.tags and not ({"T1059.001", "T1218.011", "T1041"} & _techniques(state)):
        return ChallengeStatus.UPHELD, f"{identity.username} is a service account with known noisy automation", -0.10
    if identity.privileged:
        return ChallengeStatus.REFUTED, f"{identity.username} is a privileged account - no benign automation explains this", 0.02
    return ChallengeStatus.REFUTED, f"{identity.username} is a regular interactive user", 0.0


CHECKS: tuple[CriticCheck, ...] = (
    CriticCheck("authorized_scanner", "Could the source be an authorised vulnerability scanner?",
                _scanner_applies, _scanner_eval, 10),
    CriticCheck("change_window", "Was there an approved maintenance / change window?",
                lambda s, t: True, _change_eval, 20),
    CriticCheck("allowlisted_destinations", "Are the destinations sanctioned business services?",
                _allowlist_applies, _allowlist_eval, 30),
    CriticCheck("single_signal", "Does the verdict rest on a single, uncorroborated signal?",
                _single_signal_applies, _single_signal_eval, 40),
    CriticCheck("benign_user_error", "Could the failed logons simply be a user mistyping a password?",
                _user_error_applies, _user_error_eval, 50),
    CriticCheck("past_precedent", "Were similar past cases false positives?",
                _memory_applies, _memory_eval, 60),
    CriticCheck("service_account", "Is this expected automation from a service account?",
                _service_account_applies, _service_account_eval, 70),
    CriticCheck("intel_corroboration", "Is there independent threat-intel corroboration?",
                lambda s, t: True, _intel_eval, 80),
)
CHECKS_BY_ID = {c.check_id: c for c in CHECKS}


class CriticAgent(Agent):
    name = "Critic"
    role = (
        "Devil's advocate. Challenges the team's conclusion by proposing benign explanations "
        "and missing evidence. Only selects checks from the provided catalogue."
    )

    def available_checks(self, state: CaseState) -> list[CriticCheck]:
        return [
            c for c in sorted(CHECKS, key=lambda c: c.priority)
            if c.check_id not in state.checks_done and c.applies(state, self.tools)
        ]

    def select(self, state: CaseState, available: list[CriticCheck]) -> tuple[list[CriticCheck], str, str]:
        """Choose this round's challenges. Returns (checks, commentary, proposer)."""
        default = available[:CHECKS_PER_ROUND]
        if not available:
            return [], "No further challenges.", "heuristic"
        assessment = state.assessment
        llm = self.ask_json(
            state,
            "You are reviewing the team's verdict. Argue the opposite: pick the challenges most likely "
            f"to reveal a benign explanation or a gap in evidence (at most {CHECKS_PER_ROUND}), and write "
            "a short critical commentary.",
            {
                "hypothesis": state.investigation.hypothesis if state.investigation else "",
                "current_confidence": assessment.confidence if assessment else None,
                "findings": [f.detail for f in (state.investigation.findings if state.investigation else [])],
                "previous_rounds": [
                    {"check": c.check, "status": c.status.value, "evidence": c.evidence}
                    for r in state.debate for c in r.challenges
                ],
                "available_checks": {c.check_id: c.question for c in available},
            },
            {
                "type": "object",
                "required": ["commentary", "selected_checks"],
                "properties": {
                    "commentary": {"type": "string", "maxLength": 800},
                    "selected_checks": {
                        "type": "array",
                        "items": {"type": "string", "enum": [c.check_id for c in available]},
                        "minItems": 1,
                        "maxItems": CHECKS_PER_ROUND,
                    },
                },
            },
        )
        if llm:
            chosen = [CHECKS_BY_ID[i] for i in dict.fromkeys(llm["selected_checks"])]
            return chosen, llm["commentary"].strip(), "llm"
        verdict = assessment.verdict.value if assessment else "unknown"
        commentary = (
            f"The team leans '{verdict}'. Before we act I want benign explanations ruled out: "
            + " ".join(c.question for c in default)
        )
        return default, commentary, "heuristic"

    def run(self, state: CaseState) -> None:
        round_no = state.debate_rounds + 1
        before = state.assessment.confidence if state.assessment else 0.5
        available = self.available_checks(state)
        checks, commentary, proposer = self.select(state, available)
        self.say(state, "thought", f"Round {round_no}: {commentary}")
        challenges: list[Challenge] = []
        for check in checks:
            status, evidence, impact = check.evaluate(state, self.tools)
            state.checks_done.add(check.check_id)
            challenge = Challenge(
                challenge_id=f"R{round_no}-{check.check_id}",
                check=check.check_id,
                claim=check.question,
                round=round_no,
                status=status,
                evidence=evidence,
                impact=impact,
                proposed_by=proposer,
            )
            challenges.append(challenge)
            self.say(state, "challenge", f"{check.question} -> {status.value.upper()}: {evidence} ({impact:+.2f})",
                     check=check.check_id, status=status.value, impact=impact)
        state.critic_adjustment = clamp(
            state.critic_adjustment + sum(c.impact for c in challenges),
            MAX_NEGATIVE_ADJUSTMENT,
            MAX_POSITIVE_ADJUSTMENT,
        )
        state.assessment = self.reassess(state)
        after = state.assessment.confidence
        state.debate.append(DebateRound(round_no, commentary, challenges, before, after))
        remaining = self.available_checks(state)
        state.debate_converged = not remaining or (
            round_no > 1 and abs(after - before) < CONVERGENCE_EPSILON
        )
        self.say(
            state,
            "decision",
            f"Confidence {before:.2f} -> {after:.2f} ({state.assessment.verdict.value})"
            + (" - debate converged" if state.debate_converged else ""),
            before=before,
            after=after,
        )

    def reassess(self, state: CaseState):  # noqa: ANN201 - returns Assessment
        return fuse_state(state, self.settings, state.critic_adjustment)


def fuse_state(state: CaseState, settings, adjustment: float = 0.0):  # noqa: ANN001, ANN201
    """Compute the fused assessment for the current state."""
    intel = state.intel.max_reputation if state.intel and state.intel.results else None
    return fuse(
        triage_score=state.triage.score if state.triage else 0.5,
        findings=state.investigation.findings if state.investigation else [],
        intel_reputation=intel,
        similar=state.similar_cases,
        tp_threshold=settings.tp_threshold,
        fp_threshold=settings.fp_threshold,
        adjustment=adjustment,
    )
