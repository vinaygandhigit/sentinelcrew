"""Responder agent: builds a containment plan and enforces risk gating.

Risk policy (human-in-the-loop):

* **LOW**    risk  -> executed automatically (in simulation)
* **MEDIUM** risk  -> automatic only if confidence >= ``AUTO_MEDIUM_RISK_CONFIDENCE``
* **HIGH**   risk  -> always waits for analyst approval

Guardrails: actions must exist in the allow-list, targets must be entities
that actually appear in the case (an LLM cannot introduce new targets), and
protected assets (domain controllers, built-in admin, public DNS) are
rejected outright.
"""

from __future__ import annotations

from sentinelcrew.agents.base import Agent
from sentinelcrew.models import ActionStatus, ResponseAction, RiskLevel, Verdict, utcnow
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.tools.actions import ACTION_CATALOG, ActionNotAllowed

def _next_id(state: CaseState) -> str:
    return f"{state.case_id}-A{len(state.actions) + 1:03d}"


class ResponderAgent(Agent):
    name = "Responder"
    role = "Recommends and (in simulation) executes containment, gated by risk."

    def case_targets(self, state: CaseState) -> dict[str, set[str]]:
        """Entities that legitimately belong to this case, by target type."""
        alert = state.alert
        targets: dict[str, set[str]] = {"ip": set(), "domain": set(), "host": set(), "user": set()}
        if state.intel:
            for r in state.intel.results:
                if r.malicious and r.kind in ("ip", "domain"):
                    targets[r.kind].add(r.indicator)
        if alert.host:
            targets["host"].add(alert.host)
        if alert.user:
            targets["user"].add(alert.user)
        if state.investigation:
            for f in state.investigation.findings:
                if f.technique_id == "T1078":
                    for e in state.investigation.timeline:
                        if e.event_id in f.evidence and e.user:
                            targets["user"].add(e.user)
                        if e.event_id in f.evidence and e.src_ip and e.event_type == "auth":
                            external = self._external(e.src_ip)
                            if external:
                                targets["ip"].add(external)
        return targets

    @staticmethod
    def _external(ip: str) -> str | None:
        from sentinelcrew.tools.threat_intel import is_internal_ip

        return None if is_internal_ip(ip) else ip

    def plan(self, state: CaseState) -> list[tuple[str, str, str]]:
        """Return ``(kind, target, rationale)`` tuples."""
        assert state.assessment is not None
        verdict = state.assessment.verdict
        if verdict == Verdict.FALSE_POSITIVE:
            return []
        alert = state.alert
        subject = alert.user or alert.host or alert.src_ip or alert.alert_id
        if verdict == Verdict.NEEDS_REVIEW:
            return [
                ("add_to_watchlist", subject, "Inconclusive verdict - monitor the entity closely"),
                ("create_ticket", state.case_id, "Hand over to a human analyst for review"),
            ]
        targets = self.case_targets(state)
        techniques = {f.technique_id for f in (state.investigation.findings if state.investigation else [])}
        steps: list[tuple[str, str, str]] = []
        for ip in sorted(targets["ip"]):
            steps.append(("block_ip", ip, "Malicious external IP involved in the attack"))
        for domain in sorted(targets["domain"]):
            steps.append(("block_domain", domain, "Malicious domain (C2 / exfiltration destination)"))
        if techniques & {"T1078", "T1110", "T1110.003"}:
            for user in sorted(targets["user"]):
                steps.append(("reset_sessions", user, "Credentials likely compromised - revoke sessions"))
                if "T1078" in techniques:
                    steps.append(("disable_account", user, "Account used by attacker after compromise"))
        if techniques & {"T1071.001", "T1041", "T1059.001", "T1218.011", "T1059"}:
            for host in sorted(targets["host"]):
                steps.append(("isolate_host", host, "Host shows C2 / execution / exfiltration activity"))
        steps.append(("create_ticket", state.case_id, "Track the incident in ITSM"))
        return steps

    def gate(self, kind: str, confidence: float) -> ActionStatus:
        risk = ACTION_CATALOG[kind].risk
        if risk == RiskLevel.LOW:
            return ActionStatus.EXECUTED
        if risk == RiskLevel.MEDIUM and confidence >= self.settings.auto_medium_risk_confidence:
            return ActionStatus.EXECUTED
        return ActionStatus.PENDING_APPROVAL

    def run(self, state: CaseState) -> None:
        assert state.assessment is not None
        confidence = state.assessment.confidence
        steps = self.plan(state)
        if not steps:
            self.say(state, "decision", "Verdict is benign - no response actions required.")
            return
        self.say(state, "thought", f"Building response plan ({len(steps)} step(s)) at confidence {confidence:.2f}.")
        seen: set[tuple[str, str]] = set()
        for kind, target, rationale in steps:
            if (kind, target) in seen:
                continue
            seen.add((kind, target))
            action_id = _next_id(state)
            try:
                spec = self.tools.executor.validate(kind, target)
            except ActionNotAllowed as exc:
                state.actions.append(
                    ResponseAction(action_id, kind, target, ACTION_CATALOG.get(kind, ACTION_CATALOG["create_ticket"]).risk,
                                   ActionStatus.BLOCKED, rationale, result=str(exc), decided_by="guardrail",
                                   decided_at=utcnow())
                )
                self.say(state, "error", f"Guardrail blocked {kind}({target}): {exc}")
                continue
            status = self.gate(kind, confidence)
            action = ResponseAction(action_id, kind, target, spec.risk, status, rationale)
            if status == ActionStatus.EXECUTED:
                action.result = self.tools.executor.execute(kind, target)
                action.decided_by = "auto-policy"
                action.decided_at = utcnow()
                self.say(state, "result", f"AUTO [{spec.risk.value}] {kind}({target}) -> {action.result}")
            else:
                self.say(state, "decision", f"APPROVAL REQUIRED [{spec.risk.value}] {kind}({target}): {rationale}")
            state.actions.append(action)
