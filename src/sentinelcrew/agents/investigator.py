"""Investigator agent: builds the timeline and runs correlation detectors.

The detectors are deterministic and explainable; the (optional) LLM then
writes the investigative hypothesis and lists open questions for the Critic.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

from sentinelcrew.agents.base import Agent
from sentinelcrew.ml.features import (
    BEACON_MAX_BYTES,
    FAILED_STATUSES,
    SUCCESS_STATUSES,
    beacon_regularity,
    is_off_hours,
    is_suspicious_process,
)
from sentinelcrew.models import Alert, Finding, Investigation, LogEvent
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.tools.threat_intel import classify_indicator, extract_indicators, is_internal_ip

Detector = Callable[[Alert, list[LogEvent]], list[Finding]]

TACTIC_ORDER = (
    "reconnaissance",
    "discovery",
    "credential-access",
    "initial-access",
    "execution",
    "persistence",
    "lateral-movement",
    "command-and-control",
    "exfiltration",
    "impact",
)

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "required": ["hypothesis", "attack_stage", "open_questions"],
    "properties": {
        "hypothesis": {"type": "string", "maxLength": 1200},
        "attack_stage": {"type": "string", "enum": list(TACTIC_ORDER) + ["benign", "unknown"]},
        "open_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    },
}


def detect_bruteforce(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    failed = [e for e in events if e.event_type == "auth" and (e.status or "") in FAILED_STATUSES]
    by_src: dict[str, list[LogEvent]] = defaultdict(list)
    for e in failed:
        by_src[e.src_ip or "unknown"].append(e)
    findings = []
    for src, items in by_src.items():
        users = {e.user for e in items if e.user}
        if len(items) >= 10:
            findings.append(
                Finding(
                    title="Brute-force authentication",
                    detail=f"{len(items)} failed logons from {src} against {len(users)} account(s)",
                    weight=0.5 if len(items) < 30 else 0.6,
                    technique_id="T1110",
                    evidence=[e.event_id for e in items[:5]],
                )
            )
        if len(users) >= 5:
            findings.append(
                Finding(
                    title="Password spraying",
                    detail=f"{len(users)} distinct accounts targeted from {src}",
                    weight=0.45,
                    technique_id="T1110.003",
                    evidence=[e.event_id for e in items[:5]],
                )
            )
    return findings


def detect_success_after_failures(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    auth = [e for e in events if e.event_type == "auth"]
    findings = []
    for success in (e for e in auth if (e.status or "") in SUCCESS_STATUSES):
        prior = [
            e
            for e in auth
            if (e.status or "") in FAILED_STATUSES
            and e.timestamp < success.timestamp
            and (e.src_ip == success.src_ip or e.user == success.user)
        ]
        if len(prior) >= 5:
            external = success.src_ip and not is_internal_ip(success.src_ip)
            findings.append(
                Finding(
                    title="Account compromise: success after failures",
                    detail=(
                        f"{success.user} logged on successfully from {success.src_ip} after "
                        f"{len(prior)} failures" + (" (external source)" if external else "")
                    ),
                    weight=0.65 if external else 0.5,
                    technique_id="T1078",
                    evidence=[success.event_id] + [e.event_id for e in prior[-3:]],
                )
            )
            break
    return findings


def _process_technique(process: str) -> str:
    proc = process.lower()
    if "rundll32" in proc:
        return "T1218.011"
    if "powershell" in proc:
        return "T1059.001"
    return "T1059"


def detect_suspicious_process(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    """One finding per distinct technique (e.g. encoded PowerShell AND rundll32)."""
    hits = [e for e in events if e.event_type == "process" and is_suspicious_process(e.process)]
    if not hits and alert.process and is_suspicious_process(alert.process):
        return [
            Finding(
                title="Suspicious process execution",
                detail=f"alert process: {alert.process}",
                weight=0.45,
                technique_id=_process_technique(alert.process),
            )
        ]
    grouped: dict[str, list[LogEvent]] = defaultdict(list)
    for e in hits:
        grouped[_process_technique(e.process or "")].append(e)
    return [
        Finding(
            title="Suspicious process execution",
            detail=f"{len(items)} suspicious process(es) on {items[0].host}: {items[0].process}",
            weight=0.55,
            technique_id=technique,
            evidence=[e.event_id for e in items[:5]],
        )
        for technique, items in grouped.items()
    ]


def detect_beaconing(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    by_dest: dict[str, list[LogEvent]] = defaultdict(list)
    for e in events:
        if e.event_type in ("network", "proxy") and e.bytes_out <= BEACON_MAX_BYTES:
            key = e.domain or e.dst_ip
            if key and not (classify_indicator(key) == "ip" and is_internal_ip(key)):
                by_dest[key].append(e)
    findings = []
    for dest, items in by_dest.items():
        regularity = beacon_regularity(e.timestamp for e in items)
        if len(items) >= 6 and regularity >= 0.8:
            gaps = sorted(items, key=lambda e: e.timestamp)
            interval = (gaps[-1].timestamp - gaps[0].timestamp).total_seconds() / max(1, len(gaps) - 1)
            findings.append(
                Finding(
                    title="C2 beaconing",
                    detail=f"{len(items)} connections to {dest} every ~{interval:.0f}s (regularity {regularity:.2f})",
                    weight=0.6,
                    technique_id="T1071.001",
                    evidence=[e.event_id for e in items[:5]],
                )
            )
    return findings


def detect_exfiltration(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    by_dest: Counter[str] = Counter()
    ids: dict[str, list[str]] = defaultdict(list)
    for e in events:
        dest = e.domain or e.dst_ip
        if e.bytes_out and dest and not (classify_indicator(dest) == "ip" and is_internal_ip(dest)):
            by_dest[dest] += e.bytes_out
            ids[dest].append(e.event_id)
    findings = []
    for dest, total in by_dest.most_common(3):
        mb = total / 1_000_000
        if mb >= 50:
            findings.append(
                Finding(
                    title="Large outbound data transfer",
                    detail=f"{mb:.1f} MB sent to {dest}",
                    weight=0.55 if mb >= 100 else 0.4,
                    technique_id="T1041",
                    evidence=ids[dest][:5],
                )
            )
    return findings


def detect_port_scan(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    by_src: dict[str, set[int]] = defaultdict(set)
    ids: dict[str, list[str]] = defaultdict(list)
    for e in events:
        if e.event_type == "network" and e.dst_port and e.src_ip:
            by_src[e.src_ip].add(e.dst_port)
            ids[e.src_ip].append(e.event_id)
    findings = []
    for src, ports in by_src.items():
        if len(ports) >= 20:
            findings.append(
                Finding(
                    title="Port scanning",
                    detail=f"{src} probed {len(ports)} distinct ports",
                    weight=0.45,
                    technique_id="T1046",
                    evidence=ids[src][:5],
                )
            )
    return findings


def detect_off_hours(alert: Alert, events: list[LogEvent]) -> list[Finding]:
    if is_off_hours(alert.timestamp):
        return [
            Finding(
                title="Off-hours activity",
                detail=f"activity at {alert.timestamp:%a %H:%M} UTC, outside business hours",
                weight=0.15,
            )
        ]
    return []


DETECTORS: tuple[Detector, ...] = (
    detect_bruteforce,
    detect_success_after_failures,
    detect_suspicious_process,
    detect_beaconing,
    detect_exfiltration,
    detect_port_scan,
    detect_off_hours,
)


def collect_indicators(alert: Alert, events: list[LogEvent]) -> dict[str, list[str]]:
    found = extract_indicators([alert.description, alert.process or ""])
    for value in (alert.src_ip, alert.dst_ip, alert.domain, alert.file_hash):
        if value:
            kind = classify_indicator(value)
            if kind in found:
                found[kind].add(value.lower() if kind != "ip" else value)
    for e in events:
        for value in (e.src_ip, e.dst_ip, e.domain):
            if value:
                kind = classify_indicator(value)
                if kind in found:
                    found[kind].add(value.lower() if kind != "ip" else value)
    found["ip"] = {ip for ip in found["ip"] if not is_internal_ip(ip)}
    return {k: sorted(v) for k, v in found.items()}


class InvestigatorAgent(Agent):
    name = "Investigator"
    role = "Builds the event timeline, correlates evidence and forms a hypothesis."

    def gather(self, alert: Alert) -> list[LogEvent]:
        """Two-hop pivot: alert entities, then entities seen with them."""
        minutes = self.settings.log_window_minutes
        first = self.tools.logs.window(
            alert.timestamp, minutes, user=alert.user, host=alert.host, src_ip=alert.src_ip,
            dst_ip=alert.dst_ip, domain=alert.domain,
        )
        pivots: dict[str, set[str]] = {"host": set(), "user": set()}
        for e in first:
            if e.event_type == "auth" and (e.status or "") in SUCCESS_STATUSES:
                if e.host:
                    pivots["host"].add(e.host)
                if e.user:
                    pivots["user"].add(e.user)
        seen = {e.event_id: e for e in first}
        for host in pivots["host"]:
            for e in self.tools.logs.window(alert.timestamp, minutes, host=host):
                seen.setdefault(e.event_id, e)
        return sorted(seen.values(), key=lambda e: (e.timestamp, e.event_id))

    def analyse(self, alert: Alert, events: list[LogEvent]) -> Investigation:
        findings: list[Finding] = []
        for detector in DETECTORS:
            findings.extend(detector(alert, events))
        findings.sort(key=lambda f: -f.weight)
        indicators = collect_indicators(alert, events)
        tactics = {
            t.tactic
            for t in (self.tools.mitre.get(f.technique_id) for f in findings if f.technique_id)
            if t
        }
        stage = next((t for t in reversed(TACTIC_ORDER) if t in tactics), "unknown")
        if findings:
            hypothesis = (
                f"Activity consistent with {stage.replace('-', ' ')}: "
                + "; ".join(f.detail for f in findings[:3])
                + "."
            )
        else:
            hypothesis = "No correlated malicious behaviour found around the alert."
        return Investigation(
            timeline=events,
            findings=findings,
            indicators=indicators,
            hypothesis=hypothesis,
            attack_stage=stage,
        )

    def run(self, state: CaseState) -> None:
        alert = state.alert
        self.say(state, "thought", "Pivoting on alert entities (user, host, IPs, domain) to build a timeline.")
        events = self.gather(alert)
        investigation = self.analyse(alert, events)
        self.say(
            state,
            "tool",
            f"query_logs + 2-hop pivot -> {len(events)} events; {len(investigation.findings)} detector hit(s)",
        )
        for finding in investigation.findings:
            self.say(state, "result", f"Finding [{finding.technique_id or '-'}] {finding.title}: {finding.detail}",
                     weight=finding.weight)
        llm = self.ask_json(
            state,
            "Write a concise investigative hypothesis for this alert based ONLY on the findings and "
            "timeline excerpt, name the most advanced attack stage, and list up to 3 open questions "
            "an adversarial reviewer should check.",
            {
                "alert": {"rule": alert.rule_name, "description": alert.description, **alert.entities()},
                "findings": [{"title": f.title, "detail": f.detail, "technique": f.technique_id} for f in investigation.findings],
                "timeline_excerpt": [e.short() for e in events[:25]],
            },
            HYPOTHESIS_SCHEMA,
        )
        if llm:
            investigation.hypothesis = llm["hypothesis"].strip() or investigation.hypothesis
            if llm["attack_stage"] in TACTIC_ORDER:
                investigation.attack_stage = llm["attack_stage"]
            investigation.open_questions = [q for q in llm["open_questions"] if q.strip()][:5]
            investigation.llm_used = True
        state.investigation = investigation
        self.say(state, "result", f"Hypothesis: {investigation.hypothesis}")
