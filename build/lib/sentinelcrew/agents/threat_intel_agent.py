"""Threat-Intel agent: enrich indicators and map behaviour to MITRE ATT&CK."""

from __future__ import annotations

from sentinelcrew.agents.base import Agent
from sentinelcrew.agents.investigator import collect_indicators
from sentinelcrew.models import IndicatorVerdict, IntelReport
from sentinelcrew.orchestration.state import CaseState

MAX_LOOKUPS = 25


class ThreatIntelAgent(Agent):
    name = "ThreatIntel"
    role = "Checks IPs, domains and hashes against intelligence and maps to MITRE ATT&CK."

    def enrich(self, indicators: dict[str, list[str]]) -> list[IndicatorVerdict]:
        results: list[IndicatorVerdict] = []
        for kind in ("hash", "domain", "ip"):
            for value in indicators.get(kind, []):
                if len(results) >= MAX_LOOKUPS:
                    return results
                results.append(self.tools.intel.lookup(value, kind))
        results.sort(key=lambda r: -r.reputation)
        return results

    def run(self, state: CaseState) -> None:
        alert = state.alert
        events = self.tools.logs.window(
            alert.timestamp, self.settings.log_window_minutes, user=alert.user, host=alert.host,
            src_ip=alert.src_ip, dst_ip=alert.dst_ip, domain=alert.domain,
        )
        indicators = collect_indicators(alert, events)
        total = sum(len(v) for v in indicators.values())
        self.say(state, "thought", f"Extracted {total} external indicator(s); querying intel sources.")
        results = self.enrich(indicators)
        for r in results:
            if r.malicious or r.reputation >= 0.5 or "allowlisted" in r.tags:
                self.say(
                    state,
                    "result",
                    f"{r.kind.upper()} {r.indicator}: reputation {r.reputation:.2f} "
                    f"[{', '.join(r.tags)}] {r.context}",
                )
        technique_ids = list(state.triage.techniques) if state.triage else []
        for tech in self.tools.mitre.map_text(" ".join(t for r in results for t in r.tags)):
            if tech.technique_id not in technique_ids:
                technique_ids.append(tech.technique_id)
        techniques = self.tools.mitre.describe(technique_ids)
        malicious = [r for r in results if r.malicious]
        max_rep = max((r.reputation for r in results), default=0.0)
        summary = (
            f"{len(malicious)} malicious indicator(s) of {len(results)} checked; max reputation {max_rep:.2f}"
        )
        state.intel = IntelReport(results=results, techniques=techniques, max_reputation=max_rep, summary=summary)
        self.say(
            state,
            "result",
            summary + "; ATT&CK: " + (", ".join(f"{t['id']} {t['name']}" for t in techniques[:5]) or "none"),
        )
