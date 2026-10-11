"""Triage agent: hybrid ML anomaly score + expert rules -> priority & category."""

from __future__ import annotations

from sentinelcrew.agents.base import Agent
from sentinelcrew.ml.features import extract_features, to_vector
from sentinelcrew.models import Alert, LogEvent, Priority, TriageResult
from sentinelcrew.orchestration.state import CaseState
from sentinelcrew.scoring import clamp, noisy_or

# raw Isolation-Forest scores for normal data cluster around 0.40-0.50;
# map [ANOMALY_FLOOR, ANOMALY_CEIL] -> [0, 1]
ANOMALY_FLOOR = 0.45
ANOMALY_CEIL = 0.70
ML_WEIGHT = 0.4
RULE_WEIGHT = 0.6

# (feature, predicate, weight, technique, explanation)
RULES: tuple[tuple[str, float, float, str, str], ...] = (
    ("failed_auth_count", 10, 0.45, "T1110", "{v:.0f} failed authentications (brute-force pattern)"),
    ("distinct_users_from_src", 5, 0.40, "T1110.003", "{v:.0f} accounts targeted from one source (spraying)"),
    ("success_after_failures", 1, 0.45, "T1078", "successful logon after repeated failures"),
    ("beacon_regularity", 0.8, 0.50, "T1071.001", "highly periodic outbound connections (beacon {v:.2f})"),
    ("mb_out", 50, 0.45, "T1041", "{v:.1f} MB sent outbound"),
    ("suspicious_process", 1, 0.50, "T1059.001", "suspicious process / LOLBin execution"),
    ("distinct_dst_ports", 20, 0.30, "T1046", "{v:.0f} distinct destination ports (scanning)"),
)

FEATURE_CATEGORY = {
    "T1110": "credential-access",
    "T1110.003": "credential-access",
    "T1078": "initial-access",
    "T1071.001": "command-and-control",
    "T1041": "exfiltration",
    "T1059.001": "execution",
    "T1046": "discovery",
}


def normalise_anomaly(raw: float) -> float:
    return clamp((raw - ANOMALY_FLOOR) / (ANOMALY_CEIL - ANOMALY_FLOOR))


def priority_for(score: float, criticality: float) -> Priority:
    weighted = score * (0.75 + 0.5 * criticality)  # critical assets bump priority
    if weighted >= 0.80:
        return Priority.CRITICAL
    if weighted >= 0.60:
        return Priority.HIGH
    if weighted >= 0.40:
        return Priority.MEDIUM
    if weighted >= 0.20:
        return Priority.LOW
    return Priority.INFO


class TriageAgent(Agent):
    name = "Triage"
    role = "Normalises alerts, scores severity with ML + rules and filters noise."

    def window_events(self, alert: Alert) -> list[LogEvent]:
        return self.tools.logs.window(
            alert.timestamp,
            self.settings.log_window_minutes,
            user=alert.user,
            host=alert.host,
            src_ip=alert.src_ip,
            dst_ip=alert.dst_ip,
            domain=alert.domain,
        )

    def assess(self, alert: Alert, events: list[LogEvent]) -> TriageResult:
        asset = self.tools.context.asset(alert.host) or self.tools.context.asset(alert.dst_ip)
        criticality = asset.criticality_score if asset else 0.5
        features = extract_features(alert, events, criticality)
        raw = float(self.tools.model.score_samples([to_vector(features)])[0])
        anomaly = normalise_anomaly(raw)

        fired: list[tuple[float, str, str]] = []
        for feature, threshold, weight, technique, text in RULES:
            value = features.get(feature, 0.0)
            if value >= threshold:
                fired.append((weight, technique, text.format(v=value)))
        severity_weight = 0.25 * features["severity_hint"]
        rule_score = noisy_or([w for w, _, _ in fired] + [severity_weight])
        score = round(clamp(ML_WEIGHT * anomaly + RULE_WEIGHT * rule_score), 4)

        techniques = [t for _, t, _ in fired]
        mapped = self.tools.mitre.map_text(f"{alert.rule_name} {alert.description} {alert.process or ''}")
        techniques += [t.technique_id for t in mapped if t.technique_id not in techniques]
        category = FEATURE_CATEGORY.get(techniques[0], "unknown") if fired else (
            self.tools.mitre.primary_tactic(techniques) if techniques else "unknown"
        )
        reasons = [text for _, _, text in fired]
        explanation = self.tools.model.explain(to_vector(features), top_k=3)
        reasons.append(
            "ML anomaly {:.2f} (raw {:.3f}); top drivers: {}".format(
                anomaly, raw, ", ".join(f"{n} {w:.0%}" for n, w in explanation) or "n/a"
            )
        )
        return TriageResult(
            score=score,
            priority=priority_for(score, criticality),
            category=category,
            techniques=techniques,
            features=features,
            anomaly_score=round(anomaly, 4),
            rule_score=round(rule_score, 4),
            reasons=reasons,
            is_noise=score < self.settings.noise_threshold and not fired,
        )

    def run(self, state: CaseState) -> None:
        alert = state.alert
        self.say(state, "thought", f"New alert '{alert.rule_name}' from {alert.source}. Pulling context window.")
        events = self.window_events(alert)
        self.say(state, "tool", f"query_logs(window=±{self.settings.log_window_minutes}m) -> {len(events)} events")
        result = self.assess(alert, events)
        state.triage = result
        verdict = "NOISE - auto-closing" if result.is_noise else "escalating to investigation"
        self.say(
            state,
            "result",
            f"Score {result.score:.2f} (ML {result.anomaly_score:.2f} / rules {result.rule_score:.2f}), "
            f"priority {result.priority.value.upper()}, category {result.category}: {verdict}",
            score=result.score,
            reasons=result.reasons,
        )
