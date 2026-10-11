"""Confidence fusion: combine ML, evidence, intel and memory into one score.

``confidence = sum(w_i * component_i) + critic_adjustment`` (clamped).

* **triage**   - hybrid ML anomaly + rule score from the Triage agent
* **evidence** - noisy-OR of the Investigator's finding weights:
  ``1 - prod(1 - w)``; independent pieces of evidence reinforce each other
* **intel**    - highest reputation of any indicator in the case
* **memory**   - similarity-weighted verdicts of similar past cases
  (0.5 = neutral, >0.5 = past cases were real attacks)
"""

from __future__ import annotations

from typing import Iterable

from sentinelcrew.models import Assessment, Finding, SimilarCase, Verdict

DEFAULT_WEIGHTS: dict[str, float] = {"triage": 0.25, "evidence": 0.40, "intel": 0.25, "memory": 0.10}

MALICIOUS_VERDICTS = {Verdict.TRUE_POSITIVE.value}
BENIGN_VERDICTS = {Verdict.FALSE_POSITIVE.value, Verdict.NOISE.value}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def noisy_or(weights: Iterable[float]) -> float:
    product = 1.0
    for weight in weights:
        product *= 1.0 - clamp(weight)
    return 1.0 - product


def evidence_strength(findings: Iterable[Finding]) -> float:
    return noisy_or(f.weight for f in findings)


def memory_prior(similar: Iterable[SimilarCase]) -> float:
    """0.5 when there is no memory; pulled toward 1/0 by malicious/benign precedents."""
    numerator = 0.0
    denominator = 0.0
    for case in similar:
        if case.verdict in MALICIOUS_VERDICTS:
            target = 1.0
        elif case.verdict in BENIGN_VERDICTS:
            target = 0.0
        else:
            target = 0.5
        numerator += case.similarity * target
        denominator += case.similarity
    if denominator == 0:
        return 0.5
    # shrink toward neutral when similarity is low
    strength = clamp(denominator)
    return 0.5 + (numerator / denominator - 0.5) * strength


def decide(confidence: float, tp_threshold: float, fp_threshold: float) -> Verdict:
    if confidence >= tp_threshold:
        return Verdict.TRUE_POSITIVE
    if confidence < fp_threshold:
        return Verdict.FALSE_POSITIVE
    return Verdict.NEEDS_REVIEW


def fuse(
    triage_score: float,
    findings: list[Finding],
    intel_reputation: float | None,
    similar: list[SimilarCase],
    tp_threshold: float = 0.70,
    fp_threshold: float = 0.40,
    weights: dict[str, float] | None = None,
    adjustment: float = 0.0,
) -> Assessment:
    """Fuse the components. ``intel_reputation=None`` (no indicators) drops the
    intel term and re-normalises the remaining weights."""
    w = dict(weights or DEFAULT_WEIGHTS)
    components = {
        "triage": round(clamp(triage_score), 4),
        "evidence": round(evidence_strength(findings), 4),
        "memory": round(memory_prior(similar), 4),
    }
    if intel_reputation is None:
        w.pop("intel", None)
    else:
        components["intel"] = round(clamp(intel_reputation), 4)
    base = sum(w[k] * components[k] for k in components) / (sum(w[k] for k in components) or 1.0)
    confidence = round(clamp(base + adjustment, 0.01, 0.99), 4)
    rationale = [f"{k}={v:.2f} (w={w[k]:.2f})" for k, v in components.items()]
    if adjustment:
        rationale.append(f"critic adjustment {adjustment:+.2f}")
    return Assessment(
        confidence=confidence,
        verdict=decide(confidence, tp_threshold, fp_threshold),
        components=components | {"adjustment": round(adjustment, 4), "base": round(base, 4)},
        rationale=rationale,
    )
