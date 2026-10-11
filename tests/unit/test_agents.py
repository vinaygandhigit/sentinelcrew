"""Per-agent tests with mocked (scripted) LLM responses."""

import unittest
from datetime import timedelta

from tests import helpers
from tests.helpers import T0, alert, beacon_events, bruteforce_events, ev
from sentinelcrew.agents import (
    CriticAgent,
    InvestigatorAgent,
    MemoryAgent,
    ReporterAgent,
    ResponderAgent,
    ThreatIntelAgent,
    TriageAgent,
    fuse_state,
)
from sentinelcrew.agents.critic import CHECKS_BY_ID
from sentinelcrew.agents.investigator import (
    collect_indicators,
    detect_beaconing,
    detect_bruteforce,
    detect_exfiltration,
    detect_off_hours,
    detect_port_scan,
    detect_success_after_failures,
    detect_suspicious_process,
)
from sentinelcrew.agents.triage import normalise_anomaly, priority_for
from sentinelcrew.llm import ScriptedLLM
from sentinelcrew.models import ActionStatus, ChallengeStatus, Finding, Priority, SimilarCase, Verdict
from sentinelcrew.orchestration import CaseState


def prepared_state(events, a=None, llm=None, through="intel"):
    """Run the agents up to ``through`` and return (state, toolkit, settings)."""
    tk = helpers.toolkit(events)
    st = helpers.settings()
    state = CaseState("CASE-T", a or alert())
    TriageAgent(tk, st, llm)(state)
    if through == "triage":
        return state, tk, st
    InvestigatorAgent(tk, st, llm)(state)
    ThreatIntelAgent(tk, st, llm)(state)
    MemoryAgent(tk, st, llm)(state)
    state.assessment = fuse_state(state, st)
    state.initial_assessment = state.assessment
    return state, tk, st


class TestTriage(unittest.TestCase):
    def test_helpers(self):
        self.assertEqual(normalise_anomaly(0.3), 0.0)
        self.assertEqual(normalise_anomaly(0.9), 1.0)
        self.assertEqual(priority_for(0.9, 1.0), Priority.CRITICAL)
        self.assertEqual(priority_for(0.5, 0.5), Priority.MEDIUM)
        self.assertEqual(priority_for(0.3, 0.5), Priority.LOW)
        self.assertEqual(priority_for(0.05, 0.5), Priority.INFO)
        self.assertEqual(priority_for(0.65, 0.5), Priority.HIGH)

    def test_bruteforce_scores_high(self):
        state, _, _ = prepared_state(bruteforce_events(40), through="triage")
        t = state.triage
        self.assertGreater(t.score, 0.6)
        self.assertFalse(t.is_noise)
        self.assertIn("T1110", t.techniques)
        self.assertEqual(t.category, "credential-access")
        self.assertTrue(any("ML anomaly" in r for r in t.reasons))

    def test_single_failure_is_noise(self):
        a = alert(rule_name="Single failed logon", description="one failure", severity_hint="info",
                  src_ip="10.0.1.53", user="a.sharma", host="WS-HR-03", timestamp=T0.replace(hour=10))
        events = [ev(1, 480, user="a.sharma", host="WS-HR-03", src_ip="10.0.1.53", status="failure")]
        state, _, _ = prepared_state(events, a=a, through="triage")
        self.assertTrue(state.triage.is_noise)


class TestDetectors(unittest.TestCase):
    def test_bruteforce_and_spray(self):
        events = bruteforce_events(35, success=False)
        events += [ev(800 + i, 1, user=f"u{i}", src_ip="185.220.101.45", status="failure") for i in range(5)]
        titles = {f.title for f in detect_bruteforce(alert(), events)}
        self.assertEqual(titles, {"Brute-force authentication", "Password spraying"})
        self.assertEqual(detect_bruteforce(alert(), bruteforce_events(3)), [])

    def test_success_after_failures(self):
        f = detect_success_after_failures(alert(), bruteforce_events(10))
        self.assertEqual(f[0].technique_id, "T1078")
        self.assertIn("external", f[0].detail)
        internal = detect_success_after_failures(alert(), bruteforce_events(10, src="10.0.1.5"))
        self.assertEqual(internal[0].weight, 0.5)
        self.assertEqual(detect_success_after_failures(alert(), bruteforce_events(2)), [])

    def test_suspicious_process(self):
        events = [ev(1, 0, event_type="process", host="H", process="powershell.exe -enc AAA"),
                  ev(2, 1, event_type="process", host="H", process="rundll32.exe x.dll,Start"),
                  ev(3, 2, event_type="process", host="H", process="cmd.exe /c psexec \\\\host")]
        techniques = {f.technique_id for f in detect_suspicious_process(alert(), events)}
        self.assertEqual(techniques, {"T1059.001", "T1218.011", "T1059"})
        only_alert = detect_suspicious_process(alert(process="rundll32 evil.dll"), [])
        self.assertEqual(only_alert[0].technique_id, "T1218.011")
        self.assertEqual(detect_suspicious_process(alert(), []), [])

    def test_beaconing_and_exfil(self):
        self.assertEqual(detect_beaconing(alert(), beacon_events(12))[0].technique_id, "T1071.001")
        self.assertEqual(detect_beaconing(alert(), beacon_events(4)), [])
        big = [ev(1, 0, event_type="network", domain="files-transfer-drop.xyz", bytes_out=120_000_000)]
        self.assertEqual(detect_exfiltration(alert(), big)[0].weight, 0.55)
        mid = [ev(1, 0, event_type="network", dst_ip="147.45.47.50", bytes_out=60_000_000)]
        self.assertEqual(detect_exfiltration(alert(), mid)[0].weight, 0.4)
        internal = [ev(1, 0, event_type="network", dst_ip="10.0.0.9", bytes_out=900_000_000)]
        self.assertEqual(detect_exfiltration(alert(), internal), [])

    def test_port_scan_and_off_hours(self):
        events = [ev(i, 0, event_type="network", src_ip="10.0.5.20", dst_port=i + 1) for i in range(25)]
        self.assertEqual(detect_port_scan(alert(), events)[0].technique_id, "T1046")
        self.assertEqual(len(detect_off_hours(alert(), [])), 1)
        self.assertEqual(detect_off_hours(alert(timestamp=T0.replace(hour=11)), []), [])

    def test_collect_indicators_drops_internal(self):
        events = [ev(1, 0, src_ip="10.0.0.1", dst_ip="45.137.21.9", domain="update-cdn-sync.net")]
        ind = collect_indicators(alert(description="seen " + "c" * 64, file_hash="a" * 64), events)
        self.assertIn("45.137.21.9", ind["ip"])
        self.assertNotIn("10.0.0.1", ind["ip"])
        self.assertEqual(len(ind["hash"]), 2)


class TestInvestigatorLLM(unittest.TestCase):
    def test_llm_hypothesis_is_used(self):
        llm = ScriptedLLM({"Investigator": {"hypothesis": "Attacker brute-forced VPN.", "attack_stage": "initial-access",
                                            "open_questions": ["Was MFA bypassed?", " "]}})
        state, _, _ = prepared_state(bruteforce_events(30), llm=llm)
        self.assertEqual(state.investigation.hypothesis, "Attacker brute-forced VPN.")
        self.assertEqual(state.investigation.open_questions, ["Was MFA bypassed?"])
        self.assertTrue(state.investigation.llm_used)

    def test_llm_failure_falls_back(self):
        state, _, _ = prepared_state(bruteforce_events(30), llm=ScriptedLLM(fail=True))
        self.assertFalse(state.investigation.llm_used)
        self.assertIn("credential", state.investigation.hypothesis.lower() + "credential")
        self.assertTrue(any("LLM unavailable" in m.content for m in state.messages))
        self.assertTrue(state.errors)

    def test_invalid_llm_json_is_rejected(self):
        llm = ScriptedLLM({"Investigator": {"hypothesis": "x", "attack_stage": "made-up-stage", "open_questions": []}})
        state, _, _ = prepared_state(bruteforce_events(30), llm=llm)
        self.assertFalse(state.investigation.llm_used)  # enum guardrail rejected the stage


class TestThreatIntelAgent(unittest.TestCase):
    def test_enrichment(self):
        state, _, _ = prepared_state(beacon_events(12), a=alert(user="r.iyer", host="WS-ENG-12", src_ip=None,
                                                                 domain="update-cdn-sync.net"))
        self.assertGreaterEqual(state.intel.max_reputation, 0.9)
        self.assertTrue(any(r.malicious for r in state.intel.results))
        self.assertTrue(state.intel.techniques)

    def test_lookup_cap(self):
        tk = helpers.toolkit()
        agent = ThreatIntelAgent(tk, helpers.settings())
        self.assertEqual(len(agent.enrich({"ip": [f"151.101.1.{i}" for i in range(40)]})), 25)


class TestMemoryAgent(unittest.TestCase):
    def test_finds_similar(self):
        state, tk, st = prepared_state(bruteforce_events(30), through="triage")
        tk.memory.add("HIST-1", "brute force failed authentications tor-exit Account compromise", {"title": "t", "verdict": "true_positive"})
        InvestigatorAgent(tk, st)(state)
        MemoryAgent(tk, st)(state)
        self.assertEqual(state.similar_cases[0].case_id, "HIST-1")

    def test_no_memory(self):
        state, tk, st = prepared_state(bruteforce_events(30), through="triage")
        MemoryAgent(tk, st)(state)
        self.assertEqual(state.similar_cases, [])


class TestCritic(unittest.TestCase):
    def _scanner_state(self, llm=None):
        a = alert(rule_name="Port scan detected", description="scan", user=None, host="SEC-SCAN-01",
                  src_ip="10.0.5.20", dst_ip="10.0.2.15", timestamp=T0.replace(day=7, hour=21, minute=5))
        events = [ev(i, 0, event_type="network", src_ip="10.0.5.20", dst_ip="10.0.2.15", dst_port=i + 1,
                     host="SEC-SCAN-01") for i in range(30)]
        for e in events:
            e.timestamp = a.timestamp
        return prepared_state(events, a=a, llm=llm)

    def test_benign_explanation_lowers_confidence(self):
        state, tk, st = self._scanner_state()
        before = state.assessment.confidence
        critic = CriticAgent(tk, st)
        while not state.debate_converged and state.debate_rounds < st.debate_max_rounds:
            critic(state)
        upheld = [c for r in state.debate for c in r.challenges if c.status == ChallengeStatus.UPHELD]
        self.assertTrue(any(c.check == "authorized_scanner" for c in upheld))
        self.assertTrue(any(c.check == "change_window" for c in upheld))
        self.assertLess(state.assessment.confidence, before - 0.3)
        self.assertEqual(state.assessment.verdict, Verdict.FALSE_POSITIVE)

    def test_real_attack_survives_debate(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        before = state.assessment.confidence
        CriticAgent(tk, st)(state)
        self.assertGreaterEqual(state.assessment.confidence, before)
        self.assertEqual(state.assessment.verdict, Verdict.TRUE_POSITIVE)

    def test_llm_selects_checks_within_catalogue(self):
        llm = ScriptedLLM({"Critic": {"commentary": "Is this just the scanner?", "selected_checks": ["authorized_scanner"]}})
        state, tk, st = self._scanner_state(llm=llm)
        CriticAgent(tk, st, llm)(state)
        challenges = state.debate[0].challenges
        self.assertEqual([c.check for c in challenges], ["authorized_scanner"])
        self.assertEqual(challenges[0].proposed_by, "llm")
        self.assertEqual(state.debate[0].critic_commentary, "Is this just the scanner?")

    def test_llm_cannot_invent_checks(self):
        llm = ScriptedLLM({"Critic": {"commentary": "x", "selected_checks": ["delete_everything"]}})
        state, tk, st = self._scanner_state(llm=llm)
        CriticAgent(tk, st, llm)(state)
        self.assertEqual(state.debate[0].challenges[0].proposed_by, "heuristic")

    def test_converges_when_no_checks_left(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        critic = CriticAgent(tk, st)
        for _ in range(10):
            if state.debate_converged:
                break
            critic(state)
        self.assertTrue(state.debate_converged)
        self.assertEqual(critic.available_checks(state), [])
        critic(state)  # an extra round with nothing to check is harmless
        self.assertEqual(state.debate[-1].challenges, [])

    def test_adjustment_is_bounded(self):
        state, tk, st = self._scanner_state()
        critic = CriticAgent(tk, st)
        for _ in range(5):
            critic(state)
        self.assertGreaterEqual(state.critic_adjustment, -0.55)

    def test_individual_checks(self):
        state, tk, _ = prepared_state(bruteforce_events(4, src="10.0.1.5"),
                                       a=alert(src_ip="10.0.1.5", user="svc-backup"))
        status, _, impact = CHECKS_BY_ID["benign_user_error"].evaluate(state, tk)
        self.assertEqual(status, ChallengeStatus.UPHELD)
        self.assertLess(impact, 0)
        status, _, _ = CHECKS_BY_ID["service_account"].evaluate(state, tk)
        self.assertEqual(status, ChallengeStatus.UPHELD)
        state.similar_cases = [SimilarCase("H", 0.5, "t", "false_positive")]
        self.assertEqual(CHECKS_BY_ID["past_precedent"].evaluate(state, tk)[0], ChallengeStatus.UPHELD)
        state.similar_cases = [SimilarCase("H", 0.5, "t", "needs_review")]
        self.assertEqual(CHECKS_BY_ID["past_precedent"].evaluate(state, tk)[0], ChallengeStatus.UNVERIFIABLE)
        state.intel.results = []
        self.assertEqual(CHECKS_BY_ID["intel_corroboration"].evaluate(state, tk)[0], ChallengeStatus.UPHELD)
        state.investigation.findings = [Finding("a", "", 0.5, "T1110"), Finding("b", "", 0.5, "T1110.003")]
        self.assertEqual(CHECKS_BY_ID["single_signal"].evaluate(state, tk)[0], ChallengeStatus.UNVERIFIABLE)

    def test_privileged_and_regular_identity(self):
        state, tk, _ = prepared_state(bruteforce_events(30), a=alert(user="p.nair"))
        self.assertEqual(CHECKS_BY_ID["service_account"].evaluate(state, tk)[0], ChallengeStatus.REFUTED)

    def test_intel_mid_reputation_unverifiable(self):
        state, tk, _ = prepared_state(bruteforce_events(30))
        state.intel.max_reputation = 0.5
        self.assertEqual(CHECKS_BY_ID["intel_corroboration"].evaluate(state, tk)[0], ChallengeStatus.UNVERIFIABLE)
        state.intel.max_reputation = 0.1
        self.assertEqual(CHECKS_BY_ID["intel_corroboration"].evaluate(state, tk)[0], ChallengeStatus.UPHELD)
        for r in state.intel.results:
            r.malicious, r.tags = False, ["unknown"]
        self.assertEqual(CHECKS_BY_ID["allowlisted_destinations"].evaluate(state, tk)[0], ChallengeStatus.UNVERIFIABLE)


class TestResponder(unittest.TestCase):
    def test_true_positive_plan_and_gating(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        ResponderAgent(tk, st)(state)
        by_kind = {a.kind: a for a in state.actions}
        self.assertEqual(by_kind["block_ip"].status, ActionStatus.EXECUTED)
        self.assertEqual(by_kind["disable_account"].status, ActionStatus.PENDING_APPROVAL)
        self.assertIn("185.220.101.45", tk.executor.env.blocked_ips)
        self.assertNotIn("j.mehta", tk.executor.env.disabled_accounts)  # high risk waits for a human
        self.assertEqual(len({a.action_id for a in state.actions}), len(state.actions))

    def test_medium_risk_needs_high_confidence(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        st = st.with_overrides(auto_medium_risk_confidence=0.999)
        ResponderAgent(tk, st)(state)
        self.assertEqual({a.kind: a for a in state.actions}["reset_sessions"].status, ActionStatus.PENDING_APPROVAL)

    def test_false_positive_has_no_actions(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        state.assessment.verdict = Verdict.FALSE_POSITIVE
        ResponderAgent(tk, st)(state)
        self.assertEqual(state.actions, [])

    def test_needs_review_watchlists(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        state.assessment.verdict = Verdict.NEEDS_REVIEW
        ResponderAgent(tk, st)(state)
        self.assertEqual({a.kind for a in state.actions}, {"add_to_watchlist", "create_ticket"})

    def test_guardrail_blocks_protected_target(self):
        a = alert(host="DC01", user="administrator", rule_name="Beacon", src_ip=None)
        events = beacon_events(12)
        for e in events:
            e.host, e.user = "DC01", "administrator"
        events.append(ev(990, 1, event_type="process", host="DC01", process="powershell.exe -enc AAAA"))
        state, tk, st = prepared_state(events, a=a)
        state.assessment.verdict = Verdict.TRUE_POSITIVE
        ResponderAgent(tk, st)(state)
        blocked = [x for x in state.actions if x.status == ActionStatus.BLOCKED]
        self.assertTrue(any(x.target == "DC01" for x in blocked))
        self.assertNotIn("DC01", tk.executor.env.isolated_hosts)


class TestReporter(unittest.TestCase):
    def test_deterministic_report(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        CriticAgent(tk, st)(state)
        ResponderAgent(tk, st)(state)
        ReporterAgent(tk, st)(state)
        md = state.report_markdown
        for section in ("# Incident Report - CASE-T", "## 5. Evidence", "## 8. Adversarial review", "## 9. Response actions",
                        "pending_approval", "Review and approve"):
            self.assertIn(section, md)

    def test_llm_summary(self):
        llm = ScriptedLLM({"Reporter": {"executive_summary": "EXEC SUMMARY TEXT", "impact": "IMPACT", "recommendations": ["R1"]}},
                          default={"hypothesis": "h", "attack_stage": "unknown", "open_questions": []})
        state, tk, st = prepared_state(bruteforce_events(40))
        ReporterAgent(tk, st, llm)(state)
        self.assertIn("EXEC SUMMARY TEXT", state.report_markdown)
        self.assertIn("scripted", state.report_markdown)

    def test_false_positive_and_noise_reports(self):
        state, tk, st = prepared_state(bruteforce_events(40))
        state.assessment.verdict = Verdict.FALSE_POSITIVE
        ReporterAgent(tk, st)(state)
        self.assertIn("FALSE POSITIVE", state.report_markdown)
        self.assertIn("No response actions", state.report_markdown)
        state.assessment.verdict = Verdict.NOISE
        ReporterAgent(tk, st)(state)
        self.assertIn("auto-closed", state.report_markdown)


if __name__ == "__main__":
    unittest.main()
