"""End-to-end tests: service, workflow, API, CLI and evaluation."""

import io
import json
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from http import HTTPStatus
from pathlib import Path

from tests import helpers
from sentinelcrew.api.server import SentinelAPI, create_server
from sentinelcrew.cli import main as cli_main
from sentinelcrew.evaluation import evaluate, format_table
from sentinelcrew.llm import ScriptedLLM
from sentinelcrew.models import ActionStatus, CaseStatus, Verdict
from sentinelcrew.service import SentinelService, ServiceError, derive_status

EXPECTED = {
    "01_bruteforce_compromise": Verdict.TRUE_POSITIVE,
    "02_scanner_false_positive": Verdict.FALSE_POSITIVE,
    "03_beacon_exfiltration": Verdict.TRUE_POSITIVE,
    "04_low_value_noise": Verdict.NOISE,
    "05_backup_false_positive": Verdict.FALSE_POSITIVE,
}


def make_service(tmp, llm=None, **overrides):
    return SentinelService(helpers.settings(Path(tmp), **overrides), llm=llm, store=helpers.store(),
                           model=helpers.small_model())


class TestScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.svc = make_service(cls.tmp.name)
        cls.cases = {sid: cls.svc.run_scenario(sid) for sid in EXPECTED}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_scenario_gets_the_expected_verdict(self):
        for sid, expected in EXPECTED.items():
            with self.subTest(sid):
                self.assertEqual(self.cases[sid].verdict, expected)
                self.assertFalse(self.cases[sid].metadata["errors"])

    def test_bruteforce_case_details(self):
        case = self.cases["01_bruteforce_compromise"]
        self.assertEqual(case.status, CaseStatus.AWAITING_APPROVAL)
        self.assertTrue(case.debate)
        self.assertIn("T1078", {f.technique_id for f in case.investigation.findings})
        self.assertIn("185.220.101.45", self.svc.environment.blocked_ips)
        agents = {m.agent for m in case.messages}
        self.assertTrue({"Triage", "Investigator", "ThreatIntel", "CaseMemory", "Critic", "Responder", "Reporter"} <= agents)
        nodes = [t["node"] for t in case.metadata["trace"]]
        self.assertEqual(nodes[:3], ["triage", "analysis", "memory"])
        self.assertEqual(nodes[-2:], ["respond", "report"])
        self.assertLessEqual(nodes.count("critic"), 3)

    def test_beacon_requires_host_isolation_approval(self):
        case = self.cases["03_beacon_exfiltration"]
        iso = [a for a in case.actions if a.kind == "isolate_host"]
        self.assertEqual(iso[0].status, ActionStatus.PENDING_APPROVAL)
        self.assertIn("update-cdn-sync.net", self.svc.environment.blocked_domains)

    def test_false_positives_shift_confidence_down(self):
        for sid in ("02_scanner_false_positive", "05_backup_false_positive"):
            case = self.cases[sid]
            self.assertLess(case.confidence, case.initial_assessment.confidence)
            self.assertEqual(case.actions, [])

    def test_noise_skips_agents(self):
        case = self.cases["04_low_value_noise"]
        nodes = [t["node"] for t in case.metadata["trace"]]
        self.assertEqual(nodes, ["triage", "close_noise", "report"])
        self.assertNotIn(case.case_id, self.svc.toolkit.memory.records)

    def test_cases_are_memorised_and_persisted(self):
        self.assertIn(self.cases["01_bruteforce_compromise"].case_id, self.svc.toolkit.memory.records)
        self.assertEqual(self.svc.store.count(), 5)
        m = self.svc.metrics()
        self.assertEqual(m["total_cases"], 5)
        self.assertGreater(m["auto_close_rate"], 0.5)
        self.assertGreaterEqual(m["pending_approvals"], 2)

    def test_report_saved(self):
        case = self.cases["01_bruteforce_compromise"]
        path = self.svc.save_report(case.case_id, Path(self.tmp.name) / "r")
        self.assertIn("Incident Report", path.read_text(encoding="utf-8"))
        with self.assertRaises(ServiceError):
            self.svc.save_report("CASE-9999")


class TestHumanInTheLoop(unittest.TestCase):
    def test_approve_and_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = make_service(tmp)
            case = svc.run_scenario("01_bruteforce_compromise")
            pending = [a for a in case.actions if a.status == ActionStatus.PENDING_APPROVAL]
            payload = svc.decide_action(case.case_id, pending[0].action_id, True, "alice", "confirmed with user")
            self.assertIn("j.mehta", svc.environment.disabled_accounts)
            self.assertEqual(payload["status"], "contained")
            self.assertEqual(svc.store.decisions(case.case_id)[0]["analyst"], "alice")
            with self.assertRaises(ServiceError):
                svc.decide_action(case.case_id, pending[0].action_id, True)  # already decided
            case2 = svc.run_scenario("03_beacon_exfiltration")
            iso = next(a for a in case2.actions if a.kind == "isolate_host")
            payload = svc.decide_action(case2.case_id, iso.action_id, False, "bob", "business critical")
            self.assertEqual(next(a for a in payload["actions"] if a["kind"] == "isolate_host")["status"], "rejected")
            self.assertNotIn("WS-ENG-12", svc.environment.isolated_hosts)
            with self.assertRaises(ServiceError):
                svc.decide_action("CASE-NOPE", "x", True)
            with self.assertRaises(ServiceError):
                svc.decide_action(case.case_id, "nope", True)

    def test_derive_status(self):
        self.assertEqual(derive_status(Verdict.NEEDS_REVIEW, []), CaseStatus.UNDER_REVIEW)
        self.assertEqual(derive_status(Verdict.TRUE_POSITIVE, [{"status": "approved"}]), CaseStatus.CONTAINED)


class TestWithLLM(unittest.TestCase):
    def test_full_pipeline_with_scripted_llm(self):
        llm = ScriptedLLM({
            "Investigator": {"hypothesis": "LLM: VPN brute force led to compromise.", "attack_stage": "initial-access",
                             "open_questions": ["MFA?"]},
            "Critic": lambda prompt: json.dumps({"commentary": "LLM critic", "selected_checks": [
                json.loads(prompt.split("CONTEXT (JSON):\n", 1)[1].split("\n\nRespond ONLY", 1)[0])["available_checks"].popitem()[0]]}),
            "Reporter": {"executive_summary": "LLM summary", "impact": "LLM impact", "recommendations": ["Rotate creds"]},
        })
        with tempfile.TemporaryDirectory() as tmp:
            svc = make_service(tmp, llm=llm)
            case = svc.run_scenario("01_bruteforce_compromise")
        self.assertEqual(case.verdict, Verdict.TRUE_POSITIVE)
        self.assertEqual(case.llm_provider, "scripted")
        self.assertIn("LLM summary", case.report_markdown)
        self.assertTrue(all(c.proposed_by == "llm" for r in case.debate for c in r.challenges))
        self.assertIn("usage", json.dumps(svc.metrics()["llm_usage"]) + "usage")

    def test_llm_outage_does_not_change_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = make_service(tmp, llm=ScriptedLLM(fail=True))
            for sid, expected in EXPECTED.items():
                self.assertEqual(svc.run_scenario(sid).verdict, expected)

    def test_auto_llm_without_key_runs_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = SentinelService(helpers.settings(Path(tmp), llm_provider="gemini", gemini_api_key=""),
                                  store=helpers.store(), model=helpers.small_model())
            self.assertIsNone(svc.llm)
            self.assertIn("offline", svc.llm_note)


class TestServiceMisc(unittest.TestCase):
    def test_ingest_text_and_unknown_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = make_service(tmp)
            events, errors = svc.ingest_text('{"timestamp": 0, "type": "auth"}\n{"bad": 1}\n', kind="events", fmt="jsonl")
            self.assertEqual((len(events), len(errors)), (1, 1))
            alerts, _ = svc.ingest_text('[{"timestamp": 0, "rule": "x"}]')
            self.assertEqual(len(alerts), 1)
            with self.assertRaises(ServiceError):
                svc.scenario("../../etc")

    def test_model_trained_and_cached_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = helpers.settings(Path(tmp))
            svc = SentinelService(s, llm=None, store=helpers.store(), load_history=False)
            self.assertTrue(s.model_path.exists())
            self.assertEqual(len(svc.toolkit.memory), 0)
            s.model_path.write_text("{broken", encoding="utf-8")
            SentinelService(s, llm=None, store=helpers.store(), load_history=False)  # corrupt file -> retrain

    def test_case_ids_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = helpers.store()
            svc = SentinelService(helpers.settings(Path(tmp)), llm=None, store=store, model=helpers.small_model())
            first = svc.run_scenario("04_low_value_noise").case_id
            svc2 = SentinelService(helpers.settings(Path(tmp)), llm=None, store=store, model=helpers.small_model())
            self.assertNotEqual(svc2.new_case_id(), first)


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.api = SentinelAPI(make_service(cls.tmp.name), synchronous=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def call(self, method, path, body=None, query=None):
        return self.api.dispatch(method, path, query or {}, body)

    def test_health_graph_scenarios(self):
        status, body, _ = self.call("GET", "/api/health")
        self.assertEqual((status, body["status"]), (HTTPStatus.OK, "ok"))
        self.assertIn("flowchart", self.call("GET", "/api/graph")[1]["mermaid"])
        self.assertEqual(len(self.call("GET", "/api/scenarios")[1]), 5)

    def test_run_scenario_flow(self):
        status, body, _ = self.call("POST", "/api/scenarios/03_beacon_exfiltration/run")
        self.assertEqual(status, HTTPStatus.ACCEPTED)
        status, run, _ = self.call("GET", f"/api/runs/{body['run_id']}", query={"since": ["2"]})
        self.assertTrue(run["done"])
        self.assertIsNone(run["error"])
        case_id = run["case_id"]
        status, case, _ = self.call("GET", f"/api/cases/{case_id}")
        self.assertEqual(case["verdict"], "true_positive")
        status, report, ctype = self.call("GET", f"/api/cases/{case_id}/report")
        self.assertTrue(ctype.startswith("text/markdown"))
        action = next(a for a in case["actions"] if a["status"] == "pending_approval")
        status, _, _ = self.call("POST", f"/api/cases/{case_id}/actions/{action['action_id']}", {"approve": True})
        self.assertEqual(status, HTTPStatus.OK)
        status, _, _ = self.call("POST", f"/api/cases/{case_id}/actions/{action['action_id']}", {"approve": True})
        self.assertEqual(status, HTTPStatus.CONFLICT)
        self.assertTrue(self.call("GET", "/api/cases", query={"limit": ["5"]})[1])
        self.assertIn("total_cases", self.call("GET", "/api/metrics")[1])

    def test_submit_alerts(self):
        body = {"alert": {"timestamp": "2026-10-09T10:00:00Z", "rule": "Port scan detected", "src_ip": "10.0.5.20",
                          "host": "SEC-SCAN-01"},
                "events": [{"timestamp": "2026-10-09T10:00:00Z", "type": "network", "src_ip": "10.0.5.20", "port": 22}]}
        status, out, _ = self.call("POST", "/api/alerts", body)
        self.assertEqual((status, out["events_ingested"]), (HTTPStatus.ACCEPTED, 1))
        run = self.call("GET", f"/api/runs/{out['run_ids'][0]}")[1]
        self.assertTrue(run["case_id"])

    def test_errors(self):
        checks = [
            (("GET", "/api/nope"), HTTPStatus.NOT_FOUND),
            (("DELETE", "/api/health"), HTTPStatus.METHOD_NOT_ALLOWED),
            (("POST", "/api/scenarios/nope/run"), HTTPStatus.NOT_FOUND),
            (("GET", "/api/runs/RUN-9999"), HTTPStatus.NOT_FOUND),
            (("GET", "/api/cases/CASE-9999"), HTTPStatus.NOT_FOUND),
            (("GET", "/api/cases/CASE-9999/report"), HTTPStatus.NOT_FOUND),
            (("POST", "/api/alerts", []), HTTPStatus.BAD_REQUEST),
            (("POST", "/api/alerts", {}), HTTPStatus.BAD_REQUEST),
            (("POST", "/api/alerts", {"alert": {"rule": "no timestamp"}}), HTTPStatus.UNPROCESSABLE_ENTITY),
            (("POST", "/api/cases/C/actions/A", {"approve": "yes"}), HTTPStatus.BAD_REQUEST),
            (("POST", "/api/cases/CASE-9999/actions/A", {"approve": True}), HTTPStatus.NOT_FOUND),
        ]
        for args, expected in checks:
            with self.subTest(args=args):
                self.assertEqual(self.call(*args)[0], expected)
        run_id = self.call("POST", "/api/scenarios/04_low_value_noise/run")[1]["run_id"]
        self.assertEqual(self.call("GET", f"/api/runs/{run_id}", query={"since": ["x"]})[0], HTTPStatus.BAD_REQUEST)
        self.assertEqual(self.call("GET", "/api/cases", query={"limit": ["x"]})[0], HTTPStatus.BAD_REQUEST)

    def test_failed_job_reports_error(self):
        def job(listener):
            raise RuntimeError("kaboom")

        run = self.api.runs.start("x", job)
        self.assertEqual(run.view(0)["error"], "kaboom")


class TestHTTPServer(unittest.TestCase):
    def test_real_http_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = create_server(make_service(tmp), "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                with urllib.request.urlopen(base + "/") as r:
                    self.assertIn(b"SentinelCrew", r.read())
                with urllib.request.urlopen(base + "/api/health") as r:
                    self.assertEqual(json.loads(r.read())["status"], "ok")
                req = urllib.request.Request(base + "/api/alerts", data=b"{not json", method="POST",
                                             headers={"Content-Length": "9"})
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req)
                self.assertEqual(ctx.exception.code, 400)
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(base + "/secret.txt")
                self.assertEqual(ctx.exception.code, 404)
                req = urllib.request.Request(base + "/api/scenarios/04_low_value_noise/run", data=b"", method="POST")
                with urllib.request.urlopen(req) as r:
                    self.assertEqual(r.status, 202)
            finally:
                server.shutdown()
                server.server_close()


class TestCLIAndEvaluation(unittest.TestCase):
    def test_evaluation_shows_critic_value(self):
        result = evaluate(helpers.settings())
        self.assertEqual(result["with_critic"]["accuracy"], 1.0)
        self.assertGreater(result["without_critic"]["false_escalations"], result["with_critic"]["false_escalations"])
        self.assertIn("with_critic", format_table(result))

    def test_cli_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_patch = {"SENTINEL_VAR_DIR": tmp, "LLM_PROVIDER": "offline"}
            import os

            old = {k: os.environ.get(k) for k in env_patch}
            os.environ.update(env_patch)
            try:
                alerts = Path(tmp) / "alerts.json"
                alerts.write_text(json.dumps([{"timestamp": "2026-10-09T10:00:00Z", "rule": "x", "user": "k.rao"}]))
                bad = Path(tmp) / "bad.json"
                bad.write_text("[]")
                events = Path(tmp) / "e.jsonl"
                events.write_text('{"timestamp": "2026-10-09T10:00:00Z", "type": "auth", "user": "k.rao"}\n')
                out = io.StringIO()
                with redirect_stdout(out):
                    self.assertEqual(cli_main(["scenarios"]), 0)
                    self.assertEqual(cli_main(["--provider", "offline", "run", "02_scanner_false_positive"]), 0)
                    self.assertEqual(cli_main(["run", "all", "--quiet"]), 0)
                    self.assertEqual(cli_main(["analyze", str(alerts), "--events", str(events)]), 0)
                    self.assertEqual(cli_main(["analyze", str(bad)]), 1)
                    self.assertEqual(cli_main(["check-llm"]), 0)
                    self.assertEqual(cli_main(["--provider", "gemini", "check-llm"]), 2 if not os.environ.get("GEMINI_API_KEY") else 0)
                    self.assertEqual(cli_main(["train"]), 0)
                    self.assertEqual(cli_main(["evaluate"]), 0)
                text = out.getvalue()
                self.assertIn("TRUE_POSITIVE", text)
                self.assertIn("[Critic", text)
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
