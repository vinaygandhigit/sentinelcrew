"""Resilience: real HTTP transport behaviour and agent failure isolation."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from tests import helpers
from tests.helpers import alert, bruteforce_events
from sentinelcrew.llm.http import TransportError, urllib_transport
from sentinelcrew.models import Verdict
from sentinelcrew.orchestration import CaseState
from sentinelcrew.orchestration.workflow import SOCWorkflow


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = body.get("mode")
        if mode == "ok":
            payload, status = json.dumps({"echo": body, "key": self.headers.get("x-api-key")}).encode(), 200
        elif mode == "bad-json":
            payload, status = b"not json", 200
        else:
            payload, status = b'{"error": "overloaded"}', int(mode)
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestUrllibTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_success_sends_headers_and_json(self):
        out = urllib_transport(self.url, {"x-api-key": "secret"}, {"mode": "ok"}, 5)
        self.assertEqual(out["key"], "secret")
        self.assertEqual(out["echo"]["mode"], "ok")

    def test_retryable_http_error(self):
        with self.assertRaises(TransportError) as ctx:
            urllib_transport(self.url, {}, {"mode": "503"}, 5)
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(ctx.exception.status, 503)

    def test_non_retryable_http_error(self):
        with self.assertRaises(TransportError) as ctx:
            urllib_transport(self.url, {}, {"mode": "401"}, 5)
        self.assertFalse(ctx.exception.retryable)

    def test_invalid_json(self):
        with self.assertRaises(TransportError):
            urllib_transport(self.url, {}, {"mode": "bad-json"}, 5)

    def test_network_error_is_retryable(self):
        with self.assertRaises(TransportError) as ctx:
            urllib_transport("http://127.0.0.1:9/", {}, {}, 1)
        self.assertTrue(ctx.exception.retryable)


class TestFailureIsolation(unittest.TestCase):
    def test_crashing_agent_does_not_kill_the_case(self):
        tk = helpers.toolkit(bruteforce_events(40))
        wf = SOCWorkflow(tk, helpers.settings())

        def broken(state):
            raise RuntimeError("intel feed down")

        wf.graph._nodes["analysis"].parallel["threat_intel"] = broken
        state = wf.run(CaseState("CASE-X", alert()))
        self.assertTrue(any("intel feed down" in e for e in state.errors))
        self.assertTrue(any(m.kind == "error" and m.agent == "Orchestrator" for m in state.messages))
        self.assertIsNone(state.intel)
        self.assertIsNotNone(state.investigation)  # the parallel sibling still ran
        self.assertEqual(state.assessment.verdict, Verdict.TRUE_POSITIVE)
        self.assertTrue(state.report_markdown)


if __name__ == "__main__":
    unittest.main()
