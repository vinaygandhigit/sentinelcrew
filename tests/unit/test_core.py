"""Tests for scoring, memory, persistence, config, models and the graph engine."""

import os
import tempfile
import threading
import unittest
from enum import Enum
from pathlib import Path

import numpy as np

from tests import helpers
from tests.helpers import alert
from sentinelcrew.config import Settings, load_dotenv
from sentinelcrew.memory import CaseStore, TfidfEmbedder, VectorStore, cosine, tokenize
from sentinelcrew.models import (
    AgentMessage,
    CaseStatus,
    Finding,
    Priority,
    SimilarCase,
    TriageResult,
    Verdict,
    to_jsonable,
)
from sentinelcrew.orchestration import END, CaseState, GraphError, StateGraph
from sentinelcrew.scoring import decide, evidence_strength, fuse, memory_prior, noisy_or


class TestScoring(unittest.TestCase):
    def test_noisy_or(self):
        self.assertEqual(noisy_or([]), 0.0)
        self.assertAlmostEqual(noisy_or([0.5, 0.5]), 0.75)
        self.assertEqual(noisy_or([1.5]), 1.0)  # clamped

    def test_evidence(self):
        self.assertAlmostEqual(evidence_strength([Finding("a", "", 0.6), Finding("b", "", 0.5)]), 0.8)

    def test_memory_prior(self):
        self.assertEqual(memory_prior([]), 0.5)
        mal = [SimilarCase("H", 0.8, "t", "true_positive")]
        ben = [SimilarCase("H", 0.8, "t", "false_positive")]
        unk = [SimilarCase("H", 0.8, "t", "weird")]
        self.assertGreater(memory_prior(mal), 0.8)
        self.assertLess(memory_prior(ben), 0.2)
        self.assertEqual(memory_prior(unk), 0.5)
        self.assertLess(memory_prior([SimilarCase("H", 0.1, "t", "true_positive")]), 0.6)  # weak similarity shrinks

    def test_decide(self):
        self.assertEqual(decide(0.7, 0.7, 0.4), Verdict.TRUE_POSITIVE)
        self.assertEqual(decide(0.5, 0.7, 0.4), Verdict.NEEDS_REVIEW)
        self.assertEqual(decide(0.39, 0.7, 0.4), Verdict.FALSE_POSITIVE)

    def test_fuse(self):
        a = fuse(0.9, [Finding("a", "", 0.6)], 0.9, [], adjustment=0.0)
        self.assertEqual(a.verdict, Verdict.TRUE_POSITIVE)
        self.assertIn("intel", a.components)
        b = fuse(0.9, [Finding("a", "", 0.6)], None, [])
        self.assertNotIn("intel", b.components)
        c = fuse(0.9, [Finding("a", "", 0.6)], 0.9, [], adjustment=-0.6)
        self.assertEqual(c.verdict, Verdict.FALSE_POSITIVE)
        self.assertIn("critic adjustment", c.rationale[-1])
        self.assertGreaterEqual(fuse(0, [], 0, [], adjustment=-5).confidence, 0.01)
        self.assertLessEqual(fuse(1, [Finding("a", "", 1)], 1, [], adjustment=5).confidence, 0.99)


class TestVectorStore(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The C2 beacon, via update-cdn-sync.net!"), ["c2", "beacon", "update-cdn-sync.net"])

    def test_cosine(self):
        self.assertEqual(cosine({}, {"a": 1}), 0.0)
        self.assertAlmostEqual(cosine({"a": 1.0}, {"a": 1.0, "b": 0.0}), 1.0)

    def test_search_ranks_relevant_case(self):
        vs = VectorStore()
        vs.add_many([("A", "brute force failed logons tor exit", {"verdict": "true_positive"}),
                     ("B", "port scan authorized scanner", {"verdict": "false_positive"}),
                     ("C", "beacon c2 powershell", {})])
        vs.add("D", "nightly backup upload", {})
        hits = vs.search("many failed logons brute force", k=2)
        self.assertEqual(hits[0][0].record_id, "A")
        self.assertEqual(len(vs), 4)
        self.assertEqual(vs.search("brute force", exclude={"A"}, min_score=0.5), [])

    def test_embedder_empty(self):
        e = TfidfEmbedder()
        e.fit([])
        self.assertEqual(e.embed(""), {})


class TestCaseStore(unittest.TestCase):
    def _case(self, case_id="CASE-1", verdict=Verdict.FALSE_POSITIVE):
        from sentinelcrew.models import Assessment, Case

        triage = TriageResult(0.5, Priority.LOW, "x", [], {}, 0.1, 0.1, [], False)
        return Case(case_id=case_id, alert=alert(), status=CaseStatus.CLOSED_FALSE_POSITIVE, verdict=verdict,
                    confidence=0.2, priority=Priority.LOW, triage=triage,
                    initial_assessment=Assessment(0.5, Verdict.NEEDS_REVIEW), duration_ms=10)

    def test_save_get_list_metrics(self):
        s = helpers.store()
        s.save(self._case())
        s.save(self._case())  # upsert
        s.save(self._case("CASE-2", Verdict.TRUE_POSITIVE))
        self.assertEqual(s.count(), 2)
        self.assertEqual(s.get("CASE-1")["verdict"], "false_positive")
        self.assertIsNone(s.get("nope"))
        self.assertEqual(len(s.list(limit=1)), 1)
        m = s.metrics()
        self.assertEqual(m["total_cases"], 2)
        self.assertEqual(m["auto_closed"], 1)
        self.assertAlmostEqual(m["mean_debate_shift"], 0.3)

    def test_empty_metrics_and_decisions(self):
        s = helpers.store()
        self.assertEqual(s.metrics()["total_cases"], 0)
        s.record_decision("C", "A", "approve", "bob", "", "now")
        self.assertEqual(len(s.decisions("C")), 1)
        self.assertEqual(len(s.decisions()), 1)
        s.close()

    def test_file_backed(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = CaseStore(Path(tmp) / "sub" / "db.sqlite")
            s.save(self._case())
            s.close()
            self.assertEqual(CaseStore(Path(tmp) / "sub" / "db.sqlite").count(), 1)

    def test_rollback_on_error(self):
        s = helpers.store()
        with self.assertRaises(Exception):
            with s._tx() as cur:
                cur.execute("INSERT INTO decisions (case_id) VALUES ('x')")  # violates NOT NULL
        self.assertEqual(s.decisions(), [])


class TestConfig(unittest.TestCase):
    def test_defaults_and_overrides(self):
        s = Settings.from_env(env={"LLM_PROVIDER": "Gemini", "DEBATE_MAX_ROUNDS": "5", "TP_THRESHOLD": "bad",
                                   "LLM_MAX_RETRIES": "x"})
        self.assertEqual(s.resolved_provider(), "gemini")
        self.assertEqual(s.debate_max_rounds, 5)
        self.assertEqual(s.tp_threshold, 0.70)
        self.assertEqual(s.llm_max_retries, 2)
        self.assertTrue(str(s.db_path).endswith("sentinelcrew.db"))
        self.assertTrue(str(s.reports_dir).endswith("reports"))

    def test_google_api_key_alias(self):
        self.assertEqual(Settings.from_env(env={"GOOGLE_API_KEY": "g"}).gemini_api_key, "g")

    def test_auto_and_invalid(self):
        self.assertEqual(Settings(app_env="prod").resolved_provider(), "anthropic")
        self.assertEqual(Settings().resolved_provider(), "gemini")
        with self.assertRaises(ValueError):
            Settings(llm_provider="openai").resolved_provider()

    def test_load_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("# c\nexport SC_TEST_A=1\nSC_TEST_B=\"two\" # note\nbad line\n=x\n", encoding="utf-8")
            os.environ.pop("SC_TEST_A", None)
            os.environ["SC_TEST_B"] = "keep"
            loaded = load_dotenv(path)
            self.assertEqual(loaded, {"SC_TEST_A": "1", "SC_TEST_B": "two"})
            self.assertEqual(os.environ["SC_TEST_B"], "keep")
            load_dotenv(path, override=True)
            self.assertEqual(os.environ["SC_TEST_B"], "two")
            self.assertEqual(load_dotenv(Path(tmp) / "missing"), {})
            for key in ("SC_TEST_A", "SC_TEST_B"):
                os.environ.pop(key, None)


class TestModels(unittest.TestCase):
    def test_to_jsonable(self):
        class Color(Enum):
            RED = "red"

        data = to_jsonable({"a": (Color.RED, {1}), "n": np.float64(1.5), "m": AgentMessage("x", "y", "z")})
        self.assertEqual(data["a"], ["red", [1]])
        self.assertEqual(data["n"], 1.5)
        self.assertIsInstance(data["m"]["timestamp"], str)

    def test_alert_helpers(self):
        a = alert(user=None, host=None, src_ip=None)
        self.assertEqual(a.title(), "Multiple failed logons followed by success (unknown entity)")
        self.assertEqual(a.entities(), {})


class TestState(unittest.TestCase):
    def test_emit_and_listeners(self):
        state = CaseState("C", alert())
        seen = []
        state.listeners.append(seen.append)
        state.listeners.append(lambda m: 1 / 0)  # broken listener must not break the case
        state.emit("Agent", "thought", "hello", k=1)
        self.assertEqual(seen[0].data, {"k": 1})
        state.record_error("x", "boom")
        self.assertEqual(state.errors, ["x: boom"])
        self.assertEqual(state.debate_rounds, 0)


class TestGraph(unittest.TestCase):
    def test_linear_and_trace(self):
        g = StateGraph()
        g.add_node("a", lambda s: s.append("a")).add_node("b", lambda s: s.append("b"))
        g.set_entry_point("a").add_edge("a", "b").add_edge("b", END)
        trace = []
        self.assertEqual(g.run([], trace=trace), ["a", "b"])
        self.assertEqual([t["node"] for t in trace], ["a", "b"])
        self.assertIn("START", g.to_mermaid())
        self.assertEqual(g.describe()["entry"], "a")

    def test_conditional_loop_terminates(self):
        g = StateGraph()
        g.add_node("loop", lambda s: s.append(1))
        g.set_entry_point("loop")
        g.add_conditional_edges("loop", lambda s: "again" if len(s) < 3 else "stop", {"again": "loop", "stop": END})
        self.assertEqual(len(g.run([])), 3)
        self.assertIn("again", g.to_mermaid())

    def test_max_steps_guard(self):
        g = StateGraph()
        g.add_node("loop", lambda s: None).set_entry_point("loop")
        g.add_conditional_edges("loop", lambda s: "again", {"again": "loop"})
        with self.assertRaises(GraphError):
            g.run([], max_steps=5)

    def test_parallel_runs_concurrently(self):
        barrier = threading.Barrier(2, timeout=2)
        g = StateGraph()
        g.add_parallel_node("p", {"x": lambda s: (barrier.wait(), s.append("x")),
                                  "y": lambda s: (barrier.wait(), s.append("y"))})
        g.set_entry_point("p").add_edge("p", END)
        self.assertEqual(sorted(g.run([])), ["x", "y"])  # would deadlock if sequential
        self.assertIn("parallel", g.to_mermaid())

    def test_errors_are_routed_to_handler(self):
        def boom(s):
            raise RuntimeError("bad")

        g = StateGraph()
        g.add_node("a", boom).add_parallel_node("p", {"x": boom, "y": lambda s: s.append("y")})
        g.set_entry_point("a").add_edge("a", "p").add_edge("p", END)
        errors = []
        trace = []
        out = g.run([], trace=trace, on_error=lambda n, e, s: errors.append(n))
        self.assertEqual(errors, ["a", "p.x"])
        self.assertEqual(out, ["y"])
        self.assertTrue(trace[0]["status"].startswith("error"))
        with self.assertRaises(RuntimeError):
            g.run([])

    def test_parallel_error_without_handler(self):
        g = StateGraph()
        g.add_parallel_node("p", {"x": lambda s: 1 / 0}).set_entry_point("p").add_edge("p", END)
        with self.assertRaises(ZeroDivisionError):
            g.run([])

    def test_validation(self):
        cases = []
        g = StateGraph(); cases.append(g)  # no entry
        g = StateGraph(); g.add_node("a", print).set_entry_point("a"); cases.append(g)  # no edge
        g = StateGraph(); g.add_node("a", print).set_entry_point("a").add_edge("a", "zz"); cases.append(g)
        g = StateGraph(); g.add_node("a", print).set_entry_point("a").add_edge("b", END); g.add_edge("a", END); cases.append(g)
        g = StateGraph(); g.add_node("a", print).set_entry_point("a"); g.add_conditional_edges("a", str, {"x": "zz"}); cases.append(g)
        g = StateGraph(); g.add_node("a", print).set_entry_point("a"); g.add_edge("a", END); g.add_conditional_edges("q", str, {}); cases.append(g)
        for graph in cases:
            with self.assertRaises(GraphError):
                graph.validate()

    def test_build_errors(self):
        g = StateGraph()
        g.add_node("a", print)
        with self.assertRaises(GraphError):
            g.add_node("a", print)
        with self.assertRaises(GraphError):
            g.add_node(END, print)
        with self.assertRaises(GraphError):
            g.add_parallel_node("p", {})
        g.add_edge("a", END)
        with self.assertRaises(GraphError):
            g.add_edge("a", END)
        with self.assertRaises(GraphError):
            g.add_conditional_edges("a", str, {})

    def test_router_unknown_label(self):
        g = StateGraph()
        g.add_node("a", lambda s: None).set_entry_point("a")
        g.add_conditional_edges("a", lambda s: "nope", {"x": END})
        with self.assertRaises(GraphError):
            g.run([])


if __name__ == "__main__":
    unittest.main()