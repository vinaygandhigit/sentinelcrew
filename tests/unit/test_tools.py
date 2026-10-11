import unittest
from datetime import timedelta

from tests import helpers
from tests.helpers import DATA, T0, ev
from sentinelcrew.models import RiskLevel
from sentinelcrew.tools import (
    ACTION_CATALOG,
    ActionExecutor,
    ActionNotAllowed,
    LocalIntelFeed,
    LogStore,
    MitreMapper,
    OrgContext,
    ThreatIntelService,
    classify_indicator,
    extract_indicators,
    is_internal_ip,
)


class TestLogStore(unittest.TestCase):
    def setUp(self):
        self.store = LogStore([
            ev(3, 3, user="bob", src_ip="1.1.1.9"),
            ev(1, 1, user="Alice", host="H1"),
            ev(2, 2, user="bob", event_type="network"),
        ])

    def test_sorted_and_dedup(self):
        self.assertEqual([e.event_id for e in self.store.all()], ["E0001", "E0002", "E0003"])
        self.assertEqual(self.store.add_many([ev(1, 1)]), 0)
        self.assertEqual(len(self.store), 3)
        self.store.add(ev(9, 0.5))
        self.assertEqual(self.store.all()[0].event_id, "E0009")
        self.assertIsNotNone(self.store.get("E0009"))

    def test_entity_or_query_case_insensitive(self):
        ids = [e.event_id for e in self.store.query(user="alice", src_ip="1.1.1.9")]
        self.assertEqual(ids, ["E0001", "E0003"])

    def test_time_type_limit(self):
        self.assertEqual(len(self.store.query(start=T0 + timedelta(minutes=2))), 2)
        self.assertEqual(len(self.store.query(end=T0 + timedelta(minutes=1))), 1)
        self.assertEqual(len(self.store.query(event_types=["NETWORK"])), 1)
        self.assertEqual(len(self.store.query(limit=1)), 1)

    def test_window(self):
        self.assertEqual(len(self.store.window(T0 + timedelta(minutes=2), 1, user="bob")), 2)

    def test_unknown_field(self):
        with self.assertRaises(ValueError):
            self.store.query(color="red")


class TestThreatIntel(unittest.TestCase):
    def setUp(self):
        self.feed = LocalIntelFeed.from_file(DATA / "intel" / "feed.json")
        self.svc = ThreatIntelService([self.feed])

    def test_classify(self):
        self.assertEqual(classify_indicator("8.8.8.8"), "ip")
        self.assertEqual(classify_indicator("evil.xyz"), "domain")
        self.assertEqual(classify_indicator("a" * 64), "hash")
        self.assertEqual(classify_indicator("j.mehta"), "unknown")
        self.assertEqual(classify_indicator("x.dll"), "unknown")

    def test_internal(self):
        self.assertTrue(is_internal_ip("10.1.2.3"))
        self.assertFalse(is_internal_ip("185.220.101.45"))
        self.assertFalse(is_internal_ip("not-an-ip"))

    def test_extract(self):
        found = extract_indicators(["beacon to update-cdn-sync.net at 45.137.21.9 hash " + "b" * 64, "", "999.1.1.1"])
        self.assertIn("update-cdn-sync.net", found["domain"])
        self.assertIn("45.137.21.9", found["ip"])
        self.assertEqual(len(found["hash"]), 1)
        self.assertNotIn("999.1.1.1", found["ip"])

    def test_known_bad(self):
        v = self.svc.lookup("185.220.101.45")
        self.assertTrue(v.malicious)
        self.assertIn("tor-exit", v.tags)

    def test_subdomain_matches_parent(self):
        self.assertTrue(self.svc.lookup("a.b.update-cdn-sync.net", "domain").malicious)

    def test_allowlist_and_internal_and_unknown(self):
        self.assertIn("allowlisted", self.svc.lookup("outlook.office365.com").tags)
        self.assertTrue(self.svc.is_allowlisted("x.github.com"))
        self.assertIn("internal", self.svc.lookup("10.0.0.5").tags)
        unknown = self.svc.lookup("151.101.1.140")
        self.assertFalse(unknown.malicious)
        self.assertIn("unknown", unknown.tags)

    def test_multiple_sources_merge(self):
        extra = LocalIntelFeed.from_dict({"indicators": [{"indicator": "185.220.101.45", "reputation": 0.5, "tags": ["x"], "source": "other"}]})
        v = ThreatIntelService([self.feed, extra]).lookup("185.220.101.45")
        self.assertEqual(v.reputation, 0.92)
        self.assertEqual(set(v.sources), {"local-feed", "other"})
        self.assertIn("x", v.tags)


class TestMitre(unittest.TestCase):
    def setUp(self):
        self.m = MitreMapper.from_file(DATA / "mitre" / "techniques.json")

    def test_map_text(self):
        ids = [t.technique_id for t in self.m.map_text("encoded powershell -enc beacon")]
        self.assertIn("T1059.001", ids)
        self.assertIn("T1071.001", ids)
        self.assertEqual(self.m.map_text(""), [])

    def test_describe_and_tactic(self):
        d = self.m.describe(["T1110", "T1110", "", "T9999"])
        self.assertEqual([x["id"] for x in d], ["T1110", "T9999"])
        self.assertEqual(d[1]["tactic"], "unknown")
        self.assertEqual(self.m.primary_tactic(["T9999", "T1046"]), "discovery")
        self.assertEqual(self.m.primary_tactic([]), "unknown")


class TestContext(unittest.TestCase):
    def setUp(self):
        self.ctx = OrgContext.from_file(DATA / "context" / "org.json")

    def test_asset_lookup(self):
        self.assertEqual(self.ctx.asset("10.0.5.20").hostname, "SEC-SCAN-01")
        self.assertEqual(self.ctx.asset("ws-fin-07").criticality_score, 0.75)
        self.assertIsNone(self.ctx.asset(None))
        self.assertTrue(self.ctx.has_tag("SEC-SCAN-01", "authorized-scanner"))
        self.assertFalse(self.ctx.has_tag("nope", "x"))

    def test_identity(self):
        self.assertTrue(self.ctx.identity("P.NAIR").privileged)
        self.assertIsNone(self.ctx.identity(None))

    def test_change_windows(self):
        from sentinelcrew.ingestion import parse_timestamp

        inside = parse_timestamp("2026-10-07T21:00:00Z")
        self.assertEqual(self.ctx.active_changes(inside, "10.0.5.20")[0].change_id, "CHG-2041")
        self.assertEqual(self.ctx.active_changes(inside, "WS-FIN-07"), [])
        self.assertEqual(self.ctx.active_changes(parse_timestamp("2026-10-09T00:00:00Z"), "10.0.5.20"), [])

    def test_global_change_window(self):
        from sentinelcrew.ingestion import parse_timestamp

        ctx = OrgContext.from_dict({"changes": [{"change_id": "C", "start": "2026-01-01T00:00:00Z",
                                                  "end": "2026-01-02T00:00:00Z", "description": "all"}]})
        self.assertEqual(len(ctx.active_changes(parse_timestamp("2026-01-01T05:00:00Z"), "anything")), 1)


class TestActions(unittest.TestCase):
    def setUp(self):
        self.ex = ActionExecutor()

    def test_catalog_risk_levels(self):
        self.assertEqual(ACTION_CATALOG["block_ip"].risk, RiskLevel.LOW)
        self.assertEqual(ACTION_CATALOG["isolate_host"].risk, RiskLevel.HIGH)

    def test_execute_all_actions(self):
        for kind, target in [("block_ip", "1.2.3.4"), ("block_domain", "Evil.com"), ("isolate_host", "WS1"),
                             ("disable_account", "bob"), ("reset_sessions", "bob"), ("add_to_watchlist", "x"),
                             ("create_ticket", "CASE-1")]:
            self.assertIn("[SIMULATED]", self.ex.execute(kind, target))
        snap = self.ex.env.snapshot()
        self.assertEqual(snap["blocked_domains"], ["evil.com"])
        self.assertEqual(snap["tickets"], ["INC-1001:CASE-1"])
        self.assertEqual(len(self.ex.env.audit_log), 7)

    def test_allowlist_guardrail(self):
        with self.assertRaises(ActionNotAllowed):
            self.ex.execute("format_disk", "WS1")
        with self.assertRaises(ActionNotAllowed):
            self.ex.execute("block_ip", " ")

    def test_protected_targets(self):
        for kind, target in [("isolate_host", "DC01"), ("disable_account", "Administrator"), ("block_ip", "8.8.8.8")]:
            with self.assertRaises(ActionNotAllowed):
                self.ex.execute(kind, target)
        self.assertIn("ticket", self.ex.execute("create_ticket", "dc01"))


if __name__ == "__main__":
    unittest.main()
