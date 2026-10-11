import json
import unittest
from datetime import datetime, timezone

from tests import helpers  # noqa: F401  (sets sys.path)
from sentinelcrew.ingestion import (
    NormalizationError,
    detect_format,
    normalize_alert,
    normalize_event,
    normalize_severity,
    parse_alerts,
    parse_events,
    parse_kv,
    parse_syslog_line,
    parse_timestamp,
)


class TestTimestamps(unittest.TestCase):
    def test_iso_with_z(self):
        self.assertEqual(parse_timestamp("2026-10-08T02:04:00Z"), datetime(2026, 10, 8, 2, 4, tzinfo=timezone.utc))

    def test_naive_is_utc(self):
        self.assertEqual(parse_timestamp("2026-10-08 02:04:00").tzinfo, timezone.utc)

    def test_offset_converted_to_utc(self):
        self.assertEqual(parse_timestamp("2026-10-08T07:34:00+05:30").hour, 2)

    def test_epoch_seconds_and_millis(self):
        self.assertEqual(parse_timestamp(0).year, 1970)
        self.assertEqual(parse_timestamp(1_700_000_000_000), parse_timestamp(1_700_000_000))
        self.assertEqual(parse_timestamp("1700000000"), parse_timestamp(1_700_000_000))

    def test_datetime_passthrough(self):
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(parse_timestamp(dt), dt)

    def test_invalid(self):
        for bad in (None, "", "yesterday-ish", [1]):
            with self.assertRaises(NormalizationError):
                parse_timestamp(bad)


class TestSeverity(unittest.TestCase):
    def test_words(self):
        self.assertEqual(normalize_severity("CRIT"), "critical")
        self.assertEqual(normalize_severity("Informational"), "info")
        self.assertEqual(normalize_severity("weird"), "medium")
        self.assertEqual(normalize_severity(None), "medium")

    def test_numeric_scales(self):
        self.assertEqual(normalize_severity(10), "critical")
        self.assertEqual(normalize_severity(7), "high")
        self.assertEqual(normalize_severity(5), "critical")  # 1-5 scale
        self.assertEqual(normalize_severity(2), "medium")
        self.assertEqual(normalize_severity(1), "low")
        self.assertEqual(normalize_severity(0), "info")


class TestNormalizeAlert(unittest.TestCase):
    def test_aliases_and_nesting(self):
        a = normalize_alert({
            "id": "A1", "@timestamp": "2026-10-08T02:00:00Z", "signature": "Port scan",
            "source": {"ip": "10.0.5.20"}, "destination_ip": "10.0.2.15", "dport": "443",
            "hostname": "SEC-SCAN-01", "level": "high",
        })
        self.assertEqual(a.alert_id, "A1")
        self.assertEqual(a.rule_name, "Port scan")
        self.assertEqual(a.src_ip, "10.0.5.20")
        self.assertEqual(a.dst_port, 443)
        self.assertEqual(a.severity_hint, "high")
        self.assertIn("host", a.entities())

    def test_generates_stable_id(self):
        rec = {"timestamp": "2026-10-08T02:00:00Z", "rule": "x"}
        self.assertEqual(normalize_alert(rec).alert_id, normalize_alert(dict(rec)).alert_id)
        self.assertTrue(normalize_alert(rec).alert_id.startswith("ALR-"))

    def test_rejects_bad_records(self):
        for bad in ({}, {"timestamp": "2026-10-08T02:00:00Z"}, {"rule": "x"}, "string"):
            with self.assertRaises(NormalizationError):
                normalize_alert(bad)  # type: ignore[arg-type]

    def test_bad_port_becomes_none(self):
        a = normalize_alert({"timestamp": 0, "rule": "x", "port": "abc"})
        self.assertIsNone(a.dst_port)

    def test_title(self):
        a = normalize_alert({"timestamp": 0, "rule": "Rule", "user": "bob"})
        self.assertEqual(a.title(), "Rule (bob)")


class TestNormalizeEvent(unittest.TestCase):
    def test_event_fields(self):
        e = normalize_event({"event_id": "E1", "time": 0, "type": "AUTH", "outcome": "FAILURE", "bytes_sent": "12",
                             "logon_type": "vpn"})
        self.assertEqual(e.event_type, "auth")
        self.assertEqual(e.status, "failure")
        self.assertEqual(e.bytes_out, 12)
        self.assertEqual(e.details, {"logon_type": "vpn"})
        self.assertIn("auth", e.short())

    def test_missing_type(self):
        with self.assertRaises(NormalizationError):
            normalize_event({"timestamp": 0})
        with self.assertRaises(NormalizationError):
            normalize_event({})


class TestParsers(unittest.TestCase):
    def test_kv(self):
        self.assertEqual(parse_kv('a=1 b="two words" c.d=x'), {"a": "1", "b": "two words", "c.d": "x"})

    def test_syslog_5424(self):
        rec = parse_syslog_line("<34>1 2026-10-08T02:00:00Z fw01 firewall - - - action=deny src=1.2.3.4 rule=\"Port scan\"")
        self.assertEqual(rec["host"], "fw01")
        self.assertEqual(rec["src"], "1.2.3.4")
        self.assertEqual(normalize_alert(rec).rule_name, "Port scan")

    def test_syslog_3164(self):
        rec = parse_syslog_line("Oct  8 02:00:00 web01 sshd: user=root status=failure", year=2026)
        self.assertEqual(rec["source"], "sshd")
        self.assertTrue(rec["timestamp"].startswith("2026-10-08"))

    def test_syslog_invalid(self):
        with self.assertRaises(NormalizationError):
            parse_syslog_line("garbage")

    def test_detect_format(self):
        self.assertEqual(detect_format("[]"), "json")
        self.assertEqual(detect_format('{"a":1}\n{"b":2}'), "jsonl")
        self.assertEqual(detect_format('{"a":1}'), "json")
        self.assertEqual(detect_format("a,b\n1,2"), "csv")
        self.assertEqual(detect_format("<34>1 2026 h a - - - x"), "syslog")
        self.assertEqual(detect_format("", filename="x.ndjson"), "ndjson")
        self.assertEqual(detect_format("", filename="x.log"), "syslog")

    def test_parse_alerts_formats_and_errors(self):
        good = {"timestamp": "2026-10-08T02:00:00Z", "rule": "r"}
        alerts, errors = parse_alerts(json.dumps({"alerts": [good, {"rule": "no-ts"}]}))
        self.assertEqual((len(alerts), len(errors)), (1, 1))
        alerts, _ = parse_alerts("timestamp,rule,user\n2026-10-08T02:00:00Z,r,bob\n", fmt="csv")
        self.assertEqual(alerts[0].user, "bob")
        alerts, _ = parse_alerts(json.dumps(good))
        self.assertEqual(len(alerts), 1)

    def test_parse_events_jsonl(self):
        text = "\n".join(json.dumps({"timestamp": i, "type": "auth"}) for i in range(3)) + "\n\n"
        events, errors = parse_events(text, fmt="jsonl")
        self.assertEqual((len(events), errors), (3, []))

    def test_unsupported_format(self):
        with self.assertRaises(NormalizationError):
            parse_alerts("x", fmt="xml")
        with self.assertRaises(NormalizationError):
            parse_alerts("42", fmt="json")


if __name__ == "__main__":
    unittest.main()
