"""Deterministically (re)generate the synthetic demo data set.

Creates ``data/logs/baseline.jsonl`` (a week of normal enterprise activity)
and one folder per demo scenario under ``data/scenarios/`` containing
``scenario.json`` (metadata + the triggering alert) and ``logs.jsonl``
(the telemetry around the incident).

Usage:  python scripts/generate_data.py
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UTC = timezone.utc

USERS = [
    ("j.mehta", "WS-FIN-07", "10.0.1.37"),
    ("a.sharma", "WS-HR-03", "10.0.1.53"),
    ("r.iyer", "WS-ENG-12", "10.0.3.12"),
    ("k.rao", "WS-SAL-21", "10.0.1.81"),
    ("p.nair", "SRV-WEB-01", "10.0.2.15"),
]
BENIGN_DOMAINS = ["outlook.office365.com", "www.google.com", "github.com", "login.salesforce.com", "zoom.us"]
BENIGN_PROCS = ["chrome.exe", "outlook.exe", "teams.exe", "excel.exe", "code.exe", "explorer.exe"]


class Writer:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.n = 0
        self.rows: list[dict] = []

    def add(self, ts: datetime, event_type: str, **fields: object) -> None:
        self.n += 1
        row = {"event_id": f"{self.prefix}-{self.n:05d}", "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "event_type": event_type}
        row.update({k: v for k, v in fields.items() if v is not None})
        self.rows.append(row)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.sort(key=lambda r: (r["timestamp"], r["event_id"]))
        path.write_text("\n".join(json.dumps(r) for r in self.rows) + "\n", encoding="utf-8")


def baseline(rng: random.Random) -> Writer:
    w = Writer("BASE")
    start = datetime(2026, 10, 1, tzinfo=UTC)
    for day in range(9):
        date = start + timedelta(days=day)
        if date.weekday() >= 5:
            continue
        for user, host, ip in USERS:
            t = date + timedelta(hours=8, minutes=rng.randint(0, 50))
            if rng.random() < 0.1:
                w.add(t - timedelta(seconds=30), "auth", user=user, host=host, src_ip=ip, status="failure",
                      logon_type="interactive")
            w.add(t, "auth", user=user, host=host, src_ip=ip, status="success", logon_type="interactive")
            for _ in range(rng.randint(4, 8)):
                t2 = t + timedelta(minutes=rng.randint(5, 500))
                w.add(t2, "process", user=user, host=host, process=rng.choice(BENIGN_PROCS))
                w.add(t2 + timedelta(seconds=5), "network", user=user, host=host, src_ip=ip,
                      domain=rng.choice(BENIGN_DOMAINS), dst_port=443, bytes_out=rng.randint(2_000, 900_000))
    return w


def scenario_bruteforce(rng: random.Random) -> tuple[dict, Writer]:
    w = Writer("S1")
    attacker = "185.220.101.45"
    t0 = datetime(2026, 10, 8, 2, 4, tzinfo=UTC)
    for i in range(46):
        w.add(t0 + timedelta(seconds=7 * i + rng.randint(0, 3)), "auth", user="j.mehta", host="VPN-GW-01",
              src_ip=attacker, status="failure", logon_type="vpn", reason="bad password")
    for i, user in enumerate(["a.sharma", "k.rao"]):
        w.add(t0 + timedelta(seconds=3 + 11 * i), "auth", user=user, host="VPN-GW-01", src_ip=attacker,
              status="failure", logon_type="vpn")
    t_success = t0 + timedelta(minutes=6, seconds=12)
    w.add(t_success, "auth", user="j.mehta", host="VPN-GW-01", src_ip=attacker, status="success", logon_type="vpn")
    w.add(t_success + timedelta(minutes=1), "auth", user="j.mehta", host="WS-FIN-07", src_ip="10.8.0.14",
          status="success", logon_type="remote-interactive")
    w.add(t_success + timedelta(minutes=3), "process", user="j.mehta", host="WS-FIN-07", process="outlook.exe")
    w.add(t_success + timedelta(minutes=5), "network", user="j.mehta", host="WS-FIN-07", src_ip="10.0.1.37",
          domain="outlook.office365.com", dst_port=443, bytes_out=4_800_000)
    alert = {
        "alert_id": "ALR-1001",
        "timestamp": (t_success + timedelta(seconds=40)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Azure AD / VPN",
        "rule_name": "Multiple failed logons followed by success",
        "description": f"46 failed VPN logons for j.mehta from {attacker} followed by a successful logon",
        "severity": "high",
        "user": "j.mehta",
        "host": "VPN-GW-01",
        "src_ip": attacker,
    }
    meta = {
        "title": "Brute-force -> account compromise",
        "description": "Off-hours VPN brute-force from a Tor exit node succeeds against a finance user.",
        "expected_verdict": "true_positive",
        "alert": alert,
    }
    return meta, w


def scenario_scanner(rng: random.Random) -> tuple[dict, Writer]:
    w = Writer("S2")
    scanner, target = "10.0.5.20", "10.0.2.15"
    t0 = datetime(2026, 10, 7, 21, 0, tzinfo=UTC)
    ports = rng.sample(range(1, 10000), 140)
    for i, port in enumerate(ports):
        w.add(t0 + timedelta(seconds=2 * i), "network", host="SEC-SCAN-01", src_ip=scanner, dst_ip=target,
              dst_port=port, bytes_out=rng.randint(60, 400), status="allowed")
    for i in range(6):
        w.add(t0 + timedelta(minutes=6, seconds=10 * i), "auth", user="svc-scanner", host="SRV-WEB-01",
              src_ip=scanner, status="failure", logon_type="network", reason="probe default creds")
    alert = {
        "alert_id": "ALR-1002",
        "timestamp": (t0 + timedelta(minutes=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Network IDS",
        "rule_name": "Port scan detected",
        "description": f"Port scan from {scanner} against {target}: 140 ports in 5 minutes",
        "severity": "high",
        "host": "SEC-SCAN-01",
        "src_ip": scanner,
        "dst_ip": target,
    }
    meta = {
        "title": "Noisy false positive: authorised vulnerability scanner",
        "description": "IT Security's scheduled scan trips the IDS port-scan rule during an approved change.",
        "expected_verdict": "false_positive",
        "alert": alert,
    }
    return meta, w


def scenario_beacon(rng: random.Random) -> tuple[dict, Writer]:
    w = Writer("S3")
    host, user, ip = "WS-ENG-12", "r.iyer", "10.0.3.12"
    c2_domain, c2_ip = "update-cdn-sync.net", "45.137.21.9"
    t0 = datetime(2026, 10, 9, 10, 2, tzinfo=UTC)
    w.add(t0, "process", user=user, host=host, process="winword.exe")
    w.add(t0 + timedelta(seconds=20), "process", user=user, host=host,
          process="powershell.exe -nop -w hidden -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA",
          parent="winword.exe", sha256="e3b98a4da31a127d4bde6e43033f66ba274cab0eb7eb1c70ec41402bf6273dd8")
    w.add(t0 + timedelta(seconds=25), "dns", user=user, host=host, src_ip=ip, domain=c2_domain, dst_ip=c2_ip)
    t = t0 + timedelta(seconds=30)
    for i in range(36):
        t += timedelta(seconds=60 + rng.uniform(-2, 2))
        w.add(t, "network", user=user, host=host, src_ip=ip, dst_ip=c2_ip, domain=c2_domain, dst_port=443,
              bytes_out=rng.randint(300, 900))
    w.add(t + timedelta(minutes=2), "process", user=user, host=host,
          process="rundll32.exe C:\\Users\\Public\\x.dll,Start")
    for i in range(5):
        w.add(t + timedelta(minutes=4, seconds=30 * i), "network", user=user, host=host, src_ip=ip,
              dst_ip=c2_ip, domain=c2_domain, dst_port=443, bytes_out=52_000_000 + rng.randint(0, 900_000))
    alert = {
        "alert_id": "ALR-1003",
        "timestamp": (t + timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "EDR",
        "rule_name": "Periodic outbound connections to rare domain",
        "description": f"{host} made 41 connections to {c2_domain} with fixed interval; encoded PowerShell parent",
        "severity": "medium",
        "user": user,
        "host": host,
        "domain": c2_domain,
        "dst_ip": c2_ip,
        "process": "powershell.exe -nop -w hidden -enc SQBFAFgA...",
    }
    meta = {
        "title": "Malware beaconing + data exfiltration",
        "description": "Macro-launched encoded PowerShell beacons to a C2 domain then exfiltrates ~260 MB.",
        "expected_verdict": "true_positive",
        "alert": alert,
    }
    return meta, w


def scenario_noise(rng: random.Random) -> tuple[dict, Writer]:
    w = Writer("S4")
    t0 = datetime(2026, 10, 9, 9, 15, tzinfo=UTC)
    w.add(t0, "auth", user="a.sharma", host="WS-HR-03", src_ip="10.0.1.53", status="failure", logon_type="interactive")
    w.add(t0 + timedelta(seconds=20), "auth", user="a.sharma", host="WS-HR-03", src_ip="10.0.1.53", status="success",
          logon_type="interactive")
    alert = {
        "alert_id": "ALR-1004",
        "timestamp": (t0 + timedelta(seconds=25)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Windows Security Log",
        "rule_name": "Single failed interactive logon",
        "description": "One failed logon for a.sharma on WS-HR-03",
        "severity": "info",
        "user": "a.sharma",
        "host": "WS-HR-03",
        "src_ip": "10.0.1.53",
    }
    meta = {
        "title": "Low-value noise",
        "description": "A single mistyped password - closed at triage without spending agent time.",
        "expected_verdict": "noise",
        "alert": alert,
    }
    return meta, w


def scenario_backup(rng: random.Random) -> tuple[dict, Writer]:
    w = Writer("S5")
    t0 = datetime(2026, 10, 9, 1, 30, tzinfo=UTC)
    for i in range(6):
        w.add(t0 + timedelta(minutes=4 * i + rng.randint(0, 2)), "network", user="svc-backup", host="SRV-BKP-01",
              src_ip="10.0.2.40", domain="backup.corp-cloud.com", dst_port=443,
              bytes_out=60_000_000 + rng.randint(0, 5_000_000))
    alert = {
        "alert_id": "ALR-1005",
        "timestamp": (t0 + timedelta(minutes=26)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "DLP / Proxy",
        "rule_name": "Large outbound data transfer",
        "description": "SRV-BKP-01 uploaded ~380 MB to backup.corp-cloud.com outside business hours",
        "severity": "high",
        "user": "svc-backup",
        "host": "SRV-BKP-01",
        "domain": "backup.corp-cloud.com",
    }
    meta = {
        "title": "Off-hours bulk upload by backup service",
        "description": "Looks like exfiltration, but it is the nightly backup to a sanctioned cloud tenant.",
        "expected_verdict": "false_positive",
        "alert": alert,
    }
    return meta, w


SCENARIOS = {
    "01_bruteforce_compromise": scenario_bruteforce,
    "02_scanner_false_positive": scenario_scanner,
    "03_beacon_exfiltration": scenario_beacon,
    "04_low_value_noise": scenario_noise,
    "05_backup_false_positive": scenario_backup,
}


def main() -> None:
    rng = random.Random(1337)
    baseline(rng).dump(DATA / "logs" / "baseline.jsonl")
    for name, builder in SCENARIOS.items():
        meta, writer = builder(rng)
        folder = DATA / "scenarios" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "scenario.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        writer.dump(folder / "logs.jsonl")
        print(f"{name}: {len(writer.rows)} events")
    alerts = [json.loads((DATA / "scenarios" / n / "scenario.json").read_text())["alert"] for n in SCENARIOS]
    (DATA / "samples").mkdir(parents=True, exist_ok=True)
    (DATA / "samples" / "alerts.json").write_text(json.dumps(alerts, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
