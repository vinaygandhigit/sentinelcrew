"""Shared test fixtures (plain functions so they work with unittest and pytest)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentinelcrew.agents.base import Toolkit  # noqa: E402
from sentinelcrew.config import Settings  # noqa: E402
from sentinelcrew.memory.case_store import CaseStore  # noqa: E402
from sentinelcrew.memory.vector_store import VectorStore  # noqa: E402
from sentinelcrew.ml.isolation_forest import IsolationForest  # noqa: E402
from sentinelcrew.ml.train import train_model  # noqa: E402
from sentinelcrew.models import Alert, LogEvent  # noqa: E402
from sentinelcrew.tools.actions import ActionExecutor, SimulatedEnvironment  # noqa: E402
from sentinelcrew.tools.context import OrgContext  # noqa: E402
from sentinelcrew.tools.log_store import LogStore  # noqa: E402
from sentinelcrew.tools.mitre import MitreMapper  # noqa: E402
from sentinelcrew.tools.threat_intel import LocalIntelFeed, ThreatIntelService  # noqa: E402

DATA = ROOT / "data"
T0 = datetime(2026, 10, 8, 2, 0, tzinfo=timezone.utc)


def settings(tmp: Path | None = None, **overrides: object) -> Settings:
    base = Settings.from_env(env={"LLM_PROVIDER": "offline"})
    base = base.with_overrides(data_dir=DATA)
    if tmp is not None:
        base = base.with_overrides(var_dir=tmp)
    return base.with_overrides(**overrides) if overrides else base


@lru_cache(maxsize=1)
def small_model() -> IsolationForest:
    """A fast model shared across tests (training is deterministic)."""
    return train_model(n=600, n_estimators=40)


def toolkit(events: list[LogEvent] | None = None) -> Toolkit:
    return Toolkit(
        logs=LogStore(events or []),
        intel=ThreatIntelService([LocalIntelFeed.from_file(DATA / "intel" / "feed.json")]),
        mitre=MitreMapper.from_file(DATA / "mitre" / "techniques.json"),
        context=OrgContext.from_file(DATA / "context" / "org.json"),
        executor=ActionExecutor(SimulatedEnvironment()),
        memory=VectorStore(),
        model=small_model(),
    )


def store() -> CaseStore:
    return CaseStore(":memory:")


def alert(**kw: object) -> Alert:
    data: dict[str, object] = {
        "alert_id": "ALR-T1",
        "timestamp": T0 + timedelta(minutes=10),
        "source": "test",
        "rule_name": "Multiple failed logons followed by success",
        "description": "test alert",
        "severity_hint": "high",
        "user": "j.mehta",
        "host": "VPN-GW-01",
        "src_ip": "185.220.101.45",
    }
    data.update(kw)
    return Alert(**data)  # type: ignore[arg-type]


def ev(i: int, minutes: float = 0, **kw: object) -> LogEvent:
    data: dict[str, object] = {"event_id": f"E{i:04d}", "timestamp": T0 + timedelta(minutes=minutes), "event_type": "auth"}
    data.update(kw)
    return LogEvent(**data)  # type: ignore[arg-type]


def bruteforce_events(n_fail: int = 30, success: bool = True, src: str = "185.220.101.45") -> list[LogEvent]:
    events = [
        ev(i, minutes=i * 0.1, user="j.mehta", host="VPN-GW-01", src_ip=src, status="failure")
        for i in range(n_fail)
    ]
    if success:
        events.append(ev(n_fail, minutes=n_fail * 0.1 + 1, user="j.mehta", host="VPN-GW-01", src_ip=src, status="success"))
    return events


def beacon_events(n: int = 12, interval_s: float = 60, domain: str = "update-cdn-sync.net") -> list[LogEvent]:
    return [
        ev(500 + i, minutes=i * interval_s / 60, event_type="network", host="WS-ENG-12", user="r.iyer",
           src_ip="10.0.3.12", dst_ip="45.137.21.9", domain=domain, dst_port=443, bytes_out=500)
        for i in range(n)
    ]
