"""Simulated response actions with a strict allow-list and risk policy.

Nothing here touches real infrastructure: every action mutates an in-memory
:class:`SimulatedEnvironment` (firewall block-list, EDR isolation set, IdP
disabled accounts ...). The allow-list is the security boundary - an agent (or
an LLM hallucination) can never trigger an action that is not registered here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

from sentinelcrew.models import RiskLevel


class ActionNotAllowed(PermissionError):
    """Raised when an action is not on the allow-list or the target is protected."""


@dataclass
class SimulatedEnvironment:
    blocked_ips: set[str] = field(default_factory=set)
    blocked_domains: set[str] = field(default_factory=set)
    isolated_hosts: set[str] = field(default_factory=set)
    disabled_accounts: set[str] = field(default_factory=set)
    reset_sessions: set[str] = field(default_factory=set)
    watchlist: set[str] = field(default_factory=set)
    tickets: list[str] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                "blocked_ips": sorted(self.blocked_ips),
                "blocked_domains": sorted(self.blocked_domains),
                "isolated_hosts": sorted(self.isolated_hosts),
                "disabled_accounts": sorted(self.disabled_accounts),
                "reset_sessions": sorted(self.reset_sessions),
                "watchlist": sorted(self.watchlist),
                "tickets": list(self.tickets),
            }


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    risk: RiskLevel
    description: str
    target_type: str  # ip | domain | host | user | case


ACTION_CATALOG: dict[str, ActionSpec] = {
    "add_to_watchlist": ActionSpec("add_to_watchlist", RiskLevel.LOW, "Add entity to watchlist", "any"),
    "create_ticket": ActionSpec("create_ticket", RiskLevel.LOW, "Open an ITSM ticket", "case"),
    "block_ip": ActionSpec("block_ip", RiskLevel.LOW, "Block external IP at perimeter firewall", "ip"),
    "block_domain": ActionSpec("block_domain", RiskLevel.LOW, "Sinkhole domain at DNS", "domain"),
    "reset_sessions": ActionSpec("reset_sessions", RiskLevel.MEDIUM, "Revoke sessions & force MFA", "user"),
    "disable_account": ActionSpec("disable_account", RiskLevel.HIGH, "Disable user account in IdP", "user"),
    "isolate_host": ActionSpec("isolate_host", RiskLevel.HIGH, "Network-isolate host via EDR", "host"),
}

# Targets that must never be actioned automatically, regardless of approval.
PROTECTED_TARGETS = {"administrator", "krbtgt", "dc01", "10.0.0.1", "8.8.8.8", "1.1.1.1"}


class ActionExecutor:
    def __init__(
        self,
        environment: SimulatedEnvironment | None = None,
        protected: set[str] | None = None,
    ) -> None:
        self.env = environment or SimulatedEnvironment()
        self.protected = {p.lower() for p in (protected or PROTECTED_TARGETS)}
        self._handlers: dict[str, Callable[[str], str]] = {
            "add_to_watchlist": self._watch,
            "create_ticket": self._ticket,
            "block_ip": self._block_ip,
            "block_domain": self._block_domain,
            "reset_sessions": self._reset_sessions,
            "disable_account": self._disable,
            "isolate_host": self._isolate,
        }

    @staticmethod
    def spec(kind: str) -> ActionSpec:
        try:
            return ACTION_CATALOG[kind]
        except KeyError as exc:
            raise ActionNotAllowed(f"action '{kind}' is not on the allow-list") from exc

    def validate(self, kind: str, target: str) -> ActionSpec:
        spec = self.spec(kind)
        if not target or not str(target).strip():
            raise ActionNotAllowed("action target is empty")
        if str(target).lower() in self.protected and kind != "create_ticket":
            raise ActionNotAllowed(f"target '{target}' is protected and cannot be actioned")
        return spec

    def execute(self, kind: str, target: str) -> str:
        self.validate(kind, target)
        result = self._handlers[kind](target)
        with self.env._lock:
            self.env.audit_log.append(f"{kind}:{target}")
        return result

    def _watch(self, target: str) -> str:
        with self.env._lock:
            self.env.watchlist.add(target)
        return f"[SIMULATED] {target} added to watchlist"

    def _ticket(self, target: str) -> str:
        with self.env._lock:
            ticket_id = f"INC-{len(self.env.tickets) + 1001}"
            self.env.tickets.append(f"{ticket_id}:{target}")
        return f"[SIMULATED] ticket {ticket_id} opened for {target}"

    def _block_ip(self, target: str) -> str:
        with self.env._lock:
            self.env.blocked_ips.add(target)
        return f"[SIMULATED] firewall rule added: deny any <-> {target}"

    def _block_domain(self, target: str) -> str:
        with self.env._lock:
            self.env.blocked_domains.add(target.lower())
        return f"[SIMULATED] DNS sinkhole created for {target}"

    def _reset_sessions(self, target: str) -> str:
        with self.env._lock:
            self.env.reset_sessions.add(target)
        return f"[SIMULATED] sessions revoked and MFA re-registration forced for {target}"

    def _disable(self, target: str) -> str:
        with self.env._lock:
            self.env.disabled_accounts.add(target)
        return f"[SIMULATED] account {target} disabled in identity provider"

    def _isolate(self, target: str) -> str:
        with self.env._lock:
            self.env.isolated_hosts.add(target)
        return f"[SIMULATED] host {target} network-isolated via EDR"
