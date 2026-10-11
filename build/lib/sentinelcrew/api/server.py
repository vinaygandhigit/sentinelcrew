"""REST API + analyst console, built on the standard library.

Zero web-framework dependencies keeps deployment trivial for any DevOps
pipeline: ``python -m sentinelcrew serve --host 0.0.0.0 --port 8000``.

Endpoints
---------
GET  /                                   analyst console (single-page app)
GET  /api/health                         liveness + active LLM provider
GET  /api/scenarios                      demo scenarios
POST /api/scenarios/{id}/run             start a scenario run  -> {run_id}
POST /api/alerts                         submit alert(s) (+ optional events) -> {run_ids}
GET  /api/runs/{run_id}?since=N          live agent messages since N (+ case_id when done)
GET  /api/cases                          case list
GET  /api/cases/{id}                     full case (evidence, debate, actions, report)
GET  /api/cases/{id}/report              incident report (text/markdown)
POST /api/cases/{id}/actions/{aid}       analyst decision {"approve": bool, "analyst", "comment"}
GET  /api/metrics                        SOC metrics
GET  /api/graph                          workflow graph (Mermaid)
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from sentinelcrew import __version__
from sentinelcrew.ingestion.normalizer import NormalizationError, normalize_alert, normalize_event
from sentinelcrew.models import AgentMessage, to_jsonable
from sentinelcrew.service import SentinelService, ServiceError

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 2_000_000


@dataclass
class Run:
    run_id: str
    label: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    case_id: str | None = None
    error: str | None = None
    done: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, message: AgentMessage) -> None:
        with self.lock:
            self.messages.append(to_jsonable(message))

    def view(self, since: int) -> dict[str, Any]:
        with self.lock:
            return {
                "run_id": self.run_id,
                "label": self.label,
                "messages": self.messages[since:],
                "next": len(self.messages),
                "done": self.done,
                "case_id": self.case_id,
                "error": self.error,
            }


class RunManager:
    """Executes cases on background threads so the UI can stream progress."""

    def __init__(self, service: SentinelService, synchronous: bool = False) -> None:
        self.service = service
        self.synchronous = synchronous
        self.runs: dict[str, Run] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._exec_lock = threading.Lock()  # one case at a time keeps the shared log store consistent

    def start(self, label: str, job: Callable[[Callable[[AgentMessage], None]], Any]) -> Run:
        with self._lock:
            run = Run(run_id=f"RUN-{next(self._ids):04d}", label=label)
            self.runs[run.run_id] = run

        def target() -> None:
            try:
                with self._exec_lock:
                    case = job(run.push)
                run.case_id = case.case_id
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                run.error = str(exc)
            finally:
                run.done = True

        if self.synchronous:
            target()
        else:
            threading.Thread(target=target, name=run.run_id, daemon=True).start()
        return run

    def get(self, run_id: str) -> Run | None:
        return self.runs.get(run_id)


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


Route = tuple[str, re.Pattern[str], str]
ROUTES: list[Route] = [
    ("GET", re.compile(r"^/api/health$"), "health"),
    ("GET", re.compile(r"^/api/scenarios$"), "list_scenarios"),
    ("POST", re.compile(r"^/api/scenarios/(?P<sid>[\w\-]+)/run$"), "run_scenario"),
    ("POST", re.compile(r"^/api/alerts$"), "submit_alerts"),
    ("GET", re.compile(r"^/api/runs/(?P<rid>[\w\-]+)$"), "get_run"),
    ("GET", re.compile(r"^/api/cases$"), "list_cases"),
    ("GET", re.compile(r"^/api/cases/(?P<cid>[\w\-]+)$"), "get_case"),
    ("GET", re.compile(r"^/api/cases/(?P<cid>[\w\-]+)/report$"), "get_report"),
    ("POST", re.compile(r"^/api/cases/(?P<cid>[\w\-]+)/actions/(?P<aid>[\w\-]+)$"), "decide"),
    ("GET", re.compile(r"^/api/metrics$"), "metrics"),
    ("GET", re.compile(r"^/api/graph$"), "graph"),
]


class SentinelAPI:
    """Framework-free request router (directly unit-testable)."""

    def __init__(self, service: SentinelService, synchronous: bool = False) -> None:
        self.service = service
        self.runs = RunManager(service, synchronous=synchronous)

    def dispatch(
        self, method: str, path: str, query: dict[str, list[str]], body: Any
    ) -> tuple[HTTPStatus, Any, str]:
        """Return ``(status, payload, content_type)``."""
        allowed_methods = set()
        for route_method, pattern, handler in ROUTES:
            match = pattern.match(path)
            if not match:
                continue
            allowed_methods.add(route_method)
            if route_method != method:
                continue
            try:
                return getattr(self, handler)(query=query, body=body, **match.groupdict())
            except ApiError as exc:
                return exc.status, {"error": exc.message}, "application/json"
            except ServiceError as exc:
                status = HTTPStatus.NOT_FOUND if "unknown" in str(exc) else HTTPStatus.CONFLICT
                return status, {"error": str(exc)}, "application/json"
        if allowed_methods:
            return HTTPStatus.METHOD_NOT_ALLOWED, {"error": f"use {sorted(allowed_methods)}"}, "application/json"
        return HTTPStatus.NOT_FOUND, {"error": f"no route for {path}"}, "application/json"

    # ---- handlers ------------------------------------------------------
    def health(self, **_: Any) -> tuple[HTTPStatus, Any, str]:
        return HTTPStatus.OK, {"status": "ok", "version": __version__, "llm": self.service.llm_note}, "application/json"

    def list_scenarios(self, **_: Any) -> tuple[HTTPStatus, Any, str]:
        items = [
            {"id": s.scenario_id, "title": s.title, "description": s.description,
             "expected_verdict": s.expected_verdict, "alert": s.alert.rule_name, "events": len(s.events)}
            for s in self.service.scenarios()
        ]
        return HTTPStatus.OK, items, "application/json"

    def run_scenario(self, sid: str, **_: Any) -> tuple[HTTPStatus, Any, str]:
        scenario = self.service.scenario(sid)  # raises ServiceError for unknown ids
        run = self.runs.start(scenario.title, lambda listener: self.service.run_scenario(sid, listener))
        return HTTPStatus.ACCEPTED, {"run_id": run.run_id}, "application/json"

    def submit_alerts(self, body: Any, **_: Any) -> tuple[HTTPStatus, Any, str]:
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        raw_alerts = body.get("alerts") or ([body["alert"]] if "alert" in body else [])
        if not raw_alerts or not isinstance(raw_alerts, list):
            raise ApiError(HTTPStatus.BAD_REQUEST, "provide 'alert' or a non-empty 'alerts' list")
        try:
            events = [normalize_event(e) for e in body.get("events", [])]
            alerts = [normalize_alert(a) for a in raw_alerts]
        except (NormalizationError, TypeError, AttributeError) as exc:
            raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, f"invalid input: {exc}") from exc
        self.service.ingest_events(events)
        run_ids = []
        for alert in alerts:
            run = self.runs.start(alert.rule_name, lambda listener, a=alert: self.service.process_alert(a, listener))
            run_ids.append(run.run_id)
        return HTTPStatus.ACCEPTED, {"run_ids": run_ids, "events_ingested": len(events)}, "application/json"

    def get_run(self, rid: str, query: dict[str, list[str]], **_: Any) -> tuple[HTTPStatus, Any, str]:
        run = self.runs.get(rid)
        if run is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown run {rid}")
        try:
            since = max(0, int(query.get("since", ["0"])[0]))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "since must be an integer") from exc
        return HTTPStatus.OK, run.view(since), "application/json"

    def list_cases(self, query: dict[str, list[str]], **_: Any) -> tuple[HTTPStatus, Any, str]:
        try:
            limit = min(500, max(1, int(query.get("limit", ["100"])[0])))
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "limit must be an integer") from exc
        return HTTPStatus.OK, self.service.store.list(limit), "application/json"

    def get_case(self, cid: str, **_: Any) -> tuple[HTTPStatus, Any, str]:
        payload = self.service.store.get(cid)
        if payload is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown case {cid}")
        return HTTPStatus.OK, payload, "application/json"

    def get_report(self, cid: str, **_: Any) -> tuple[HTTPStatus, Any, str]:
        payload = self.service.store.get(cid)
        if payload is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"unknown case {cid}")
        return HTTPStatus.OK, payload["report_markdown"], "text/markdown; charset=utf-8"

    def decide(self, cid: str, aid: str, body: Any, **_: Any) -> tuple[HTTPStatus, Any, str]:
        if not isinstance(body, dict) or not isinstance(body.get("approve"), bool):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must contain boolean 'approve'")
        analyst = str(body.get("analyst") or "analyst")[:64]
        comment = str(body.get("comment") or "")[:500]
        payload = self.service.decide_action(cid, aid, body["approve"], analyst, comment)
        return HTTPStatus.OK, payload, "application/json"

    def metrics(self, **_: Any) -> tuple[HTTPStatus, Any, str]:
        return HTTPStatus.OK, self.service.metrics(), "application/json"

    def graph(self, **_: Any) -> tuple[HTTPStatus, Any, str]:
        graph = self.service.workflow.graph
        return HTTPStatus.OK, {"mermaid": graph.to_mermaid(), "structure": graph.describe()}, "application/json"


def make_handler(api: SentinelAPI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"SentinelCrew/{__version__}"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            if "/api/runs/" not in (args[0] if args else ""):
                super().log_message(fmt, *args)

        def _send(self, status: HTTPStatus, payload: Any, content_type: str) -> None:
            if isinstance(payload, (bytes, bytearray)):
                data = bytes(payload)
            elif content_type.startswith("application/json"):
                data = json.dumps(payload, default=str).encode("utf-8")
            else:
                data = str(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("/", "/index.html") else None
            if name is None:
                self._send(HTTPStatus.NOT_FOUND, {"error": "not found"}, "application/json")
                return
            self._send(HTTPStatus.OK, (STATIC_DIR / name).read_bytes(), "text/html; charset=utf-8")

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            if method == "GET" and not parsed.path.startswith("/api/"):
                self._static(parsed.path)
                return
            body: Any = None
            if method == "POST":
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY_BYTES:
                    self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body too large"}, "application/json")
                    return
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else {}
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON body"}, "application/json")
                    return
            status, payload, content_type = api.dispatch(method, parsed.path, parse_qs(parsed.query), body)
            self._send(status, payload, content_type)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

    return Handler


def create_server(service: SentinelService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    api = SentinelAPI(service)
    return ThreadingHTTPServer((host, port), make_handler(api))


def serve(service: SentinelService, host: str = "127.0.0.1", port: int = 8000) -> None:  # pragma: no cover
    server = create_server(service, host, port)
    print(f"SentinelCrew console on http://{host}:{port}  (LLM: {service.llm_note})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
