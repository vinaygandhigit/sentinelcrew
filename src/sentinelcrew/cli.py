"""Command-line interface.

    python -m sentinelcrew serve      [--host 0.0.0.0] [--port 8000]
    python -m sentinelcrew scenarios
    python -m sentinelcrew run        <scenario_id>|all [--quiet]
    python -m sentinelcrew analyze    <alerts.json> [--events logs.jsonl]
    python -m sentinelcrew evaluate
    python -m sentinelcrew train
    python -m sentinelcrew check-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from sentinelcrew import __version__
from sentinelcrew.config import Settings
from sentinelcrew.models import AgentMessage, Case

KIND_GLYPH = {"thought": "  ", "tool": " >", "result": " =", "challenge": " ?", "decision": " !", "error": " x"}


def _print_message(message: AgentMessage) -> None:
    print(f"{KIND_GLYPH.get(message.kind, '  ')} [{message.agent:<12}] {message.content}")


def _print_case(case: Case) -> None:
    before = case.initial_assessment.confidence if case.initial_assessment else case.confidence
    print(
        f"\n==> {case.case_id}: {case.verdict.value.upper()}  confidence {before:.2f} -> {case.confidence:.2f}  "
        f"status={case.status.value}  ({case.duration_ms:.0f} ms)"
    )
    for action in case.actions:
        print(f"    - {action.kind}({action.target}) [{action.risk.value}] -> {action.status.value}")


def _service(settings: Settings):  # noqa: ANN202
    from sentinelcrew.service import SentinelService

    return SentinelService(settings)


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:  # pragma: no cover - blocking
    from sentinelcrew.api.server import serve

    serve(_service(settings), args.host, args.port)
    return 0


def cmd_scenarios(args: argparse.Namespace, settings: Settings) -> int:
    for s in _service(settings).scenarios():
        print(f"{s.scenario_id:30} expected={s.expected_verdict:15} {s.title}")
    return 0


def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    service = _service(settings)
    print(f"LLM: {service.llm_note}")
    ids = [s.scenario_id for s in service.scenarios()] if args.scenario == "all" else [args.scenario]
    for scenario_id in ids:
        print(f"\n### Scenario {scenario_id}")
        case = service.run_scenario(scenario_id, listener=None if args.quiet else _print_message)
        _print_case(case)
        path = service.save_report(case.case_id)
        print(f"    report: {path}")
    return 0


def cmd_analyze(args: argparse.Namespace, settings: Settings) -> int:
    service = _service(settings)
    if args.events:
        events, errors = service.ingest_text(Path(args.events).read_text(encoding="utf-8"), kind="events")
        print(f"ingested {len(events)} events ({len(errors)} rejected)")
    alerts, errors = service.ingest_text(Path(args.alerts).read_text(encoding="utf-8"), kind="alerts")
    for error in errors:
        print(f"skipped: {error}", file=sys.stderr)
    for alert in alerts:
        case = service.process_alert(alert, listener=None if args.quiet else _print_message)
        _print_case(case)
    return 0 if alerts else 1


def cmd_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    from sentinelcrew.evaluation import evaluate, format_table

    print(format_table(evaluate(settings)))
    return 0


def cmd_train(args: argparse.Namespace, settings: Settings) -> int:
    from sentinelcrew.ml.train import main as train_main

    train_main(["--out", str(settings.model_path)])
    return 0


def cmd_check_llm(args: argparse.Namespace, settings: Settings) -> int:
    from sentinelcrew.llm.base import LLMError
    from sentinelcrew.llm.factory import create_llm

    try:
        llm = create_llm(settings)
        if llm is None:
            print("LLM_PROVIDER=offline - no LLM configured.")
            return 0
        answer = llm.complete_json(
            "You are a connectivity check.", "Reply with ok=true.",
            {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}},
        )
    except (LLMError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 2
    print(f"OK - {llm.name}:{llm.model} answered {answer}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinelcrew", description="Autonomous multi-agent SOC team")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--provider", choices=["auto", "gemini", "anthropic", "offline"],
                        help="override LLM_PROVIDER for this command")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="start the API + analyst console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)
    sub.add_parser("scenarios", help="list demo scenarios").set_defaults(fn=cmd_scenarios)
    p = sub.add_parser("run", help="run a demo scenario (or 'all')")
    p.add_argument("scenario")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("analyze", help="analyse alerts from a file")
    p.add_argument("alerts")
    p.add_argument("--events")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_analyze)
    sub.add_parser("evaluate", help="measure the effect of the Critic debate").set_defaults(fn=cmd_evaluate)
    sub.add_parser("train", help="(re)train the triage model").set_defaults(fn=cmd_train)
    sub.add_parser("check-llm", help="verify the configured LLM key works").set_defaults(fn=cmd_check_llm)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    if args.provider:
        settings = settings.with_overrides(llm_provider=args.provider)
    return int(args.fn(args, settings))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())