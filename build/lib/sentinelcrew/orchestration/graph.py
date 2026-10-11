"""A minimal, LangGraph-style state-graph engine.

Supports the three control-flow primitives the SOC workflow needs:

* **sequential edges**      ``add_edge("a", "b")``
* **conditional routing**   ``add_conditional_edges("a", router, {"x": "b", "y": "c"})``
  (this is what implements the Critic's debate *loop*)
* **parallel fan-out/in**   ``add_parallel_node("analysis", {...})`` runs several
  agent functions concurrently on a thread pool and joins them.

Safety: a hard ``max_steps`` budget guarantees termination even if a router
misbehaves, and every node execution is timed and traced.

The engine is ~150 lines, has zero dependencies and is fully unit tested.
Its API intentionally mirrors LangGraph so the workflow can be ported
to LangGraph with minimal changes if desired.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

S = TypeVar("S")

END = "__end__"
NodeFn = Callable[[S], None]
Router = Callable[[S], str]


class GraphError(RuntimeError):
    pass


@dataclass
class _Node(Generic[S]):
    name: str
    fn: Callable[[S], None] | None = None
    parallel: dict[str, Callable[[S], None]] | None = None


class StateGraph(Generic[S]):
    def __init__(self, max_workers: int = 4) -> None:
        self._nodes: dict[str, _Node[S]] = {}
        self._edges: dict[str, str] = {}
        self._conditional: dict[str, tuple[Router[S], dict[str, str]]] = {}
        self._entry: str | None = None
        self.max_workers = max_workers

    # ---- building ------------------------------------------------------
    def add_node(self, name: str, fn: Callable[[S], None]) -> "StateGraph[S]":
        self._check_new(name)
        self._nodes[name] = _Node(name=name, fn=fn)
        return self

    def add_parallel_node(self, name: str, branches: dict[str, Callable[[S], None]]) -> "StateGraph[S]":
        self._check_new(name)
        if not branches:
            raise GraphError("parallel node needs at least one branch")
        self._nodes[name] = _Node(name=name, parallel=dict(branches))
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph[S]":
        if source in self._edges or source in self._conditional:
            raise GraphError(f"node '{source}' already has an outgoing edge")
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: Router[S], mapping: dict[str, str]
    ) -> "StateGraph[S]":
        if source in self._edges or source in self._conditional:
            raise GraphError(f"node '{source}' already has an outgoing edge")
        self._conditional[source] = (router, dict(mapping))
        return self

    def set_entry_point(self, name: str) -> "StateGraph[S]":
        self._entry = name
        return self

    def _check_new(self, name: str) -> None:
        if not name or name == END:
            raise GraphError("invalid node name")
        if name in self._nodes:
            raise GraphError(f"duplicate node '{name}'")

    def validate(self) -> None:
        if self._entry is None or self._entry not in self._nodes:
            raise GraphError("entry point is not set or unknown")
        for source, target in self._edges.items():
            if source not in self._nodes:
                raise GraphError(f"edge from unknown node '{source}'")
            if target != END and target not in self._nodes:
                raise GraphError(f"edge to unknown node '{target}'")
        for source, (_, mapping) in self._conditional.items():
            if source not in self._nodes:
                raise GraphError(f"conditional edge from unknown node '{source}'")
            for target in mapping.values():
                if target != END and target not in self._nodes:
                    raise GraphError(f"conditional edge to unknown node '{target}'")
        for name in self._nodes:
            if name not in self._edges and name not in self._conditional:
                raise GraphError(f"node '{name}' has no outgoing edge")

    # ---- running -------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "entry": self._entry,
            "nodes": {
                n: (sorted(node.parallel) if node.parallel else "fn") for n, node in self._nodes.items()
            },
            "edges": dict(self._edges),
            "conditional": {k: dict(v[1]) for k, v in self._conditional.items()},
        }

    def to_mermaid(self) -> str:
        lines = ["flowchart TD", f"    START((start)) --> {self._entry}"]
        for name, node in self._nodes.items():
            if node.parallel:
                for branch in node.parallel:
                    lines.append(f"    {name} -.parallel.-> {name}_{branch}[{branch}]")
        for source, target in self._edges.items():
            lines.append(f"    {source} --> {'END((end))' if target == END else target}")
        for source, (_, mapping) in self._conditional.items():
            for label, target in mapping.items():
                lines.append(f"    {source} -- {label} --> {'END((end))' if target == END else target}")
        return "\n".join(lines)

    def run(
        self,
        state: S,
        max_steps: int = 50,
        trace: list[dict[str, Any]] | None = None,
        on_error: Callable[[str, Exception, S], None] | None = None,
    ) -> S:
        self.validate()
        current = self._entry
        steps = 0
        while current != END:
            if steps >= max_steps:
                raise GraphError(f"max_steps={max_steps} exceeded (possible infinite loop)")
            steps += 1
            assert current is not None
            node = self._nodes[current]
            started = time.perf_counter()
            status = "ok"
            try:
                self._execute(node, state, on_error)
            except Exception as exc:  # noqa: BLE001
                status = f"error: {exc}"
                if on_error is None:
                    raise
                on_error(current, exc, state)
            if trace is not None:
                trace.append(
                    {
                        "step": steps,
                        "node": current,
                        "ms": round((time.perf_counter() - started) * 1000, 2),
                        "status": status,
                    }
                )
            current = self._next(current, state)
        return state

    def _execute(
        self, node: _Node[S], state: S, on_error: Callable[[str, Exception, S], None] | None
    ) -> None:
        if node.fn is not None:
            node.fn(state)
            return
        assert node.parallel is not None
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(node.parallel))) as pool:
            futures = {name: pool.submit(fn, state) for name, fn in node.parallel.items()}
            failures: list[tuple[str, Exception]] = []
            for name, future in futures.items():
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    failures.append((name, exc))
        for name, exc in failures:
            if on_error is None:
                raise exc
            on_error(f"{node.name}.{name}", exc, state)

    def _next(self, current: str, state: S) -> str:
        if current in self._edges:
            return self._edges[current]
        router, mapping = self._conditional[current]
        label = router(state)
        if label not in mapping:
            raise GraphError(f"router of '{current}' returned unknown label '{label}'")
        return mapping[label]
