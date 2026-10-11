"""A from-scratch Isolation Forest (Liu, Ting & Zhou, 2008) in NumPy.

Implemented natively (rather than imported from scikit-learn) so that:

* the model is fully explainable - we can report *which* features caused
  early isolation (used in the Triage agent's reasoning);
* it can be serialised to plain JSON (no pickle -> no arbitrary code exec
  when loading a model file);
* the runtime dependency footprint stays tiny.

Anomaly score: ``s(x, n) = 2 ** (-E[h(x)] / c(n))`` where ``h(x)`` is the path
length of ``x`` in a tree and ``c(n)`` is the average path length of an
unsuccessful BST search. ``s -> 1`` means anomalous, ``s < 0.5`` means normal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

EULER_GAMMA = 0.5772156649015329


def average_path_length(n: int) -> float:
    """``c(n)`` - expected path length of an unsuccessful search in a BST."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1) + EULER_GAMMA) - 2.0 * (n - 1) / n


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None
    size: int = 0  # number of training samples at a leaf

    @property
    def is_leaf(self) -> bool:
        return self.left is None

    def to_dict(self) -> dict[str, Any]:
        if self.is_leaf:
            return {"s": self.size}
        assert self.left is not None and self.right is not None
        return {"f": self.feature, "t": self.threshold, "l": self.left.to_dict(), "r": self.right.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_Node":
        if "s" in data:
            return cls(size=int(data["s"]))
        return cls(
            feature=int(data["f"]),
            threshold=float(data["t"]),
            left=cls.from_dict(data["l"]),
            right=cls.from_dict(data["r"]),
        )


class IsolationForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        random_state: int | None = 42,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if max_samples < 2:
            raise ValueError("max_samples must be >= 2")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.feature_names = list(feature_names) if feature_names else []
        self.trees: list[_Node] = []
        self.sample_size = 0
        self.n_features = 0

    @property
    def fitted(self) -> bool:
        return bool(self.trees)

    def fit(self, X: np.ndarray) -> "IsolationForest":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be a 2-D array with at least 2 rows")
        rng = np.random.default_rng(self.random_state)
        self.n_features = X.shape[1]
        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(self.n_features)]
        self.sample_size = min(self.max_samples, X.shape[0])
        height_limit = int(math.ceil(math.log2(self.sample_size)))
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=self.sample_size, replace=False)
            self.trees.append(self._build(X[idx], 0, height_limit, rng))
        return self

    def _build(self, X: np.ndarray, depth: int, limit: int, rng: np.random.Generator) -> _Node:
        n = X.shape[0]
        if depth >= limit or n <= 1:
            return _Node(size=n)
        mins, maxs = X.min(axis=0), X.max(axis=0)
        splittable = np.flatnonzero(maxs > mins)
        if splittable.size == 0:
            return _Node(size=n)
        feature = int(rng.choice(splittable))
        threshold = float(rng.uniform(mins[feature], maxs[feature]))
        mask = X[:, feature] < threshold
        return _Node(
            feature=feature,
            threshold=threshold,
            left=self._build(X[mask], depth + 1, limit, rng),
            right=self._build(X[~mask], depth + 1, limit, rng),
        )

    @staticmethod
    def _path(x: np.ndarray, node: _Node, depth: int = 0, trace: list[int] | None = None) -> float:
        while not node.is_leaf:
            if trace is not None:
                trace.append(node.feature)
            node = node.left if x[node.feature] < node.threshold else node.right  # type: ignore[assignment]
            depth += 1
        return depth + average_path_length(node.size)

    def _check(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("model is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")
        return X

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in ``(0, 1]`` for every row (higher = more anomalous)."""
        X = self._check(X)
        c = average_path_length(self.sample_size)
        scores = np.empty(X.shape[0])
        for i, row in enumerate(X):
            mean_path = np.mean([self._path(row, tree) for tree in self.trees])
            scores[i] = 2.0 ** (-mean_path / c) if c > 0 else 0.5
        return scores

    def explain(self, x: Sequence[float], top_k: int = 3) -> list[tuple[str, float]]:
        """Which features isolate ``x``? Each tree credits the feature of the
        split that separated ``x`` into its leaf, weighted by ``1 / depth``
        (early isolation = strong evidence of anomaly)."""
        row = self._check(np.asarray(x))[0]
        weights = np.zeros(self.n_features)
        for tree in self.trees:
            trace: list[int] = []
            self._path(row, tree, trace=trace)
            if trace:
                weights[trace[-1]] += 1.0 / len(trace)
        total = weights.sum() or 1.0
        order = np.argsort(-weights)[:top_k]
        return [(self.feature_names[i], float(weights[i] / total)) for i in order if weights[i] > 0]

    # ---- persistence (JSON, no pickle) ---------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "sample_size": self.sample_size,
            "n_features": self.n_features,
            "trees": [t.to_dict() for t in self.trees],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IsolationForest":
        model = cls(
            n_estimators=int(data["n_estimators"]),
            max_samples=int(data["max_samples"]),
            random_state=data.get("random_state"),
            feature_names=data.get("feature_names"),
        )
        model.sample_size = int(data["sample_size"])
        model.n_features = int(data["n_features"])
        model.trees = [_Node.from_dict(t) for t in data["trees"]]
        return model

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IsolationForest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
