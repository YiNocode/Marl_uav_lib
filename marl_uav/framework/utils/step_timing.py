"""Lightweight per-step timing with episode-level summary (avg / p95 / max)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TimingBucket:
    """Accumulates millisecond samples for one timed section."""

    samples_ms: list[float] = field(default_factory=list)

    def record(self, ms: float) -> None:
        self.samples_ms.append(float(ms))

    def summary(self) -> dict[str, float]:
        if not self.samples_ms:
            return {"avg_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "call_count": 0.0}
        arr = np.asarray(self.samples_ms, dtype=np.float64)
        return {
            "avg_ms": float(np.mean(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
            "call_count": float(arr.size),
        }

    def clear(self) -> None:
        self.samples_ms.clear()


@dataclass
class StepTimingRecorder:
    """Episode-local timing recorder; one instance per env controller."""

    manifold: TimingBucket = field(default_factory=TimingBucket)
    cost_matrix: TimingBucket = field(default_factory=TimingBucket)
    los_check: TimingBucket = field(default_factory=TimingBucket)
    path_planning: TimingBucket = field(default_factory=TimingBucket)
    assignment: TimingBucket = field(default_factory=TimingBucket)
    waypoint: TimingBucket = field(default_factory=TimingBucket)
    cbf_filter: TimingBucket = field(default_factory=TimingBucket)
    action_post: TimingBucket = field(default_factory=TimingBucket)
    logging_viz: TimingBucket = field(default_factory=TimingBucket)
    decision_total: TimingBucket = field(default_factory=TimingBucket)

    def reset_episode(self) -> None:
        for bucket in (
            self.manifold,
            self.cost_matrix,
            self.los_check,
            self.path_planning,
            self.assignment,
            self.waypoint,
            self.cbf_filter,
            self.action_post,
            self.logging_viz,
            self.decision_total,
        ):
            bucket.clear()

    def episode_summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, bucket in (
            ("manifold", self.manifold),
            ("cost_matrix", self.cost_matrix),
            ("los_check", self.los_check),
            ("path_planning", self.path_planning),
            ("assignment", self.assignment),
            ("waypoint", self.waypoint),
            ("cbf_filter", self.cbf_filter),
            ("action_post", self.action_post),
            ("logging_viz", self.logging_viz),
            ("decision_total", self.decision_total),
        ):
            s = bucket.summary()
            for k, v in s.items():
                out[f"{name}_{k}"] = v
        dt = self.decision_total.summary()
        out["avg_decision_ms"] = dt["avg_ms"]
        out["p95_decision_ms"] = dt["p95_ms"]
        out["max_decision_ms"] = dt["max_ms"]
        return out
