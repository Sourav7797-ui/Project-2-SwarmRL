"""Episode-level reward component logger (Day 5).

RLlib (and anyone staring at a reward curve) only sees the scalar total
reward. That's enough to know training is or isn't working, but not WHY
- is the swarm getting better at exploring, or just colliding less? This
logger accumulates each reward component separately across an episode so
that question is answerable.
"""
from __future__ import annotations

import numpy as np


class RewardComponentLogger:
    """Attach one instance to env.reward_logger before running an episode;
    SwarmEnv.step() will call log_step() automatically each tick if the
    attribute is set. Call reset() between episodes to avoid mixing data."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._per_step: dict[str, list[float]] = {}
        self.num_steps = 0

    def log_step(self, breakdown: dict[str, np.ndarray]) -> None:
        """breakdown: component_name -> (N,) array of per-agent values
        for this tick, as returned by RewardShaper.compute_with_breakdown."""
        self.num_steps += 1
        for component, values in breakdown.items():
            summed = float(np.sum(values))
            self._totals[component] = self._totals.get(component, 0.0) + summed
            self._per_step.setdefault(component, []).append(summed)

    def episode_totals(self) -> dict[str, float]:
        """Sum of each component across the whole episode, across all agents."""
        return dict(self._totals)

    def episode_means_per_step(self) -> dict[str, float]:
        """Average per-tick contribution of each component."""
        return {c: t / max(self.num_steps, 1) for c, t in self._totals.items()}

    def component_trace(self, component: str) -> list[float]:
        """Per-tick time series for one component."""
        return list(self._per_step.get(component, []))

    def reset(self) -> None:
        self._totals.clear()
        self._per_step.clear()
        self.num_steps = 0

    def summary_string(self) -> str:
        """Human-readable breakdown, e.g. for printing after a debug run."""
        lines = [f"Reward breakdown over {self.num_steps} steps:"]
        total_all = sum(self._totals.values())
        for component, total in sorted(self._totals.items(), key=lambda kv: -abs(kv[1])):
            pct = (total / total_all * 100) if total_all != 0 else 0.0
            lines.append(f"  {component:20s}  total={total:12.2f}  ({pct:6.1f}% of net)")
        lines.append(f"  {'NET':20s}  total={total_all:12.2f}")
        return "\n".join(lines)