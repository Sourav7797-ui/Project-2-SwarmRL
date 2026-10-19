"""Curriculum stages and promotion logic (Task 2, Day 7).

Architecture doc, section 2.3: task difficulty ramps up, static world ->
dynamic obstacles -> wind, controlled by a curriculum callback. This module
is the pure-logic half of that: WHAT the stages are and WHEN to promote.

  * It never touches the environment's physics. Day 8 wires the stage
    parameters (moving obstacles, wind) into SwarmEnv.
  * It never imports Ray. Task 3's RLlib callback will call
    CurriculumManager.record_episode() at the end of each episode.

Promotion rule: over the most recent `window_size` episodes of the CURRENT
stage, mean final coverage must reach the stage's `min_coverage_pct` AND the
pooled collision rate must stay at or below `max_collisions_per_1k`. The
window is cleared on promotion so evidence from an easier stage never counts
toward the next one. There is no demotion.

IMPORTANT: the default thresholds are PROVISIONAL. They cannot be calibrated
until a trained policy exists. Reference points from the scripted baselines in
scripts/analyze_reward_distribution.py (50 agents, 1000 steps):
    observation-driven 'reactive' policy: ~12% coverage, ~0.27 collisions / 1k agent-steps
    random policy:                        ~13% coverage, ~0.9  collisions / 1k agent-steps
The defaults below ask for roughly twice the scripted coverage. Recalibrate
them from the Phase 1 training runs in Task 3.
"""
from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass

from backend.env.config import SwarmEnvConfig


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CurriculumStage:
    name: str
    num_obstacles: int
    # Difficulty parameters wired into the physics on Day 8. Until then they are
    # carried but unused: only num_obstacles affects the environment today.
    obstacles_move: bool = False
    obstacle_speed: float = 0.0      # m/s
    wind_speed: float = 0.0          # m/s, peak wind drift
    # Promotion criteria (ignored on the final stage)
    min_coverage_pct: float = 0.0
    max_collisions_per_1k: float = math.inf   # collision events per 1000 agent-steps

    def __post_init__(self):
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if self.num_obstacles < 0:
            raise ValueError(f"{self.name}: num_obstacles must be >= 0, got {self.num_obstacles}")
        if self.obstacle_speed < 0 or self.wind_speed < 0:
            raise ValueError(f"{self.name}: speeds must be >= 0")
        if self.obstacle_speed > 0 and not self.obstacles_move:
            raise ValueError(f"{self.name}: obstacle_speed > 0 but obstacles_move is False")
        if not 0.0 <= self.min_coverage_pct <= 100.0:
            raise ValueError(f"{self.name}: min_coverage_pct must be in [0, 100]")
        if self.max_collisions_per_1k < 0:
            raise ValueError(f"{self.name}: max_collisions_per_1k must be >= 0")

    def apply_to(self, config: SwarmEnvConfig) -> SwarmEnvConfig:
        """A copy of `config` with this stage's environment settings applied.
        The input config is frozen and is not modified."""
        return dataclasses.replace(config, num_obstacles=self.num_obstacles)


STAGE_STATIC = CurriculumStage(
    name="static_world",
    num_obstacles=0,
    min_coverage_pct=25.0,
    max_collisions_per_1k=0.2,
)
STAGE_DYNAMIC = CurriculumStage(
    name="dynamic_obstacles",
    num_obstacles=8,
    obstacles_move=True,
    obstacle_speed=1.5,
    min_coverage_pct=20.0,
    max_collisions_per_1k=0.5,
)
STAGE_WIND = CurriculumStage(
    name="wind",
    num_obstacles=8,
    obstacles_move=True,
    obstacle_speed=1.5,
    wind_speed=2.0,
)

DEFAULT_STAGES = (STAGE_STATIC, STAGE_DYNAMIC, STAGE_WIND)


# --------------------------------------------------------------------------
# Episode statistics
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EpisodeStats:
    coverage_pct: float      # final explored fraction of the voxel grid, in percent
    collision_events: int    # drone-drone + drone-obstacle collision EVENTS (not ticks)
    agent_steps: int         # agents x steps in the episode

    def __post_init__(self):
        if self.agent_steps <= 0:
            raise ValueError("agent_steps must be positive")
        if self.collision_events < 0:
            raise ValueError("collision_events must be >= 0")

    @property
    def collisions_per_1k(self) -> float:
        return self.collision_events / self.agent_steps * 1000.0


class EpisodeStatsAccumulator:
    """Feed it the `infos` dict from every env.step(); call finalize() at the end.

    Counts collision EVENTS (the `collision_event` / `obstacle_collision_event`
    flags), not ticks of overlap, matching how the reward is charged."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.collision_events = 0
        self.agent_steps = 0

    def add_step(self, infos: dict[str, dict]) -> None:
        for info in infos.values():
            self.collision_events += int(info.get("collision_event", False))
            self.collision_events += int(info.get("obstacle_collision_event", False))
        self.agent_steps += len(infos)

    def finalize(self, coverage_pct: float) -> EpisodeStats:
        return EpisodeStats(
            coverage_pct=float(coverage_pct),
            collision_events=self.collision_events,
            agent_steps=self.agent_steps,
        )


# --------------------------------------------------------------------------
# Manager
# --------------------------------------------------------------------------
class CurriculumManager:
    def __init__(self, stages=DEFAULT_STAGES, window_size: int = 20):
        stages = tuple(stages)
        if not stages:
            raise ValueError("at least one stage is required")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique, got {names}")

        self._stages = stages
        self._window_size = window_size
        self._window: deque[EpisodeStats] = deque(maxlen=window_size)
        self._stage_index = 0
        self._episodes_total = 0
        self._episodes_in_stage = 0
        self._promotions: list[dict] = []

    # ---- read-only state ----
    @property
    def stage(self) -> CurriculumStage:
        return self._stages[self._stage_index]

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def is_final(self) -> bool:
        return self._stage_index == len(self._stages) - 1

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def episodes_total(self) -> int:
        return self._episodes_total

    @property
    def episodes_in_stage(self) -> int:
        return self._episodes_in_stage

    @property
    def promotions(self) -> list[dict]:
        return [dict(p) for p in self._promotions]

    def window_means(self) -> dict | None:
        """Summary of the current window, for logging. None if it is empty."""
        if not self._window:
            return None
        coverage = sum(e.coverage_pct for e in self._window) / len(self._window)
        events = sum(e.collision_events for e in self._window)
        steps = sum(e.agent_steps for e in self._window)
        return {
            "coverage_pct": coverage,
            "collisions_per_1k": events / steps * 1000.0,   # pooled, not a mean of ratios
            "episodes": len(self._window),
        }

    # ---- promotion ----
    def should_promote(self) -> bool:
        if self.is_final or len(self._window) < self._window_size:
            return False
        m = self.window_means()
        return (m["coverage_pct"] >= self.stage.min_coverage_pct
                and m["collisions_per_1k"] <= self.stage.max_collisions_per_1k)

    def record_episode(self, stats: EpisodeStats) -> bool:
        """Record one finished episode. Returns True if this call promoted the
        curriculum to the next stage."""
        self._window.append(stats)
        self._episodes_total += 1
        self._episodes_in_stage += 1

        if not self.should_promote():
            return False

        self._promotions.append({
            "at_episode": self._episodes_total,
            "from_stage": self.stage.name,
            "to_stage": self._stages[self._stage_index + 1].name,
            "window": self.window_means(),
        })
        self._stage_index += 1
        self._window.clear()           # old-stage evidence must not count toward the new stage
        self._episodes_in_stage = 0
        return True

    # ---- persistence (RLlib restarts, checkpoints) ----
    def state_dict(self) -> dict:
        """JSON-serialisable snapshot."""
        return {
            "stage_names": [s.name for s in self._stages],
            "window_size": self._window_size,
            "stage_index": self._stage_index,
            "episodes_total": self._episodes_total,
            "episodes_in_stage": self._episodes_in_stage,
            "window": [[e.coverage_pct, e.collision_events, e.agent_steps] for e in self._window],
            "promotions": self.promotions,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["stage_names"] != [s.name for s in self._stages]:
            raise ValueError("state was saved from a different curriculum "
                             f"({state['stage_names']} vs {[s.name for s in self._stages]})")
        if state["window_size"] != self._window_size:
            raise ValueError("state was saved with a different window_size")
        self._stage_index = int(state["stage_index"])
        self._episodes_total = int(state["episodes_total"])
        self._episodes_in_stage = int(state["episodes_in_stage"])
        self._window = deque(
            (EpisodeStats(float(c), int(e), int(a)) for c, e, a in state["window"]),
            maxlen=self._window_size,
        )
        self._promotions = [dict(p) for p in state["promotions"]]