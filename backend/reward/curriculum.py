"""Curriculum stages, episode statistics, and promotion management.

The public dataclass fields and serialized state keys in this module are
kept stable for compatibility with the Day 7-Day 9 tests and environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Episode statistics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeStats:
    """Statistics for one completed episode.

    Keep exactly these three dataclass fields for the Day 8 contract.
    """

    coverage_pct: float
    collision_events: int
    agent_steps: int

    def __post_init__(self) -> None:
        if self.agent_steps <= 0:
            raise ValueError("agent_steps must be positive")
        if self.collision_events < 0:
            raise ValueError("collision_events must be >= 0")
        if not 0.0 <= self.coverage_pct <= 100.0:
            raise ValueError("coverage_pct must be between 0 and 100")

    @property
    def collisions_per_1k(self) -> float:
        return self.collision_events / self.agent_steps * 1000.0


class EpisodeStatsAccumulator:
    """Accumulate episode metrics from per-agent info dictionaries.

    Total collisions retain the existing contract. Drone-to-drone and
    drone-to-obstacle counts are also tracked separately for analysis.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.collision_events = 0
        self.drone_collision_events = 0
        self.obstacle_collision_events = 0
        self.agent_steps = 0

    def add_step(self, infos: dict[str, dict[str, Any]]) -> None:
        """Add one environment step of per-agent info dictionaries."""
        for info in infos.values():
            drone_event = int(bool(info.get("collision_event", False)))
            obstacle_event = int(
                bool(info.get("obstacle_collision_event", False))
            )

            self.drone_collision_events += drone_event
            self.obstacle_collision_events += obstacle_event
            self.collision_events += drone_event + obstacle_event

        self.agent_steps += len(infos)

    def finalize(self, coverage_pct: float) -> EpisodeStats:
        """Return the stable three-field episode statistics object."""
        return EpisodeStats(
            coverage_pct=float(coverage_pct),
            collision_events=self.collision_events,
            agent_steps=self.agent_steps,
        )

    @property
    def drone_collisions_per_1k(self) -> float:
        if self.agent_steps == 0:
            return 0.0
        return self.drone_collision_events / self.agent_steps * 1000.0

    @property
    def obstacle_collisions_per_1k(self) -> float:
        if self.agent_steps == 0:
            return 0.0
        return self.obstacle_collision_events / self.agent_steps * 1000.0


# ---------------------------------------------------------------------------
# Curriculum stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage and the environment settings it controls."""

    name: str
    num_obstacles: int
    obstacles_move: bool = False
    obstacle_speed: float = 0.0
    wind_speed: float = 0.0
    min_coverage_pct: float = 0.0
    max_collisions_per_1k: float = float("inf")

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if self.num_obstacles < 0:
            raise ValueError("num_obstacles must be >= 0")

        if self.obstacle_speed < 0:
            raise ValueError("obstacle_speed must be >= 0")

        if self.wind_speed < 0:
            raise ValueError("wind_speed must be >= 0")

        if not 0.0 <= self.min_coverage_pct <= 100.0:
            raise ValueError("min_coverage_pct must be between 0 and 100")

        if self.max_collisions_per_1k < 0:
            raise ValueError("max_collisions_per_1k must be >= 0")

        if not self.obstacles_move and self.obstacle_speed > 0:
            raise ValueError(
                "obstacle_speed must be zero when obstacles_move is False"
            )

        if self.num_obstacles == 0 and self.obstacles_move:
            raise ValueError(
                "obstacles_move cannot be True when num_obstacles is zero"
            )

    @property
    def moving_obstacles(self) -> bool:
        """Compatibility alias for code using the newer field spelling."""
        return self.obstacles_move

    def apply_to(self, config: Any) -> Any:
        """Return a config copy with only this stage's dynamics changed."""
        return replace(
            config,
            num_obstacles=self.num_obstacles,
            obstacles_move=self.obstacles_move,
            obstacle_speed=self.obstacle_speed,
            wind_speed=self.wind_speed,
        )


STAGE_STATIC = CurriculumStage(
    name="static_world",
    num_obstacles=0,
    obstacles_move=False,
    obstacle_speed=0.0,
    wind_speed=0.0,
    min_coverage_pct=25.0,
    max_collisions_per_1k=0.2,
)

STAGE_DYNAMIC = CurriculumStage(
    name="dynamic_obstacles",
    num_obstacles=8,
    obstacles_move=True,
    obstacle_speed=1.5,
    wind_speed=0.0,
    min_coverage_pct=20.0,
    max_collisions_per_1k=0.5,
)

STAGE_WIND = CurriculumStage(
    name="wind",
    num_obstacles=8,
    obstacles_move=True,
    obstacle_speed=1.5,
    wind_speed=2.0,
    min_coverage_pct=20.0,
    max_collisions_per_1k=0.5,
)

DEFAULT_STAGES = (STAGE_STATIC, STAGE_DYNAMIC, STAGE_WIND)


# ---------------------------------------------------------------------------
# Curriculum manager
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Promote through stages when a full rolling window meets its thresholds."""

    def __init__(
        self,
        stages: Iterable[CurriculumStage] = DEFAULT_STAGES,
        window_size: int = 10,
    ) -> None:
        stages = tuple(stages)

        if not stages:
            raise ValueError("stages must not be empty")

        if window_size <= 0:
            raise ValueError("window_size must be positive")

        stage_names = [stage.name for stage in stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("stage names must be unique")

        self.stages = stages
        self._window_size = int(window_size)
        self._stage_index = 0
        self._episodes_total = 0
        self._episodes_in_stage = 0
        self._window: deque[EpisodeStats] = deque(maxlen=self._window_size)
        self._promotions: list[dict[str, Any]] = []

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self._stage_index]

    @property
    def stage_index(self) -> int:
        return self._stage_index

    @property
    def is_final(self) -> bool:
        return self._stage_index >= len(self.stages) - 1

    @property
    def window_size(self) -> int:
        """Return the configured rolling-window size."""
        return self._window_size

    @property
    def episodes_total(self) -> int:
        return self._episodes_total

    @property
    def episodes_in_stage(self) -> int:
        return self._episodes_in_stage

    @property
    def promotions(self) -> list[dict[str, Any]]:
        """Return a copy so callers cannot mutate internal history."""
        return [dict(promotion) for promotion in self._promotions]

    def window_means(self) -> dict[str, float | int] | None:
        """Return rolling-window metrics, or None until the window is full.

        Coverage is averaged across episodes. Collision rate is pooled over
        all agent steps, rather than averaging the individual episode rates.
        """
        if len(self._window) < self._window_size:
            return None

        total_steps = sum(entry.agent_steps for entry in self._window)
        if total_steps <= 0:
            return None

        episode_count = len(self._window)

        mean_coverage = sum(
            entry.coverage_pct for entry in self._window
        ) / episode_count

        total_collisions = sum(
            entry.collision_events for entry in self._window
        )

        return {
            "episodes": episode_count,
            "coverage_pct": mean_coverage,
            "collision_events": total_collisions,
            "agent_steps": total_steps,
            "collisions_per_1k": total_collisions / total_steps * 1000.0,
        }

    def should_promote(self) -> bool:
        """Check the current full window against the current stage limits."""
        if self.is_final:
            return False

        means = self.window_means()
        if means is None:
            return False

        return (
            means["coverage_pct"] >= self.stage.min_coverage_pct
            and means["collisions_per_1k"]
            <= self.stage.max_collisions_per_1k
        )

    def record_episode(self, stats: EpisodeStats) -> bool:
        """Record an episode and promote if the rolling window qualifies.

        Returns True only when this call promotes to the next stage.
        """
        if not isinstance(stats, EpisodeStats):
            raise TypeError("stats must be an EpisodeStats instance")

        self._episodes_total += 1
        self._episodes_in_stage += 1
        self._window.append(stats)

        if not self.should_promote():
            return False

        old_stage = self.stage
        old_index = self._stage_index
        means = self.window_means()

        self._stage_index += 1
        new_stage = self.stage

        # Keep the established promotion-history fields and provide the
        # expected at_episode field for existing tests and callers.
        self._promotions.append(
            {
                "from_stage": old_stage.name,
                "to_stage": new_stage.name,
                "from_stage_index": old_index,
                "to_stage_index": self._stage_index,
                "at_episode": self._episodes_total,
                "episode": self._episodes_total,
                "episodes_total": self._episodes_total,
                "episodes_in_stage": self._episodes_in_stage,
                "window": dict(means) if means is not None else None,
                "window_means": dict(means) if means is not None else None,
            }
        )

        # Evidence from the previous stage must not carry over.
        self._window.clear()
        self._episodes_in_stage = 0
        return True

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state using the stable Day 8 keys."""
        return {
            "stage_names": [stage.name for stage in self.stages],
            "window_size": self._window_size,
            "stage_index": self._stage_index,
            "episodes_total": self._episodes_total,
            "episodes_in_stage": self._episodes_in_stage,
            "window": [
                [
                    entry.coverage_pct,
                    entry.collision_events,
                    entry.agent_steps,
                ]
                for entry in self._window
            ],
            "promotions": [dict(promotion) for promotion in self._promotions],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state, rejecting incompatible curriculum/window settings."""
        expected_names = [stage.name for stage in self.stages]

        if state.get("stage_names") != expected_names:
            raise ValueError("state belongs to a different curriculum")

        if state.get("window_size") != self._window_size:
            raise ValueError("state uses a different window_size")

        stage_index = int(state["stage_index"])
        if not 0 <= stage_index < len(self.stages):
            raise ValueError("invalid stage_index in saved state")

        episodes_total = int(state["episodes_total"])
        episodes_in_stage = int(state["episodes_in_stage"])
        if episodes_total < 0 or episodes_in_stage < 0:
            raise ValueError("episode counters must be non-negative")

        restored_window = []
        for item in state["window"]:
            if len(item) != 3:
                raise ValueError("each window entry must contain three values")

            coverage, collisions, agent_steps = item
            restored_window.append(
                EpisodeStats(
                    float(coverage),
                    int(collisions),
                    int(agent_steps),
                )
            )

        if len(restored_window) > self._window_size:
            raise ValueError("saved window is larger than window_size")

        promotions = state["promotions"]
        if not isinstance(promotions, list):
            raise ValueError("promotions must be a list")

        self._stage_index = stage_index
        self._episodes_total = episodes_total
        self._episodes_in_stage = episodes_in_stage
        self._window = deque(restored_window, maxlen=self._window_size)
        self._promotions = [dict(promotion) for promotion in promotions]
