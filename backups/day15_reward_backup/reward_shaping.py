"""Reward shaping, isolated from environment physics so it can be
tuned independently (architecture doc, section 2.2):

  +1    per agent that reveals a previously unvisited voxel
  -100  for mid-air collisions / near-miss separation violations
  -75   for drone-obstacle collisions
  small continuous out-of-bounds / altitude penalty
  shared team-coverage bonus for credit assignment across the swarm

Day 5 change: every weight above now lives in SwarmEnvConfig instead of
being hardcoded as a module constant here. This lets curriculum stages
(Day 7-8) and tuning sweeps (Day 6) override reward weights per-config
without touching this file.
"""
from __future__ import annotations

import numpy as np

from backend.env.config import SwarmEnvConfig

COMPONENT_EXPLORATION = "exploration"
COMPONENT_DRONE_COLLISION = "drone_collision"
COMPONENT_OBSTACLE_COLLISION = "obstacle_collision"
COMPONENT_BOUNDARY = "boundary"
COMPONENT_TEAM_BONUS = "team_bonus"

ALL_COMPONENTS = (
    COMPONENT_EXPLORATION,
    COMPONENT_DRONE_COLLISION,
    COMPONENT_OBSTACLE_COLLISION,
    COMPONENT_BOUNDARY,
    COMPONENT_TEAM_BONUS,
)


class RewardShaper:
    def __init__(self, config: SwarmEnvConfig):
        self.config = config

    def compute(
        self,
        newly_covered_by_agent: np.ndarray,
        collisions_by_agent: np.ndarray,
        positions: np.ndarray,
        world_size_xy: float,
        world_height: float,
        obstacle_collisions_by_agent: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Backward-compatible entry point: just the total reward per
        agent, no breakdown."""
        rewards, _breakdown = self.compute_with_breakdown(
            newly_covered_by_agent=newly_covered_by_agent,
            collisions_by_agent=collisions_by_agent,
            positions=positions,
            world_size_xy=world_size_xy,
            world_height=world_height,
            obstacle_collisions_by_agent=obstacle_collisions_by_agent,
        )
        return rewards

    def compute_with_breakdown(
        self,
        newly_covered_by_agent: np.ndarray,
        collisions_by_agent: np.ndarray,
        positions: np.ndarray,
        world_size_xy: float,
        world_height: float,
        obstacle_collisions_by_agent: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Same computation as compute(), but also returns a per-component
        breakdown (each a (N,) array) so callers can see where the total
        reward came from."""
        c = self.config
        n = len(positions)

        exploration = newly_covered_by_agent.astype(np.float32) * c.exploration_reward
        drone_collision = collisions_by_agent.astype(np.float32) * c.collision_penalty

        if obstacle_collisions_by_agent is not None:
            obstacle_collision = obstacle_collisions_by_agent.astype(np.float32) * c.obstacle_collision_penalty
        else:
            obstacle_collision = np.zeros(n, dtype=np.float32)

        boundary = self._boundary_penalty(positions, world_size_xy, world_height)

        team_bonus_value = float(newly_covered_by_agent.sum()) / max(n, 1) * c.team_coverage_bonus_scale
        team_bonus = np.full(n, team_bonus_value, dtype=np.float32)

        total = exploration + drone_collision + obstacle_collision + boundary + team_bonus

        breakdown = {
            COMPONENT_EXPLORATION: exploration,
            COMPONENT_DRONE_COLLISION: drone_collision,
            COMPONENT_OBSTACLE_COLLISION: obstacle_collision,
            COMPONENT_BOUNDARY: boundary,
            COMPONENT_TEAM_BONUS: team_bonus,
        }
        rewards = {f"drone_{i}": float(total[i]) for i in range(n)}
        return rewards, breakdown

    def _boundary_penalty(self, positions: np.ndarray, world_size_xy: float, world_height: float) -> np.ndarray:
        c = self.config
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        def edge_violation(v, lo, hi):
            under = np.clip(lo + c.boundary_margin - v, 0, None)
            over = np.clip(v - (hi - c.boundary_margin), 0, None)
            return under + over

        violation = (
            edge_violation(x, 0, world_size_xy)
            + edge_violation(y, 0, world_size_xy)
            + edge_violation(z, 0, world_height)
        )
        return (violation * c.boundary_penalty_scale).astype(np.float32)