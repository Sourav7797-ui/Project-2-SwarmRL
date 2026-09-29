"""Reward shaping, isolated from environment physics so it can be
tuned independently:
  +1    per agent that reveals a previously unvisited voxel
  -100  for mid-air collisions / near-miss separation violations
  small continuous out-of-bounds / altitude penalty
  shared team-coverage bonus for credit assignment across the swarm
"""
from __future__ import annotations

import numpy as np

from backend.env.config import SwarmEnvConfig

EXPLORATION_REWARD = 1.0
COLLISION_PENALTY = -100.0
BOUNDARY_MARGIN = 5.0
BOUNDARY_PENALTY_SCALE = -0.1
TEAM_COVERAGE_BONUS_SCALE = 0.05


class RewardShaper:
    def __init__(self, config: SwarmEnvConfig):
        self.config = config

    def compute(self, newly_covered_by_agent, collisions_by_agent, positions,
                world_size_xy, world_height) -> dict[str, float]:
        n = len(positions)
        rewards = np.zeros(n, dtype=np.float32)
        rewards += newly_covered_by_agent.astype(np.float32) * EXPLORATION_REWARD
        rewards += collisions_by_agent.astype(np.float32) * COLLISION_PENALTY
        rewards += self._boundary_penalty(positions, world_size_xy, world_height)
        team_bonus = float(newly_covered_by_agent.sum()) / max(n, 1) * TEAM_COVERAGE_BONUS_SCALE
        rewards += team_bonus
        return {f"drone_{i}": float(rewards[i]) for i in range(n)}

    @staticmethod
    def _boundary_penalty(positions, world_size_xy, world_height):
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        def edge_violation(v, lo, hi):
            under = np.clip(lo + BOUNDARY_MARGIN - v, 0, None)
            over = np.clip(v - (hi - BOUNDARY_MARGIN), 0, None)
            return under + over

        violation = (edge_violation(x, 0, world_size_xy)
                     + edge_violation(y, 0, world_size_xy)
                     + edge_violation(z, 0, world_height))
        return violation * BOUNDARY_PENALTY_SCALE