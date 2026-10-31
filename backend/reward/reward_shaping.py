
"""Reward shaping, isolated from environment physics.

Day 15 adds an optional logarithmic voxel-discovery reward.
The default binary mode preserves the original reward behaviour.
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
        new_voxel_counts_by_agent: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return total reward per agent."""
        rewards, _breakdown = self.compute_with_breakdown(
            newly_covered_by_agent=newly_covered_by_agent,
            collisions_by_agent=collisions_by_agent,
            positions=positions,
            world_size_xy=world_size_xy,
            world_height=world_height,
            obstacle_collisions_by_agent=obstacle_collisions_by_agent,
            new_voxel_counts_by_agent=new_voxel_counts_by_agent,
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
        new_voxel_counts_by_agent: np.ndarray | None = None,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Return total reward and per-component reward breakdown."""
        c = self.config
        n = len(positions)

        # Day 15: configurable exploration reward.
        if c.exploration_reward_mode == "binary":
            # Original behaviour: one reward per agent discovering anything.
            exploration = (
                newly_covered_by_agent.astype(np.float32)
                * c.exploration_reward
            )

        elif c.exploration_reward_mode == "voxel_log":
            if new_voxel_counts_by_agent is None:
                raise ValueError(
                    "new_voxel_counts_by_agent is required when "
                    "exploration_reward_mode='voxel_log'"
                )

            counts = np.asarray(
                new_voxel_counts_by_agent, dtype=np.float32
            )

            if counts.shape != (n,):
                raise ValueError(
                    f"Expected new_voxel_counts_by_agent shape {(n,)}, "
                    f"received {counts.shape}"
                )

            if not np.all(np.isfinite(counts)):
                raise ValueError(
                    "Voxel discovery counts must be finite"
                )

            if np.any(counts < 0):
                raise ValueError(
                    "Voxel discovery counts cannot be negative"
                )

            reference = float(c.exploration_voxel_reference)

            if not np.isfinite(reference) or reference <= 0:
                raise ValueError(
                    "exploration_voxel_reference must be greater than zero"
                )

            exploration = (
                np.log1p(counts)
                / np.log1p(reference)
                * c.exploration_reward
            ).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown exploration_reward_mode: "
                f"{c.exploration_reward_mode!r}"
            )

        drone_collision = (
            collisions_by_agent.astype(np.float32)
            * c.collision_penalty
        )

        if obstacle_collisions_by_agent is not None:
            obstacle_collision = (
                obstacle_collisions_by_agent.astype(np.float32)
                * c.obstacle_collision_penalty
            )
        else:
            obstacle_collision = np.zeros(n, dtype=np.float32)

        boundary = self._boundary_penalty(
            positions, world_size_xy, world_height
        )

        # Preserve the original team bonus semantics.
        team_bonus_value = (
            float(newly_covered_by_agent.sum())
            / max(n, 1)
            * c.team_coverage_bonus_scale
        )
        team_bonus = np.full(n, team_bonus_value, dtype=np.float32)

        total = (
            exploration
            + drone_collision
            + obstacle_collision
            + boundary
            + team_bonus
        )

        breakdown = {
            COMPONENT_EXPLORATION: exploration,
            COMPONENT_DRONE_COLLISION: drone_collision,
            COMPONENT_OBSTACLE_COLLISION: obstacle_collision,
            COMPONENT_BOUNDARY: boundary,
            COMPONENT_TEAM_BONUS: team_bonus,
        }

        rewards = {
            f"drone_{i}": float(total[i])
            for i in range(n)
        }

        return rewards, breakdown

    def _boundary_penalty(
        self,
        positions: np.ndarray,
        world_size_xy: float,
        world_height: float,
    ) -> np.ndarray:
        c = self.config
        x, y, z = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )

        def edge_violation(v, lo, hi):
            under = np.clip(
                lo + c.boundary_margin - v, 0, None
            )
            over = np.clip(
                v - (hi - c.boundary_margin), 0, None
            )
            return under + over

        violation = (
            edge_violation(x, 0, world_size_xy)
            + edge_violation(y, 0, world_size_xy)
            + edge_violation(z, 0, world_height)
        )

        return (
            violation * c.boundary_penalty_scale
        ).astype(np.float32)