
"""Reward shaping, isolated from environment physics.

Day 15 adds an optional logarithmic voxel-discovery reward.
The default binary mode preserves the original reward behaviour.

Day 16 adds an optional soft drone-to-drone separation reward.
The feature is disabled by default.
"""
from __future__ import annotations

import numpy as np

from backend.env.config import SwarmEnvConfig


COMPONENT_EXPLORATION = "exploration"
COMPONENT_DRONE_COLLISION = "drone_collision"
COMPONENT_OBSTACLE_COLLISION = "obstacle_collision"
COMPONENT_BOUNDARY = "boundary"
COMPONENT_TEAM_BONUS = "team_bonus"
COMPONENT_SEPARATION = "separation"


# Original component contract: preserve its names and order.
ALL_COMPONENTS = (
    COMPONENT_EXPLORATION,
    COMPONENT_DRONE_COLLISION,
    COMPONENT_OBSTACLE_COLLISION,
    COMPONENT_BOUNDARY,
    COMPONENT_TEAM_BONUS,
)

# Extended component list for Day 16 monitoring.
ALL_COMPONENTS_WITH_SEPARATION = (
    *ALL_COMPONENTS,
    COMPONENT_SEPARATION,
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

        # --------------------------------------------------
        # Day 15: configurable exploration reward
        # --------------------------------------------------
        if c.exploration_reward_mode == "binary":
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
                new_voxel_counts_by_agent,
                dtype=np.float32,
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

        # --------------------------------------------------
        # Existing drone collision penalty
        # --------------------------------------------------
        drone_collision = (
            collisions_by_agent.astype(np.float32)
            * c.collision_penalty
        )

        # --------------------------------------------------
        # Existing obstacle collision penalty
        # --------------------------------------------------
        if obstacle_collisions_by_agent is not None:
            obstacle_collision = (
                obstacle_collisions_by_agent.astype(np.float32)
                * c.obstacle_collision_penalty
            )
        else:
            obstacle_collision = np.zeros(
                n,
                dtype=np.float32,
            )

        # --------------------------------------------------
        # Existing boundary penalty
        # --------------------------------------------------
        boundary = self._boundary_penalty(
            positions,
            world_size_xy,
            world_height,
        )

        # --------------------------------------------------
        # Day 16: optional soft drone separation penalty
        # --------------------------------------------------
        separation = np.zeros(
            n,
            dtype=np.float32,
        )

        if c.separation_reward_enabled and n > 1:
            preferred = float(c.preferred_separation)
            hard_limit = float(c.min_separation)

            if (
                not np.isfinite(preferred)
                or preferred <= hard_limit
            ):
                raise ValueError(
                    "preferred_separation must be greater than "
                    "min_separation"
                )

            if not np.isfinite(c.separation_penalty_scale):
                raise ValueError(
                    "separation_penalty_scale must be finite"
                )

            if c.separation_penalty_scale > 0:
                raise ValueError(
                    "separation_penalty_scale must be zero or negative"
                )

            # Calculate distances between every pair of drones.
            differences = (
                positions[:, np.newaxis, :]
                - positions[np.newaxis, :, :]
            )

            distances = np.linalg.norm(
                differences,
                axis=2,
            )

            # Ignore each drone's distance from itself.
            np.fill_diagonal(distances, np.inf)

            # Find each drone's nearest neighbour.
            nearest_distances = np.min(
                distances,
                axis=1,
            )

            # Penalize only when distance is below preferred.
            separation_deficit = np.clip(
                preferred - nearest_distances,
                0.0,
                None,
            )

            normalized_deficit = (
                separation_deficit
                / (preferred - hard_limit)
            )

            separation = (
                c.separation_penalty_scale
                * np.square(normalized_deficit)
            ).astype(np.float32)

        # --------------------------------------------------
        # Existing team coverage bonus
        # --------------------------------------------------
        team_bonus_value = (
            float(newly_covered_by_agent.sum())
            / max(n, 1)
            * c.team_coverage_bonus_scale
        )

        team_bonus = np.full(
            n,
            team_bonus_value,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Calculate total reward
        # --------------------------------------------------
        total = (
            exploration
            + drone_collision
            + obstacle_collision
            + boundary
            + team_bonus
            + separation
        )

        # --------------------------------------------------
        # Preserve original breakdown contract by default
        # --------------------------------------------------
        breakdown = {
            COMPONENT_EXPLORATION: exploration,
            COMPONENT_DRONE_COLLISION: drone_collision,
            COMPONENT_OBSTACLE_COLLISION: obstacle_collision,
            COMPONENT_BOUNDARY: boundary,
            COMPONENT_TEAM_BONUS: team_bonus,
        }

        # Add the sixth component only when Day 16 is enabled.
        if c.separation_reward_enabled:
            breakdown[COMPONENT_SEPARATION] = separation

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
                lo + c.boundary_margin - v,
                0,
                None,
            )

            over = np.clip(
                v - (hi - c.boundary_margin),
                0,
                None,
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
