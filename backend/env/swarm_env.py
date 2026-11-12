"""SwarmEnv: a continuous 3D disaster-zone search environment for
50 autonomous drones, built on PettingZoo's ParallelEnv API."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from backend.env.config import SwarmEnvConfig
from backend.reward.reward_shaping import RewardShaper

if TYPE_CHECKING:
    from backend.reward.curriculum import CurriculumStage


# Fixed-capacity layout for the centralized MAPPO critic.
# Curriculum stages may use fewer obstacles, but the critic input
# dimension must remain constant across all stages.
GLOBAL_STATE_MAX_OBSTACLES = 8
GLOBAL_STATE_COVERAGE_FEATURES = 500


class SwarmEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "none"],
        "name": "swarmrl_v0",
    }

    def __init__(
        self,
        config: SwarmEnvConfig | None = None,
        render_mode: str = "none",
    ):
        self.config = config or SwarmEnvConfig()
        self.render_mode = render_mode

        self.possible_agents = [
            f"drone_{i}" for i in range(self.config.num_agents)
        ]
        self.agents = list(self.possible_agents)

        # Observation layout (Day 7), per agent:
        #   [0 : 3k]  k nearest neighbours as (distance, bearing_xy, bearing_z),
        #             bearings in the drone's BODY frame
        #   [3k]      distance to nearest obstacle
        #   [3k+1]    coverage ahead
        #   [3k+2]    battery remaining
        #   [3k+3]    distance to nearest xy wall
        #   [3k+4]    body-frame bearing to that wall
        #   [3k+5]    normalized altitude
        #   [3k+6]    own pitch
        #   [3k+7]    body-frame bearing to nearest obstacle
        self._obs_dim = self.config.k_nearest * 3 + 8

        self.reward_shaper = RewardShaper(self.config)
        self.reward_logger = None

        self.positions: np.ndarray | None = None
        self.headings: np.ndarray | None = None
        self.speed: np.ndarray | None = None
        self.battery: np.ndarray | None = None
        self.obstacles: np.ndarray | None = None
        self.obstacle_velocities: np.ndarray | None = None
        self.coverage_grid: np.ndarray | None = None

        self._step_count = 0

        # Day 11: cached voxel coordinates for sensor-based coverage.
        self._coverage_cache_key = None
        self._coverage_grid_indices = None
        self._coverage_voxel_centers = None

        # Collision event cooldown tracking.
        self._ticks_since_drone_contact: np.ndarray | None = None
        self._ticks_since_obstacle_contact: np.ndarray | None = None

        # Day 8: dynamics.
        self.wind_angle: float = 0.0
        self._rng: np.random.Generator | None = None
        self._pending_stage = None
        self.stage_name: str | None = None

        self._fig = None
        self._ax = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        c = self.config

        return Box(
            low=np.array(
                [0.0, -c.max_pitch_rate, -c.max_yaw_rate],
                dtype=np.float32,
            ),
            high=np.array(
                [c.max_velocity, c.max_pitch_rate, c.max_yaw_rate],
                dtype=np.float32,
            ),
        )

    def set_stage(self, stage: "CurriculumStage") -> None:
        """Queue a curriculum stage for the next reset."""
        self._pending_stage = stage

    def reset(self, seed=None, options=None):
        if self._pending_stage is not None:
            self.config = self._pending_stage.apply_to(self.config)
            self.reward_shaper = RewardShaper(self.config)
            self.stage_name = self._pending_stage.name
            self._pending_stage = None

        rng = np.random.default_rng(seed)
        c = self.config

        self.agents = list(self.possible_agents)
        self._step_count = 0

        n = c.num_agents

        self.positions = rng.uniform(
            low=[0.0, 0.0, 1.0],
            high=[
                c.world_size_xy,
                c.world_size_xy,
                c.world_height,
            ],
            size=(n, 3),
        ).astype(np.float32)

        self.headings = np.zeros(
            (n, 2),
            dtype=np.float32,
        )

        self.speed = np.zeros(
            n,
            dtype=np.float32,
        )

        self.battery = np.full(
            n,
            c.battery_capacity,
            dtype=np.float32,
        )

        self.obstacles = rng.uniform(
            low=[0.0, 0.0, 0.0],
            high=[
                c.world_size_xy,
                c.world_size_xy,
                c.world_height,
            ],
            size=(c.num_obstacles, 3),
        ).astype(np.float32)

        if c.randomize_initial_heading:
            self.headings[:, 1] = rng.uniform(
                0.0,
                2.0 * np.pi,
                size=n,
            ).astype(np.float32)

        self.obstacle_velocities = np.zeros(
            (c.num_obstacles, 3),
            dtype=np.float32,
        )

        if (
            c.obstacles_move
            and c.obstacle_speed > 0.0
            and c.num_obstacles > 0
        ):
            direction = rng.normal(
                size=(c.num_obstacles, 3)
            )

            direction /= np.linalg.norm(
                direction,
                axis=1,
                keepdims=True,
            )

            self.obstacle_velocities = (
                direction * c.obstacle_speed
            ).astype(np.float32)

        self.wind_angle = (
            float(rng.uniform(0.0, 2.0 * np.pi))
            if c.wind_speed > 0.0
            else 0.0
        )

        self._rng = rng

        self.coverage_grid = np.zeros(
            c.voxel_grid_shape,
            dtype=bool,
        )

        self._mark_coverage()

        self._ticks_since_drone_contact = np.full(
            n,
            c.contact_cooldown_ticks,
            dtype=np.int64,
        )

        self._ticks_since_obstacle_contact = np.full(
            n,
            c.contact_cooldown_ticks,
            dtype=np.int64,
        )

        observations = self._build_observations()
        infos = {
            agent: {} for agent in self.agents
        }

        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        c = self.config
        n = len(self.agents)

        self._advance_obstacles()

        target_speed = np.zeros(
            n,
            dtype=np.float32,
        )

        for i, agent in enumerate(self.agents):
            a = np.clip(
                actions[agent],
                self.action_space(agent).low,
                self.action_space(agent).high,
            )

            commanded_velocity, pitch_rate, yaw_rate = a

            self.headings[i, 0] += pitch_rate * c.dt
            self.headings[i, 1] += yaw_rate * c.dt

            target_speed[i] = commanded_velocity

        self.headings[:, 0] = np.clip(
            self.headings[:, 0],
            -c.max_pitch_angle,
            c.max_pitch_angle,
        )

        speed_error = target_speed - self.speed
        max_delta = c.max_acceleration * c.dt

        speed_delta = np.clip(
            speed_error,
            -max_delta,
            max_delta,
        )

        self.speed = (
            self.speed + speed_delta
        ) * (1.0 - c.drag_coefficient)

        self.speed = np.clip(
            self.speed,
            0.0,
            c.max_velocity,
        )

        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]

        direction = np.stack(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            axis=1,
        )

        self.positions += (
            direction * self.speed[:, None] * c.dt
        )

        self._apply_wind()

        self.positions[:, 0] = np.clip(
            self.positions[:, 0],
            0.0,
            c.world_size_xy,
        )

        self.positions[:, 1] = np.clip(
            self.positions[:, 1],
            0.0,
            c.world_size_xy,
        )

        self.positions[:, 2] = np.clip(
            self.positions[:, 2],
            0.0,
            c.world_height,
        )

        self.battery = np.clip(
            self.battery - c.dt / (
                c.max_episode_steps * c.dt
            ),
            0.0,
            1.0,
        )

        obstacle_collisions = (
            self._resolve_obstacle_collisions()
        )

        newly_covered = self._mark_coverage()
        drone_collisions = self._detect_collisions()

        cooldown = c.contact_cooldown_ticks

        collision_events = (
            drone_collisions
            & (self._ticks_since_drone_contact >= cooldown)
        )

        self._ticks_since_drone_contact = np.where(
            drone_collisions,
            0,
            np.minimum(
                self._ticks_since_drone_contact + 1,
                cooldown,
            ),
        )

        obstacle_events = (
            obstacle_collisions
            & (self._ticks_since_obstacle_contact >= cooldown)
        )

        self._ticks_since_obstacle_contact = np.where(
            obstacle_collisions,
            0,
            np.minimum(
                self._ticks_since_obstacle_contact + 1,
                cooldown,
            ),
        )

        rewards, reward_breakdown = (
            self.reward_shaper.compute_with_breakdown(
                newly_covered_by_agent=newly_covered,
                collisions_by_agent=collision_events,
                positions=self.positions,
                world_size_xy=c.world_size_xy,
                world_height=c.world_height,
                obstacle_collisions_by_agent=obstacle_events,
                new_voxel_counts_by_agent=self._last_new_voxel_counts,
            )
        )

        if self.reward_logger is not None:
            self.reward_logger.log_step(reward_breakdown)

        self._step_count += 1

        truncated = (
            self._step_count >= c.max_episode_steps
        )

        terminations = {
            agent: False for agent in self.agents
        }

        truncations = {
            agent: truncated for agent in self.agents
        }

        observations = self._build_observations()

        infos = {
            agent: {
                "collision": bool(drone_collisions[i]),
                "collision_event": bool(collision_events[i]),
                "obstacle_collision": bool(obstacle_collisions[i]),
                "obstacle_collision_event": bool(obstacle_events[i]),
                "battery": float(self.battery[i]),
                "speed": float(self.speed[i]),
            }
            for i, agent in enumerate(self.agents)
        }

        if truncated:
            self.agents = []

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def _advance_obstacles(self) -> None:
        """Move obstacles and reflect them at world boundaries."""
        c = self.config

        if (
            not c.obstacles_move
            or c.obstacle_speed <= 0.0
            or len(self.obstacles) == 0
        ):
            return

        world_max = np.array(
            [
                c.world_size_xy,
                c.world_size_xy,
                c.world_height,
            ],
            dtype=np.float32,
        )

        moved = (
            self.obstacles
            + self.obstacle_velocities * c.dt
        )

        below = moved < 0.0
        above = moved > world_max

        moved = np.where(
            below,
            -moved,
            moved,
        )

        moved = np.where(
            above,
            2.0 * world_max - moved,
            moved,
        )

        self.obstacle_velocities = np.where(
            below | above,
            -self.obstacle_velocities,
            self.obstacle_velocities,
        )

        self.obstacles = moved.astype(np.float32)

        self.obstacle_velocities = (
            self.obstacle_velocities.astype(np.float32)
        )

    def _apply_wind(self) -> None:
        """Apply horizontal wind drift to every drone."""
        c = self.config

        if c.wind_speed <= 0.0:
            return

        self.wind_angle += float(
            self._rng.normal(
                0.0,
                c.wind_direction_drift,
            )
        )

        wind_velocity = c.wind_speed * np.array(
            [
                np.cos(self.wind_angle),
                np.sin(self.wind_angle),
                0.0,
            ]
        )

        self.positions += (
            wind_velocity * c.dt
        ).astype(np.float32)

    def _mark_coverage(self) -> np.ndarray:
        """Mark voxels observed by each drone's mapping sensor."""
        c = self.config
        n = len(self.positions)

        newly_covered = np.zeros(
            n,
            dtype=bool,
        )

        self._last_new_voxel_counts = np.zeros(
            n,
            dtype=np.int32,
        )

        sensor_range = c.mapping_sensor_range
        voxel_size = c.voxel_size

        if sensor_range <= 0.0 or voxel_size <= 0.0:
            return newly_covered

        grid_shape = np.asarray(
            self.coverage_grid.shape
        )

        half_h_fov = np.deg2rad(
            c.mapping_horizontal_fov_deg / 2.0
        )

        half_v_fov = np.deg2rad(
            c.mapping_vertical_fov_deg / 2.0
        )

        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]

        for i in range(n):
            pos = self.positions[i]

            lower = np.maximum(
                np.floor(
                    (pos - sensor_range) / voxel_size
                ).astype(int),
                0,
            )

            upper = np.minimum(
                np.floor(
                    (pos + sensor_range) / voxel_size
                ).astype(int) + 1,
                grid_shape,
            )

            if np.any(lower >= upper):
                continue

            x0, y0, z0 = lower
            x1, y1, z1 = upper

            local_grid = self.coverage_grid[
                x0:x1,
                y0:y1,
                z0:z1,
            ]

            local_shape = local_grid.shape

            local_indices = (
                np.indices(local_shape)
                .reshape(3, -1)
                .T
            )

            local_indices += lower

            voxel_centers = (
                local_indices.astype(np.float32) + 0.5
            ) * voxel_size

            offsets = voxel_centers - pos

            distances = np.linalg.norm(
                offsets,
                axis=1,
            )

            in_range = (
                (distances <= sensor_range)
                & (distances > 0.0)
            )

            if not np.any(in_range):
                continue

            xyz = local_indices[in_range]
            vectors = offsets[in_range]

            horizontal_dist = np.linalg.norm(
                vectors[:, :2],
                axis=1,
            )

            world_yaw = np.arctan2(
                vectors[:, 1],
                vectors[:, 0],
            )

            yaw_delta = self._wrap_angle(
                world_yaw - yaw[i]
            )

            world_pitch = np.arctan2(
                vectors[:, 2],
                horizontal_dist + 1e-8,
            )

            pitch_delta = (
                world_pitch - pitch[i]
            )

            visible = (
                (np.abs(yaw_delta) <= half_h_fov)
                & (np.abs(pitch_delta) <= half_v_fov)
            )

            visible_xyz = xyz[visible]

            if visible_xyz.size == 0:
                continue

            previously_uncovered = ~self.coverage_grid[
                visible_xyz[:, 0],
                visible_xyz[:, 1],
                visible_xyz[:, 2],
            ]

            if np.any(previously_uncovered):
                newly_covered[i] = True

                self._last_new_voxel_counts[i] = int(
                    np.count_nonzero(previously_uncovered)
                )

            newly_xyz = visible_xyz[
                previously_uncovered
            ]

            self.coverage_grid[
                newly_xyz[:, 0],
                newly_xyz[:, 1],
                newly_xyz[:, 2],
            ] = True

        return newly_covered

    def _detect_collisions(self) -> np.ndarray:
        diffs = (
            self.positions[:, None, :]
            - self.positions[None, :, :]
        )

        dists = np.linalg.norm(
            diffs,
            axis=-1,
        )

        np.fill_diagonal(
            dists,
            np.inf,
        )

        close = (
            dists < self.config.min_separation
        )

        return close.any(axis=1)

    def _resolve_obstacle_collisions(self) -> np.ndarray:
        c = self.config
        n = len(self.positions)

        collided = np.zeros(
            n,
            dtype=bool,
        )

        min_dist = (
            c.drone_radius + c.obstacle_radius
        )

        if len(self.obstacles) == 0:
            return collided

        diffs = (
            self.positions[:, None, :]
            - self.obstacles[None, :, :]
        )

        dists = np.linalg.norm(
            diffs,
            axis=-1,
        )

        nearest_obstacle = np.argmin(
            dists,
            axis=1,
        )

        nearest_dist = dists[
            np.arange(n),
            nearest_obstacle,
        ]

        penetrating = nearest_dist < min_dist
        collided[penetrating] = True

        for i in np.where(penetrating)[0]:
            obs_idx = nearest_obstacle[i]

            vec = (
                self.positions[i]
                - self.obstacles[obs_idx]
            )

            dist = nearest_dist[i]

            if dist < 1e-6:
                push_dir = np.array(
                    [1.0, 0.0, 0.0],
                    dtype=np.float32,
                )
            else:
                push_dir = vec / dist

            self.positions[i] = (
                self.obstacles[obs_idx]
                + push_dir * min_dist
            )

            self.speed[i] = 0.0

        return collided

    @staticmethod
    def _wrap_angle(a: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi)."""
        return (
            (a + np.pi) % (2.0 * np.pi)
            - np.pi
        )

    def get_global_state(self) -> np.ndarray:
        """
        Return normalized global state for the centralized MAPPO critic.

        Decentralized actors continue to receive their existing local
        observations. This state is intended only for critic input.

        Fixed dimension:
            7 * num_agents + 548

        Seven features per agent:
            3 position + 2 heading + 1 speed + 1 battery

        Fixed-capacity obstacle features:
            24 obstacle positions (8 slots)
            24 obstacle velocities (8 slots)
            500 downsampled coverage values

        Stages with fewer than eight obstacles use zero-padded slots.
        """
        c = self.config

        # 1. Validate environment initialization.
        if (
            self.positions is None
            or self.headings is None
            or self.speed is None
            or self.battery is None
            or self.obstacles is None
            or self.obstacle_velocities is None
            or self.coverage_grid is None
        ):
            raise RuntimeError(
                "Environment must be reset before requesting global state."
            )

        # 2. Validate agent state arrays.
        # self.agents may be empty after episode truncation, but
        # the underlying arrays retain the complete final swarm state.
        expected_agent_shapes = {
            "positions": (c.num_agents, 3),
            "headings": (c.num_agents, 2),
            "speed": (c.num_agents,),
            "battery": (c.num_agents,),
        }

        actual_agent_shapes = {
            "positions": self.positions.shape,
            "headings": self.headings.shape,
            "speed": self.speed.shape,
            "battery": self.battery.shape,
        }

        for name, expected_shape in expected_agent_shapes.items():
            if actual_agent_shapes[name] != expected_shape:
                raise RuntimeError(
                    f"Incomplete agent state for {name}: "
                    f"expected {expected_shape}, "
                    f"received {actual_agent_shapes[name]}."
                )

        # 3. Validate obstacle configuration and arrays.
        if not 0 <= c.num_obstacles <= GLOBAL_STATE_MAX_OBSTACLES:
            raise ValueError(
                "The MAPPO global-state layout supports between "
                f"0 and {GLOBAL_STATE_MAX_OBSTACLES} obstacles; "
                f"received {c.num_obstacles}."
            )

        expected_obstacle_shape = (c.num_obstacles, 3)

        if self.obstacles.shape != expected_obstacle_shape:
            raise RuntimeError(
                "Unexpected obstacle position shape: "
                f"expected {expected_obstacle_shape}, "
                f"received {self.obstacles.shape}."
            )

        if self.obstacle_velocities.shape != expected_obstacle_shape:
            raise RuntimeError(
                "Unexpected obstacle velocity shape: "
                f"expected {expected_obstacle_shape}, "
                f"received {self.obstacle_velocities.shape}."
            )

        # 4. Agent positions.
        agent_positions = self.positions.copy()

        agent_positions[:, 0:2] /= max(
            c.world_size_xy,
            1e-6,
        )

        agent_positions[:, 2] /= max(
            c.world_height,
            1e-6,
        )

        # 5. Agent headings.
        agent_headings = (
            self.headings.copy() / np.pi
        )

        # 6. Agent speeds.
        agent_speeds = (
            self.speed.copy()
            / max(c.max_velocity, 1e-6)
        )

        # 7. Agent battery levels.
        agent_battery = (
            self.battery.copy()
            / max(c.battery_capacity, 1e-6)
        )

        # 8. Obstacle positions.
        # Reserve eight slots so the critic input dimension is stable
        # even when a curriculum stage contains fewer obstacles.
        obstacle_positions = np.zeros(
            (GLOBAL_STATE_MAX_OBSTACLES, 3),
            dtype=np.float32,
        )

        actual_obstacle_positions = self.obstacles.copy()

        actual_obstacle_positions[:, 0:2] /= max(
            c.world_size_xy,
            1e-6,
        )

        actual_obstacle_positions[:, 2] /= max(
            c.world_height,
            1e-6,
        )

        obstacle_positions[:c.num_obstacles] = (
            actual_obstacle_positions
        )

        # 9. Obstacle velocities.
        obstacle_velocities = np.zeros(
            (GLOBAL_STATE_MAX_OBSTACLES, 3),
            dtype=np.float32,
        )

        if c.obstacle_speed > 0.0:
            actual_obstacle_velocities = (
                self.obstacle_velocities.copy()
                / c.obstacle_speed
            )
        else:
            actual_obstacle_velocities = np.zeros_like(
                self.obstacle_velocities
            )

        obstacle_velocities[:c.num_obstacles] = (
            actual_obstacle_velocities
        )

        # 10. Coverage grid downsampling.
        # Original shape: (50, 50, 10)
        # Reduced shape:  (10, 10, 5)
        coverage = self.coverage_grid.astype(
            np.float32
        )

        expected_coverage_shape = (50, 50, 10)

        if coverage.shape != expected_coverage_shape:
            raise ValueError(
                f"Expected coverage grid shape "
                f"{expected_coverage_shape}, got {coverage.shape}."
            )

        coverage_small = coverage.reshape(
            10, 5,
            10, 5,
            5, 2,
        ).mean(axis=(1, 3, 5))

        # 11. Concatenate global state.
        global_state = np.concatenate(
            [
                agent_positions.flatten(),
                agent_headings.flatten(),
                agent_speeds.flatten(),
                agent_battery.flatten(),
                obstacle_positions.flatten(),
                obstacle_velocities.flatten(),
                coverage_small.flatten(),
            ]
        ).astype(np.float32)

        # 12. Validate fixed global-state dimension.
        expected_global_dim = (
            7 * c.num_agents
            + 6 * GLOBAL_STATE_MAX_OBSTACLES
            + GLOBAL_STATE_COVERAGE_FEATURES
        )

        if global_state.shape != (expected_global_dim,):
            raise RuntimeError(
                f"Unexpected global state shape: "
                f"{global_state.shape}. "
                f"Expected ({expected_global_dim},)."
            )

        # 13. Validate numerical stability.
        if not np.isfinite(global_state).all():
            raise RuntimeError(
                "Global state contains NaN or infinite values."
            )

        return global_state

    def _build_observations(self) -> dict[str, np.ndarray]:
        c = self.config
        n = len(self.agents)
        pos = self.positions
        k = c.k_nearest

        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]

        # ---- Neighbour block: distance + BODY-frame bearings ----
        diffs_all = (
            pos[None, :, :]
            - pos[:, None, :]
        )

        dists_all = np.linalg.norm(
            diffs_all,
            axis=-1,
        )

        np.fill_diagonal(
            dists_all,
            np.inf,
        )

        valid_neighbors = min(
            k,
            max(n - 1, 0),
        )

        full_sorted = np.argsort(
            dists_all,
            axis=1,
        )

        nearest_idx = full_sorted[:, :valid_neighbors]

        row_idx = np.arange(n)[:, None]

        gathered_diffs = diffs_all[
            row_idx,
            nearest_idx,
        ]

        gathered_dists = dists_all[
            row_idx,
            nearest_idx,
        ]

        gathered_dists = np.minimum(
            gathered_dists,
            c.lidar_range,
        )

        world_bearing_xy = np.arctan2(
            gathered_diffs[:, :, 1],
            gathered_diffs[:, :, 0],
        )

        world_bearing_z = np.arctan2(
            gathered_diffs[:, :, 2],
            np.linalg.norm(
                gathered_diffs[:, :, :2],
                axis=-1,
            ) + 1e-6,
        )

        bearing_xy = self._wrap_angle(
            world_bearing_xy - yaw[:, None]
        )

        bearing_z = (
            world_bearing_z - pitch[:, None]
        )

        pad_count = k - valid_neighbors

        if pad_count > 0:
            pad_dist = np.full(
                (n, pad_count),
                c.lidar_range,
                dtype=np.float32,
            )

            pad_bearing = np.zeros(
                (n, pad_count),
                dtype=np.float32,
            )

            gathered_dists = np.concatenate(
                [gathered_dists, pad_dist],
                axis=1,
            )

            bearing_xy = np.concatenate(
                [bearing_xy, pad_bearing],
                axis=1,
            )

            bearing_z = np.concatenate(
                [bearing_z, pad_bearing],
                axis=1,
            )

        neighbor_feats = np.stack(
            [
                gathered_dists,
                bearing_xy,
                bearing_z,
            ],
            axis=-1,
        )

        neighbor_feats = neighbor_feats.reshape(
            n,
            k * 3,
        ).astype(np.float32)

        # ---- Nearest obstacle ----
        if len(self.obstacles) > 0:
            to_obstacles = (
                self.obstacles[None, :, :]
                - pos[:, None, :]
            )

            obs_dists = np.linalg.norm(
                to_obstacles,
                axis=-1,
            )

            nearest_obs = np.argmin(
                obs_dists,
                axis=1,
            )

            nearest_obs_dist = obs_dists[
                np.arange(n),
                nearest_obs,
            ]

            nearest_obstacle = np.minimum(
                nearest_obs_dist,
                c.lidar_range,
            )

            to_nearest = to_obstacles[
                np.arange(n),
                nearest_obs,
            ]

            obstacle_bearing = self._wrap_angle(
                np.arctan2(
                    to_nearest[:, 1],
                    to_nearest[:, 0],
                ) - yaw
            )

            obstacle_bearing = np.where(
                nearest_obs_dist < c.lidar_range,
                obstacle_bearing,
                0.0,
            )

        else:
            nearest_obstacle = np.full(
                n,
                c.lidar_range,
                dtype=np.float32,
            )

            obstacle_bearing = np.zeros(
                n,
                dtype=np.float32,
            )

        # ---- Coverage ahead ----
        direction = np.stack(
            [
                np.cos(pitch) * np.cos(yaw),
                np.cos(pitch) * np.sin(yaw),
                np.sin(pitch),
            ],
            axis=1,
        )

        world_max = np.array(
            [
                c.world_size_xy,
                c.world_size_xy,
                c.world_height,
            ],
            dtype=np.float32,
        )

        ahead = np.clip(
            pos + direction * c.coverage_lookahead,
            0.0,
            world_max,
        )

        grid_shape = np.array(
            self.coverage_grid.shape
        )

        ahead_idx = np.minimum(
            (ahead // c.voxel_size).astype(int),
            grid_shape - 1,
        )

        padded = np.pad(
            self.coverage_grid,
            1,
            constant_values=True,
        )

        covered_count = np.zeros(
            n,
            dtype=np.float32,
        )

        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    covered_count += padded[
                        ahead_idx[:, 0] + dx,
                        ahead_idx[:, 1] + dy,
                        ahead_idx[:, 2] + dz,
                    ]

        coverage_ahead = (
            covered_count / 27.0
        )

        battery_remaining = (
            self.battery.astype(np.float32)
        )

        # ---- Wall features ----
        x = pos[:, 0]
        y = pos[:, 1]
        z = pos[:, 2]

        wall_dists = np.stack(
            [
                x,
                c.world_size_xy - x,
                y,
                c.world_size_xy - y,
            ],
            axis=1,
        )

        nearest_wall = np.argmin(
            wall_dists,
            axis=1,
        )

        wall_dist = wall_dists[
            np.arange(n),
            nearest_wall,
        ]

        wall_world_dir = np.array(
            [
                np.pi,
                0.0,
                -np.pi / 2.0,
                np.pi / 2.0,
            ]
        )[nearest_wall]

        wall_bearing = self._wrap_angle(
            wall_world_dir - yaw
        )

        wall_in_range = (
            wall_dist < c.lidar_range
        )

        wall_bearing = np.where(
            wall_in_range,
            wall_bearing,
            0.0,
        )

        wall_dist = np.minimum(
            wall_dist,
            c.lidar_range,
        )

        half_h = c.world_height / 2.0

        altitude_norm = (
            (z - half_h) / half_h
        )

        obs_matrix = np.concatenate(
            [
                neighbor_feats,
                nearest_obstacle[:, None],
                coverage_ahead[:, None],
                battery_remaining[:, None],
                wall_dist[:, None],
                wall_bearing[:, None],
                altitude_norm[:, None],
                pitch[:, None],
                obstacle_bearing[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        return {
            agent: obs_matrix[i]
            for i, agent in enumerate(self.agents)
        }

    def render(self):
        if self.render_mode != "human":
            return

        import matplotlib.pyplot as plt

        if self._fig is None:
            plt.ion()

            self._fig = plt.figure(
                figsize=(8, 8)
            )

            self._ax = self._fig.add_subplot(
                111,
                projection="3d",
            )

        c = self.config
        ax = self._ax
        ax.cla()

        ax.scatter(
            self.positions[:, 0],
            self.positions[:, 1],
            self.positions[:, 2],
            c="tab:blue",
            s=25,
            label="drones",
            depthshade=True,
        )

        if len(self.obstacles) > 0:
            ax.scatter(
                self.obstacles[:, 0],
                self.obstacles[:, 1],
                self.obstacles[:, 2],
                c="tab:red",
                s=200,
                marker="^",
                label="obstacles",
                alpha=0.7,
            )

        ax.set_xlim(
            0,
            c.world_size_xy,
        )

        ax.set_ylim(
            0,
            c.world_size_xy,
        )

        ax.set_zlim(
            0,
            c.world_height,
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        coverage_pct = (
            self.coverage_grid.mean() * 100
            if self.coverage_grid is not None
            else 0.0
        )

        ax.set_title(
            f"step={self._step_count}  "
            f"agents={len(self.agents)}  "
            f"coverage={coverage_pct:.1f}%"
        )

        ax.legend(loc="upper right")
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)

            self._fig = None
            self._ax = None