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


class SwarmEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "none"], "name": "swarmrl_v0"}

    def __init__(self, config: SwarmEnvConfig | None = None, render_mode: str = "none"):
        self.config = config or SwarmEnvConfig()
        self.render_mode = render_mode

        self.possible_agents = [f"drone_{i}" for i in range(self.config.num_agents)]
        self.agents = list(self.possible_agents)

        # Observation layout (Day 7), per agent:
        #   [0 : 3k]  k nearest neighbours as (distance, bearing_xy, bearing_z),
        #             bearings in the drone's BODY frame (relative to its heading)
        #   [3k]      distance to nearest obstacle
        #   [3k+1]    coverage ahead: fraction of explored voxels in the 3x3x3
        #             block `coverage_lookahead` metres in front of the drone
        #   [3k+2]    battery remaining
        #   [3k+3]    distance to nearest xy wall (clipped to lidar_range)
        #   [3k+4]    body-frame bearing to that wall (0 if none within range)
        #   [3k+5]    altitude, normalised to [-1, 1] (0 = mid-height)
        #   [3k+6]    own pitch (rad)
        #   [3k+7]    body-frame bearing to the nearest obstacle (0 if none in range)
        self._obs_dim = self.config.k_nearest * 3 + 8
        self.reward_shaper = RewardShaper(self.config)
        self.reward_logger = None

        self.positions: np.ndarray | None = None
        self.headings: np.ndarray | None = None
        self.speed: np.ndarray | None = None
        self.battery: np.ndarray | None = None
        self.obstacles: np.ndarray | None = None
        self.coverage_grid: np.ndarray | None = None
        self._step_count = 0

        # Collisions are rewarded per EVENT, not per tick of overlap (Day 6), and
        # a new event needs `contact_cooldown_ticks` clear ticks since the last
        # contact (Day 8), so a drone flickering in and out of contact while
        # sliding along a surface is billed once. These count the ticks since
        # each drone was last in contact.
        self._ticks_since_drone_contact: np.ndarray | None = None
        self._ticks_since_obstacle_contact: np.ndarray | None = None

        # Day 8: dynamics. Obstacle velocities are all zero in a static world.
        self.obstacle_velocities: np.ndarray | None = None   # (num_obstacles, 3) m/s
        self.wind_angle: float = 0.0                         # rad, direction the wind blows toward
        self._rng: np.random.Generator | None = None         # carried from reset() so wind is seed-reproducible
        self._pending_stage = None
        self.stage_name: str | None = None

        self._fig = None
        self._ax = None

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        c = self.config
        return Box(
            low=np.array([0.0, -c.max_pitch_rate, -c.max_yaw_rate], dtype=np.float32),
            high=np.array([c.max_velocity, c.max_pitch_rate, c.max_yaw_rate], dtype=np.float32),
        )

    def set_stage(self, stage: "CurriculumStage") -> None:
        """Queue a curriculum stage. It takes effect at the NEXT reset(), never
        mid-episode: changing the obstacle count under a running episode would
        corrupt it. Safe to call at any time, including from an RLlib callback
        on a reused env instance."""
        self._pending_stage = stage

    def reset(self, seed=None, options=None):
        if self._pending_stage is not None:
            self.config = self._pending_stage.apply_to(self.config)
            self.reward_shaper = RewardShaper(self.config)   # reward_logger is kept
            self.stage_name = self._pending_stage.name
            self._pending_stage = None
        rng = np.random.default_rng(seed)
        c = self.config
        self.agents = list(self.possible_agents)
        self._step_count = 0

        n = c.num_agents
        self.positions = rng.uniform(
            low=[0.0, 0.0, 1.0],
            high=[c.world_size_xy, c.world_size_xy, c.world_height],
            size=(n, 3),
        ).astype(np.float32)
        self.headings = np.zeros((n, 2), dtype=np.float32)
        self.speed = np.zeros(n, dtype=np.float32)
        self.battery = np.full(n, c.battery_capacity, dtype=np.float32)

        self.obstacles = rng.uniform(
            low=[0.0, 0.0, 0.0],
            high=[c.world_size_xy, c.world_size_xy, c.world_height],
            size=(c.num_obstacles, 3),
        ).astype(np.float32)

        # Drawn after positions and obstacles so a given seed still produces
        # the same positions and obstacles as before this option existed.
        if c.randomize_initial_heading:
            self.headings[:, 1] = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(np.float32)

        # Day 8 draws come LAST and only when the feature is on, so a static,
        # windless config consumes exactly the same random stream as before.
        self.obstacle_velocities = np.zeros((c.num_obstacles, 3), dtype=np.float32)
        if c.obstacles_move and c.obstacle_speed > 0.0 and c.num_obstacles > 0:
            direction = rng.normal(size=(c.num_obstacles, 3))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            self.obstacle_velocities = (direction * c.obstacle_speed).astype(np.float32)
        self.wind_angle = float(rng.uniform(0.0, 2.0 * np.pi)) if c.wind_speed > 0.0 else 0.0
        self._rng = rng

        self.coverage_grid = np.zeros(c.voxel_grid_shape, dtype=bool)
        self._mark_coverage()
        # Start "long clear" so the very first contact of an episode counts.
        self._ticks_since_drone_contact = np.full(n, c.contact_cooldown_ticks, dtype=np.int64)
        self._ticks_since_obstacle_contact = np.full(n, c.contact_cooldown_ticks, dtype=np.int64)

        observations = self._build_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        c = self.config
        n = len(self.agents)

        self._advance_obstacles()

        target_speed = np.zeros(n, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            a = np.clip(actions[agent], self.action_space(agent).low, self.action_space(agent).high)
            commanded_velocity, pitch_rate, yaw_rate = a
            self.headings[i, 0] += pitch_rate * c.dt
            self.headings[i, 1] += yaw_rate * c.dt
            target_speed[i] = commanded_velocity

        # Day 6 fix: bound pitch so drones can't accumulate past the climb
        # limit and end up flying upside-down (see config.max_pitch_angle).
        self.headings[:, 0] = np.clip(self.headings[:, 0], -c.max_pitch_angle, c.max_pitch_angle)

        speed_error = target_speed - self.speed
        max_delta = c.max_acceleration * c.dt
        speed_delta = np.clip(speed_error, -max_delta, max_delta)
        self.speed = (self.speed + speed_delta) * (1.0 - c.drag_coefficient)
        self.speed = np.clip(self.speed, 0.0, c.max_velocity)

        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]
        direction = np.stack(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
            axis=1,
        )
        self.positions += direction * self.speed[:, None] * c.dt
        self._apply_wind()
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.0, c.world_size_xy)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.0, c.world_size_xy)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0.0, c.world_height)

        self.battery = np.clip(self.battery - c.dt / (c.max_episode_steps * c.dt), 0.0, 1.0)

        obstacle_collisions = self._resolve_obstacle_collisions()
        newly_covered = self._mark_coverage()
        drone_collisions = self._detect_collisions()           # state: overlapping now
        cooldown = c.contact_cooldown_ticks
        collision_events = drone_collisions & (self._ticks_since_drone_contact >= cooldown)
        self._ticks_since_drone_contact = np.where(
            drone_collisions, 0, np.minimum(self._ticks_since_drone_contact + 1, cooldown))
        obstacle_events = obstacle_collisions & (self._ticks_since_obstacle_contact >= cooldown)
        self._ticks_since_obstacle_contact = np.where(
            obstacle_collisions, 0, np.minimum(self._ticks_since_obstacle_contact + 1, cooldown))

        rewards, reward_breakdown = self.reward_shaper.compute_with_breakdown(
            newly_covered_by_agent=newly_covered,
            collisions_by_agent=collision_events,
            positions=self.positions,
            world_size_xy=c.world_size_xy,
            world_height=c.world_height,
            obstacle_collisions_by_agent=obstacle_events,
        )
        if self.reward_logger is not None:
            self.reward_logger.log_step(reward_breakdown)

        self._step_count += 1
        truncated = self._step_count >= c.max_episode_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

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

        return observations, rewards, terminations, truncations, infos

    def _advance_obstacles(self) -> None:
        """Move each obstacle along its fixed velocity and bounce it off the
        world boundary (reflect the position and flip that velocity component)."""
        c = self.config
        if not (c.obstacles_move and c.obstacle_speed > 0.0) or len(self.obstacles) == 0:
            return
        world_max = np.array([c.world_size_xy, c.world_size_xy, c.world_height], dtype=np.float32)
        moved = self.obstacles + self.obstacle_velocities * c.dt
        below = moved < 0.0
        above = moved > world_max
        moved = np.where(below, -moved, moved)
        moved = np.where(above, 2.0 * world_max - moved, moved)
        self.obstacle_velocities = np.where(below | above, -self.obstacle_velocities, self.obstacle_velocities)
        self.obstacles = moved.astype(np.float32)
        self.obstacle_velocities = self.obstacle_velocities.astype(np.float32)

    def _apply_wind(self) -> None:
        """Horizontal wind drift on every drone. It displaces the drone but does
        not touch self.speed (the drone's own speed through the air)."""
        c = self.config
        if c.wind_speed <= 0.0:
            return
        self.wind_angle += float(self._rng.normal(0.0, c.wind_direction_drift))
        wind_velocity = c.wind_speed * np.array(
            [np.cos(self.wind_angle), np.sin(self.wind_angle), 0.0]
        )
        self.positions += (wind_velocity * c.dt).astype(np.float32)

    def _mark_coverage(self) -> np.ndarray:
        c = self.config
        idx = (self.positions // c.voxel_size).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.coverage_grid.shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.coverage_grid.shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.coverage_grid.shape[2] - 1)

        newly_covered = np.zeros(len(self.positions), dtype=bool)
        for i, (x, y, z) in enumerate(idx):
            if not self.coverage_grid[x, y, z]:
                self.coverage_grid[x, y, z] = True
                newly_covered[i] = True
        return newly_covered

    def _detect_collisions(self) -> np.ndarray:
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        close = dists < self.config.min_separation
        return close.any(axis=1)

    def _resolve_obstacle_collisions(self) -> np.ndarray:
        c = self.config
        n = len(self.positions)
        collided = np.zeros(n, dtype=bool)
        min_dist = c.drone_radius + c.obstacle_radius

        if len(self.obstacles) == 0:
            return collided

        diffs = self.positions[:, None, :] - self.obstacles[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        nearest_obstacle = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(n), nearest_obstacle]

        penetrating = nearest_dist < min_dist
        collided[penetrating] = True

        for i in np.where(penetrating)[0]:
            obs_idx = nearest_obstacle[i]
            vec = self.positions[i] - self.obstacles[obs_idx]
            dist = nearest_dist[i]
            if dist < 1e-6:
                push_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                push_dir = vec / dist
            self.positions[i] = self.obstacles[obs_idx] + push_dir * min_dist
            self.speed[i] = 0.0

        return collided

    @staticmethod
    def _wrap_angle(a: np.ndarray) -> np.ndarray:
        """Wrap angles to [-pi, pi)."""
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _build_observations(self) -> dict[str, np.ndarray]:
        c = self.config
        n = len(self.agents)
        pos = self.positions
        k = c.k_nearest
        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]

        # ---- Neighbour block: distance + BODY-frame bearings ----
        # diffs_all[i, j] = pos[j] - pos[i]  (vector FROM agent i TO agent j)
        diffs_all = pos[None, :, :] - pos[:, None, :]
        dists_all = np.linalg.norm(diffs_all, axis=-1)
        np.fill_diagonal(dists_all, np.inf)

        valid_neighbors = min(k, max(n - 1, 0))
        full_sorted = np.argsort(dists_all, axis=1)
        nearest_idx = full_sorted[:, :valid_neighbors]

        row_idx = np.arange(n)[:, None]
        gathered_diffs = diffs_all[row_idx, nearest_idx]
        gathered_dists = dists_all[row_idx, nearest_idx]
        gathered_dists = np.minimum(gathered_dists, c.lidar_range)

        # World-frame bearings, then rotated into the drone's own frame:
        # bearing 0 = dead ahead, +pi/2 = to its left. Without this the
        # policy cannot tell which way to turn to avoid a neighbour.
        world_bearing_xy = np.arctan2(gathered_diffs[:, :, 1], gathered_diffs[:, :, 0])
        world_bearing_z = np.arctan2(
            gathered_diffs[:, :, 2],
            np.linalg.norm(gathered_diffs[:, :, :2], axis=-1) + 1e-6,
        )
        bearing_xy = self._wrap_angle(world_bearing_xy - yaw[:, None])
        bearing_z = world_bearing_z - pitch[:, None]

        pad_count = k - valid_neighbors
        if pad_count > 0:
            pad_dist = np.full((n, pad_count), c.lidar_range, dtype=np.float32)
            pad_bearing = np.zeros((n, pad_count), dtype=np.float32)
            gathered_dists = np.concatenate([gathered_dists, pad_dist], axis=1)
            bearing_xy = np.concatenate([bearing_xy, pad_bearing], axis=1)
            bearing_z = np.concatenate([bearing_z, pad_bearing], axis=1)

        neighbor_feats = np.stack([gathered_dists, bearing_xy, bearing_z], axis=-1)
        neighbor_feats = neighbor_feats.reshape(n, k * 3).astype(np.float32)

        # ---- Nearest obstacle: distance + BODY-frame bearing ----
        if len(self.obstacles) > 0:
            to_obstacles = self.obstacles[None, :, :] - pos[:, None, :]   # FROM drone TO obstacle
            obs_dists = np.linalg.norm(to_obstacles, axis=-1)
            nearest_obs = np.argmin(obs_dists, axis=1)
            nearest_obs_dist = obs_dists[np.arange(n), nearest_obs]
            nearest_obstacle = np.minimum(nearest_obs_dist, c.lidar_range)
            to_nearest = to_obstacles[np.arange(n), nearest_obs]
            obstacle_bearing = self._wrap_angle(np.arctan2(to_nearest[:, 1], to_nearest[:, 0]) - yaw)
            obstacle_bearing = np.where(nearest_obs_dist < c.lidar_range, obstacle_bearing, 0.0)
        else:
            nearest_obstacle = np.full(n, c.lidar_range, dtype=np.float32)
            obstacle_bearing = np.zeros(n, dtype=np.float32)

        # ---- Coverage ahead ----
        # Fraction of explored voxels in the 3x3x3 block centred
        # `coverage_lookahead` metres in front of the drone. Cells outside the
        # world count as explored, so walls never look like places worth
        # flying to.
        direction = np.stack(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
            axis=1,
        )
        world_max = np.array([c.world_size_xy, c.world_size_xy, c.world_height], dtype=np.float32)
        ahead = np.clip(pos + direction * c.coverage_lookahead, 0.0, world_max)
        grid_shape = np.array(self.coverage_grid.shape)
        ahead_idx = np.minimum((ahead // c.voxel_size).astype(int), grid_shape - 1)

        padded = np.pad(self.coverage_grid, 1, constant_values=True)
        covered_count = np.zeros(n, dtype=np.float32)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    covered_count += padded[
                        ahead_idx[:, 0] + dx, ahead_idx[:, 1] + dy, ahead_idx[:, 2] + dz
                    ]
        coverage_ahead = covered_count / 27.0

        battery_remaining = self.battery.astype(np.float32)

        # ---- Wall features (body frame) ----
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        wall_dists = np.stack(
            [x, c.world_size_xy - x, y, c.world_size_xy - y], axis=1
        )  # distance to the x=0, x=W, y=0, y=W walls
        nearest_wall = np.argmin(wall_dists, axis=1)
        wall_dist = wall_dists[np.arange(n), nearest_wall]
        # World direction pointing at each wall: x=0 -> pi, x=W -> 0, y=0 -> -pi/2, y=W -> pi/2
        wall_world_dir = np.array([np.pi, 0.0, -np.pi / 2.0, np.pi / 2.0])[nearest_wall]
        wall_bearing = self._wrap_angle(wall_world_dir - yaw)
        wall_in_range = wall_dist < c.lidar_range
        wall_bearing = np.where(wall_in_range, wall_bearing, 0.0)  # "nothing sensed" convention
        wall_dist = np.minimum(wall_dist, c.lidar_range)

        half_h = c.world_height / 2.0
        altitude_norm = (z - half_h) / half_h

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

        return {agent: obs_matrix[i] for i, agent in enumerate(self.agents)}

    def render(self):
        if self.render_mode != "human":
            return
        import matplotlib.pyplot as plt

        if self._fig is None:
            plt.ion()
            self._fig = plt.figure(figsize=(8, 8))
            self._ax = self._fig.add_subplot(111, projection="3d")

        c = self.config
        ax = self._ax
        ax.cla()
        ax.scatter(
            self.positions[:, 0], self.positions[:, 1], self.positions[:, 2],
            c="tab:blue", s=25, label="drones", depthshade=True,
        )
        if len(self.obstacles) > 0:
            ax.scatter(
                self.obstacles[:, 0], self.obstacles[:, 1], self.obstacles[:, 2],
                c="tab:red", s=200, marker="^", label="obstacles", alpha=0.7,
            )
        ax.set_xlim(0, c.world_size_xy)
        ax.set_ylim(0, c.world_size_xy)
        ax.set_zlim(0, c.world_height)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        coverage_pct = self.coverage_grid.mean() * 100 if self.coverage_grid is not None else 0.0
        ax.set_title(f"step={self._step_count}  agents={len(self.agents)}  coverage={coverage_pct:.1f}%")
        ax.legend(loc="upper right")
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None