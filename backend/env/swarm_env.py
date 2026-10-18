"""SwarmEnv: a continuous 3D disaster-zone search environment for
50 autonomous drones, built on PettingZoo's ParallelEnv API."""
from __future__ import annotations

import functools

import numpy as np
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv

from backend.env.config import SwarmEnvConfig
from backend.reward.reward_shaping import RewardShaper


class SwarmEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "none"], "name": "swarmrl_v0"}

    def __init__(self, config: SwarmEnvConfig | None = None, render_mode: str = "none"):
        self.config = config or SwarmEnvConfig()
        self.render_mode = render_mode

        self.possible_agents = [f"drone_{i}" for i in range(self.config.num_agents)]
        self.agents = list(self.possible_agents)

        self._obs_dim = self.config.k_nearest * 3 + 3
        self.reward_shaper = RewardShaper(self.config)
        self.reward_logger = None

        self.positions: np.ndarray | None = None
        self.headings: np.ndarray | None = None
        self.speed: np.ndarray | None = None
        self.battery: np.ndarray | None = None
        self.obstacles: np.ndarray | None = None
        self.coverage_grid: np.ndarray | None = None
        self._step_count = 0

        # Day 6: drone-drone collisions are rewarded per EVENT (the tick a
        # collision starts), not per tick of overlap. This remembers last
        # tick's collision flags so we can detect the rising edge.
        self._prev_drone_collisions: np.ndarray | None = None
        # Same idea for drone-obstacle contact: a drone still pointed at an
        # obstacle re-penetrates every tick, so charging per tick re-bills
        # the same contact hundreds of times.
        self._prev_obstacle_collisions: np.ndarray | None = None

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

    def reset(self, seed=None, options=None):
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

        self.coverage_grid = np.zeros(c.voxel_grid_shape, dtype=bool)
        self._mark_coverage()
        self._prev_drone_collisions = np.zeros(n, dtype=bool)
        self._prev_obstacle_collisions = np.zeros(n, dtype=bool)

        observations = self._build_observations()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        c = self.config
        n = len(self.agents)

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
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.0, c.world_size_xy)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.0, c.world_size_xy)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0.0, c.world_height)

        self.battery = np.clip(self.battery - c.dt / (c.max_episode_steps * c.dt), 0.0, 1.0)

        obstacle_collisions = self._resolve_obstacle_collisions()
        newly_covered = self._mark_coverage()
        drone_collisions = self._detect_collisions()           # state: overlapping now
        collision_events = drone_collisions & ~self._prev_drone_collisions  # rising edge only
        self._prev_drone_collisions = drone_collisions
        obstacle_events = obstacle_collisions & ~self._prev_obstacle_collisions
        self._prev_obstacle_collisions = obstacle_collisions

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

    def _build_observations(self) -> dict[str, np.ndarray]:
        c = self.config
        n = len(self.agents)
        pos = self.positions
        k = c.k_nearest

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

        bearing_xy = np.arctan2(gathered_diffs[:, :, 1], gathered_diffs[:, :, 0])
        bearing_z = np.arctan2(
            gathered_diffs[:, :, 2],
            np.linalg.norm(gathered_diffs[:, :, :2], axis=-1) + 1e-6,
        )

        pad_count = k - valid_neighbors
        if pad_count > 0:
            pad_dist = np.full((n, pad_count), c.lidar_range, dtype=np.float32)
            pad_bearing = np.zeros((n, pad_count), dtype=np.float32)
            gathered_dists = np.concatenate([gathered_dists, pad_dist], axis=1)
            bearing_xy = np.concatenate([bearing_xy, pad_bearing], axis=1)
            bearing_z = np.concatenate([bearing_z, pad_bearing], axis=1)

        neighbor_feats = np.stack([gathered_dists, bearing_xy, bearing_z], axis=-1)
        neighbor_feats = neighbor_feats.reshape(n, k * 3).astype(np.float32)

        if len(self.obstacles) > 0:
            obs_diffs = pos[:, None, :] - self.obstacles[None, :, :]
            obs_dists = np.linalg.norm(obs_diffs, axis=-1)
            nearest_obstacle = np.minimum(obs_dists.min(axis=1), c.lidar_range)
        else:
            nearest_obstacle = np.full(n, c.lidar_range, dtype=np.float32)

        voxel_idx = (pos // c.voxel_size).astype(int)
        voxel_idx[:, 0] = np.clip(voxel_idx[:, 0], 0, self.coverage_grid.shape[0] - 1)
        voxel_idx[:, 1] = np.clip(voxel_idx[:, 1], 0, self.coverage_grid.shape[1] - 1)
        voxel_idx[:, 2] = np.clip(voxel_idx[:, 2], 0, self.coverage_grid.shape[2] - 1)
        local_density = self.coverage_grid[
            voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
        ].astype(np.float32)

        battery_remaining = self.battery.astype(np.float32)

        obs_matrix = np.concatenate(
            [
                neighbor_feats,
                nearest_obstacle[:, None].astype(np.float32),
                local_density[:, None],
                battery_remaining[:, None],
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