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

        # Runtime state, populated in reset()
        self.positions: np.ndarray | None = None
        self.headings: np.ndarray | None = None
        self.speed: np.ndarray | None = None            # (N,) current actual speed (momentum)
        self.battery: np.ndarray | None = None
        self.obstacles: np.ndarray | None = None
        self.coverage_grid: np.ndarray | None = None
        self._step_count = 0

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
        self.speed = np.zeros(n, dtype=np.float32)           # start at rest
        self.battery = np.full(n, c.battery_capacity, dtype=np.float32)

        self.obstacles = rng.uniform(
            low=[0.0, 0.0, 0.0],
            high=[c.world_size_xy, c.world_size_xy, c.world_height],
            size=(c.num_obstacles, 3),
        ).astype(np.float32)

        self.coverage_grid = np.zeros(c.voxel_grid_shape, dtype=bool)
        self._mark_coverage()

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

        # --- Momentum model ---
        # Actual speed eases toward the commanded target under an
        # acceleration cap, then loses a small fraction to drag each tick.
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
        drone_collisions = self._detect_collisions()

        rewards = self.reward_shaper.compute(
            newly_covered_by_agent=newly_covered,
            collisions_by_agent=drone_collisions,
            positions=self.positions,
            world_size_xy=c.world_size_xy,
            world_height=c.world_height,
            obstacle_collisions_by_agent=obstacle_collisions,
        )

        self._step_count += 1
        truncated = self._step_count >= c.max_episode_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        observations = self._build_observations()
        infos = {
            agent: {
                "collision": bool(drone_collisions[i]),
                "obstacle_collision": bool(obstacle_collisions[i]),
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
        """Obstacles are now physically solid (Day 1 only sensed them).
        Pushes penetrating drones back to the obstacle surface and zeroes
        their speed, so re-penetrating next tick isn't free."""
        c = self.config
        n = len(self.positions)
        collided = np.zeros(n, dtype=bool)
        min_dist = c.drone_radius + c.obstacle_radius

        if len(self.obstacles) == 0:
            return collided  # no obstacles configured (e.g. curriculum stage 1)

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
        obs = {}
        for i, agent in enumerate(self.agents):
            diffs = self.positions - self.positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nearest_idx = np.argsort(dists)[: c.k_nearest]

            neighbor_feats = []
            for j in nearest_idx:
                d = min(dists[j], c.lidar_range)
                bearing_xy = np.arctan2(diffs[j, 1], diffs[j, 0])
                bearing_z = np.arctan2(diffs[j, 2], np.linalg.norm(diffs[j, :2]) + 1e-6)
                neighbor_feats.extend([d, bearing_xy, bearing_z])
            while len(neighbor_feats) < c.k_nearest * 3:
                neighbor_feats.extend([c.lidar_range, 0.0, 0.0])

            if len(self.obstacles) > 0:
                obstacle_dists = np.linalg.norm(self.obstacles - self.positions[i], axis=1)
                nearest_obstacle = float(min(obstacle_dists.min(), c.lidar_range))
            else:
                nearest_obstacle = c.lidar_range

            vx, vy, vz = (self.positions[i] // c.voxel_size).astype(int)
            vx = np.clip(vx, 0, self.coverage_grid.shape[0] - 1)
            vy = np.clip(vy, 0, self.coverage_grid.shape[1] - 1)
            vz = np.clip(vz, 0, self.coverage_grid.shape[2] - 1)
            local_density = float(self.coverage_grid[vx, vy, vz])
            battery_remaining = float(self.battery[i])

            obs[agent] = np.array(
                neighbor_feats + [nearest_obstacle, local_density, battery_remaining],
                dtype=np.float32,
            )
        return obs

    def render(self):
        if self.render_mode == "human":
            print(f"step={self._step_count} agents={len(self.agents)} "
                  f"coverage={self.coverage_grid.mean():.3f}")

    def close(self):
        pass