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

        # Obs vector layout per agent:
        #   k_nearest * (distance, bearing_xy, bearing_z) neighbor readings
        #   + 1 distance-to-nearest-obstacle
        #   + 1 local coverage-density
        #   + 1 remaining battery/time budget
        self._obs_dim = self.config.k_nearest * 3 + 3

        self.reward_shaper = RewardShaper(self.config)

        # Runtime state, populated in reset()
        self.positions: np.ndarray | None = None      # (N, 3)
        self.headings: np.ndarray | None = None        # (N, 2) pitch, yaw
        self.battery: np.ndarray | None = None          # (N,)
        self.obstacles: np.ndarray | None = None        # (num_obstacles, 3)
        self.coverage_grid: np.ndarray | None = None    # bool, voxel_grid_shape
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
        self.headings = np.zeros((n, 2), dtype=np.float32)  # pitch, yaw
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

        vel = np.zeros(n, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            a = np.clip(actions[agent], self.action_space(agent).low, self.action_space(agent).high)
            velocity, pitch_rate, yaw_rate = a
            self.headings[i, 0] += pitch_rate * c.dt
            self.headings[i, 1] += yaw_rate * c.dt
            vel[i] = velocity

        # Kinematic integration: move each drone along its heading vector.
        pitch = self.headings[:, 0]
        yaw = self.headings[:, 1]
        direction = np.stack(
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)],
            axis=1,
        )
        self.positions += direction * vel[:, None] * c.dt
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0.0, c.world_size_xy)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0.0, c.world_size_xy)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0.0, c.world_height)

        self.battery = np.clip(self.battery - c.dt / (c.max_episode_steps * c.dt), 0.0, 1.0)

        newly_covered = self._mark_coverage()
        collisions = self._detect_collisions()

        rewards = self.reward_shaper.compute(
            newly_covered_by_agent=newly_covered,
            collisions_by_agent=collisions,
            positions=self.positions,
            world_size_xy=c.world_size_xy,
            world_height=c.world_height,
        )

        self._step_count += 1
        truncated = self._step_count >= c.max_episode_steps
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        observations = self._build_observations()
        infos = {
            agent: {"collision": bool(collisions[i]), "battery": float(self.battery[i])}
            for i, agent in enumerate(self.agents)
        }

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _mark_coverage(self) -> np.ndarray:
        """Voxelize current positions, mark visited cells, return a
        per-agent bool array of whether that agent revealed a new cell
        this tick (drives the +1 exploration reward)."""
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
        """O(N^2) pairwise distance check. Fine for N=50; swap for a
        spatial hash / KD-tree if the swarm size grows substantially."""
        diffs = self.positions[:, None, :] - self.positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        close = dists < self.config.min_separation
        return close.any(axis=1)

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
                neighbor_feats.extend([c.lidar_range, 0.0, 0.0])  # pad if <k neighbors

            obstacle_dists = np.linalg.norm(self.obstacles - self.positions[i], axis=1)
            nearest_obstacle = float(min(obstacle_dists.min(), c.lidar_range))

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