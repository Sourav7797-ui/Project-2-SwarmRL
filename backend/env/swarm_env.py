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

        # Lazy-initialized matplotlib state for render_mode="human" (Day 3)
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
        """Vectorized version of the per-agent observation builder (Day 3).
        Replaces the Day 1/2 Python-loop version with broadcasted numpy ops —
        this function gets called every tick, including millions of times
        during Task 3's RL rollouts, so the loop-vs-vector difference matters
        far more here than it does in, say, _resolve_obstacle_collisions.

        Produces bit-for-bit the same values as the old loop version (see
        tests/test_day3_edge_cases.py::test_vectorized_matches_reference_loop),
        just computed without a per-agent Python loop.
        """
        c = self.config
        n = len(self.agents)
        pos = self.positions
        k = c.k_nearest

        # ---- Neighbor distance/bearing block ----
        # diffs_all[i, j] = pos[j] - pos[i]  (vector FROM agent i TO agent j),
        # matching the reference loop's `diffs = positions - positions[i]`
        # convention. Getting this backwards flips every bearing's sign
        # while leaving distances (norm is sign-blind) looking correct —
        # an easy way to ship a silently-wrong observation.
        diffs_all = pos[None, :, :] - pos[:, None, :]            # (n, n, 3)
        dists_all = np.linalg.norm(diffs_all, axis=-1)           # (n, n)
        np.fill_diagonal(dists_all, np.inf)                      # exclude self

        # Self always sorts last (distance=inf), so taking the first
        # `valid_neighbors` columns of the full sort never includes self
        # as long as valid_neighbors <= n - 1.
        valid_neighbors = min(k, max(n - 1, 0))
        full_sorted = np.argsort(dists_all, axis=1)              # (n, n)
        nearest_idx = full_sorted[:, :valid_neighbors]           # (n, valid_neighbors)

        row_idx = np.arange(n)[:, None]                          # (n, 1), broadcasts
        gathered_diffs = diffs_all[row_idx, nearest_idx]         # (n, valid_neighbors, 3)
        gathered_dists = dists_all[row_idx, nearest_idx]         # (n, valid_neighbors)
        gathered_dists = np.minimum(gathered_dists, c.lidar_range)

        bearing_xy = np.arctan2(gathered_diffs[:, :, 1], gathered_diffs[:, :, 0])
        bearing_z = np.arctan2(
            gathered_diffs[:, :, 2],
            np.linalg.norm(gathered_diffs[:, :, :2], axis=-1) + 1e-6,
        )

        pad_count = k - valid_neighbors
        if pad_count > 0:
            # Fewer than k real neighbors exist (small swarm / near boundary
            # of the sorted list) -> pad with "nothing sensed" convention:
            # max lidar range, zero bearing. Matches Day 1's loop behavior.
            pad_dist = np.full((n, pad_count), c.lidar_range, dtype=np.float32)
            pad_bearing = np.zeros((n, pad_count), dtype=np.float32)
            gathered_dists = np.concatenate([gathered_dists, pad_dist], axis=1)
            bearing_xy = np.concatenate([bearing_xy, pad_bearing], axis=1)
            bearing_z = np.concatenate([bearing_z, pad_bearing], axis=1)

        # Interleave to [d0, bxy0, bz0, d1, bxy1, bz1, ...] — same layout
        # the reward/training code and the Day 1 loop version both expect.
        neighbor_feats = np.stack([gathered_dists, bearing_xy, bearing_z], axis=-1)  # (n, k, 3)
        neighbor_feats = neighbor_feats.reshape(n, k * 3).astype(np.float32)

        # ---- Nearest obstacle distance ----
        if len(self.obstacles) > 0:
            obs_diffs = pos[:, None, :] - self.obstacles[None, :, :]     # (n, num_obstacles, 3)
            obs_dists = np.linalg.norm(obs_diffs, axis=-1)               # (n, num_obstacles)
            nearest_obstacle = np.minimum(obs_dists.min(axis=1), c.lidar_range)
        else:
            nearest_obstacle = np.full(n, c.lidar_range, dtype=np.float32)

        # ---- Local coverage density ----
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
        ).astype(np.float32)  # (n, obs_dim)

        return {agent: obs_matrix[i] for i, agent in enumerate(self.agents)}

    def render(self):
        if self.render_mode != "human":
            return

        # Lazy import: matplotlib is a debug-only dependency, not needed
        # for training/serving, so we don't pull it in unless someone
        # actually asks for the human-viewable render mode.
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

        plt.pause(0.001)  # yields to the GUI event loop so the window actually updates

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None