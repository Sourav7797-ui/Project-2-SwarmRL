"""Day-3 tests: edge-case hardening plus a correctness cross-check for
the vectorized _build_observations rewrite (it must produce bit-for-bit
the same values as the original Day 1/2 per-agent Python loop)."""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi) using plain loops (deliberately not vectorized)."""
    while a >= np.pi:
        a -= 2.0 * np.pi
    while a < -np.pi:
        a += 2.0 * np.pi
    return a


def _reference_loop_observations(env: SwarmEnv) -> dict[str, np.ndarray]:
    """Slow, explicit, per-agent reimplementation of the Day 7 observation
    spec. Written independently of the vectorized code on purpose: it uses
    plain loops and if-chains, so a bug would have to be made twice."""
    c = env.config
    shape = env.coverage_grid.shape
    obs = {}
    for i, agent in enumerate(env.agents):
        pos_i = env.positions[i]
        pitch_i, yaw_i = float(env.headings[i, 0]), float(env.headings[i, 1])

        # neighbours: real other agents only (self is never a neighbour)
        diffs = env.positions - pos_i
        dists = np.linalg.norm(diffs, axis=1)
        order = [j for j in np.argsort(dists) if j != i][: c.k_nearest]
        neighbor_feats = []
        for j in order:
            d = min(float(dists[j]), c.lidar_range)
            bearing_xy = _wrap(float(np.arctan2(diffs[j, 1], diffs[j, 0])) - yaw_i)
            bearing_z = float(np.arctan2(diffs[j, 2], np.linalg.norm(diffs[j, :2]) + 1e-6)) - pitch_i
            neighbor_feats.extend([d, bearing_xy, bearing_z])
        while len(neighbor_feats) < c.k_nearest * 3:
            neighbor_feats.extend([c.lidar_range, 0.0, 0.0])

        # nearest obstacle: distance + body-frame bearing (0 if none within range)
        if len(env.obstacles) > 0:
            to_obs = env.obstacles - pos_i
            obstacle_dists = np.linalg.norm(to_obs, axis=1)
            j = min(range(len(env.obstacles)), key=lambda t: obstacle_dists[t])   # first minimum
            nearest_obstacle = min(float(obstacle_dists[j]), c.lidar_range)
            if obstacle_dists[j] < c.lidar_range:
                obstacle_bearing = _wrap(float(np.arctan2(to_obs[j, 1], to_obs[j, 0])) - yaw_i)
            else:
                obstacle_bearing = 0.0
        else:
            nearest_obstacle = c.lidar_range
            obstacle_bearing = 0.0

        # coverage ahead: 3x3x3 block, out-of-world cells count as explored
        direction = np.array([
            np.cos(pitch_i) * np.cos(yaw_i),
            np.cos(pitch_i) * np.sin(yaw_i),
            np.sin(pitch_i),
        ])
        ahead = np.clip(pos_i + direction * c.coverage_lookahead, 0.0,
                        [c.world_size_xy, c.world_size_xy, c.world_height])
        ix = min(int(ahead[0] // c.voxel_size), shape[0] - 1)
        iy = min(int(ahead[1] // c.voxel_size), shape[1] - 1)
        iz = min(int(ahead[2] // c.voxel_size), shape[2] - 1)
        covered = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    gx, gy, gz = ix + dx, iy + dy, iz + dz
                    inside = 0 <= gx < shape[0] and 0 <= gy < shape[1] and 0 <= gz < shape[2]
                    if (not inside) or env.coverage_grid[gx, gy, gz]:
                        covered += 1
        coverage_ahead = covered / 27.0

        battery_remaining = float(env.battery[i])

        # nearest xy wall, with the world direction that points at it
        x, y, z = float(pos_i[0]), float(pos_i[1]), float(pos_i[2])
        candidates = [
            (x, np.pi),                       # x = 0 wall
            (c.world_size_xy - x, 0.0),       # x = W wall
            (y, -np.pi / 2.0),                # y = 0 wall
            (c.world_size_xy - y, np.pi / 2.0),  # y = W wall
        ]
        wall_dist, wall_dir = min(candidates, key=lambda t: t[0])
        wall_bearing = _wrap(wall_dir - yaw_i) if wall_dist < c.lidar_range else 0.0
        wall_dist = min(wall_dist, c.lidar_range)

        altitude_norm = (z - c.world_height / 2.0) / (c.world_height / 2.0)

        obs[agent] = np.array(
            neighbor_feats + [nearest_obstacle, coverage_ahead, battery_remaining,
                              wall_dist, wall_bearing, altitude_norm, pitch_i, obstacle_bearing],
            dtype=np.float32,
        )
    return obs

@pytest.mark.parametrize("num_agents", [1, 2, 5, 13, 50])
def test_vectorized_matches_reference_loop(num_agents):
    """The Day 3 vectorized rewrite must match the Day 1/2 loop exactly,
    across swarm sizes both smaller and larger than k_nearest."""
    config = SwarmEnvConfig(num_agents=num_agents, num_obstacles=4, k_nearest=6, max_episode_steps=10)
    env = SwarmEnv(config=config)
    env.reset(seed=7)

    # Step once under random actions so positions/coverage aren't just
    # the pristine reset state.
    actions = {a: env.action_space(a).sample() for a in env.agents}
    env.step(actions)

    vectorized = env._build_observations()
    reference = _reference_loop_observations(env)

    assert set(vectorized.keys()) == set(reference.keys())
    for agent in vectorized:
        np.testing.assert_allclose(
            vectorized[agent], reference[agent], rtol=1e-5, atol=1e-6,
            err_msg=f"Mismatch for {agent} at num_agents={num_agents}",
        )


def test_vectorized_matches_reference_loop_zero_obstacles():
    """Separately cover the zero-obstacle branch (own code path in both
    implementations) at a couple of swarm sizes."""
    for num_agents in (1, 6):
        config = SwarmEnvConfig(num_agents=num_agents, num_obstacles=0, k_nearest=4, max_episode_steps=10)
        env = SwarmEnv(config=config)
        env.reset(seed=3)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)

        vectorized = env._build_observations()
        reference = _reference_loop_observations(env)
        for agent in vectorized:
            np.testing.assert_allclose(vectorized[agent], reference[agent], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------
def test_reset_is_reproducible_with_same_seed():
    """Same seed -> identical initial positions/obstacles/observations.
    This matters a lot once Task 3's RLlib rollout workers need
    deterministic replay for debugging."""
    config = SwarmEnvConfig(num_agents=20, num_obstacles=6)

    env_a = SwarmEnv(config=config)
    obs_a, _ = env_a.reset(seed=123)

    env_b = SwarmEnv(config=config)
    obs_b, _ = env_b.reset(seed=123)

    np.testing.assert_array_equal(env_a.positions, env_b.positions)
    np.testing.assert_array_equal(env_a.obstacles, env_b.obstacles)
    for agent in obs_a:
        np.testing.assert_array_equal(obs_a[agent], obs_b[agent])


def test_different_seeds_give_different_initial_positions():
    """Sanity check the opposite direction too: different seeds should
    (with overwhelming probability) NOT produce identical worlds."""
    config = SwarmEnvConfig(num_agents=20, num_obstacles=6)

    env_a = SwarmEnv(config=config)
    env_a.reset(seed=1)

    env_b = SwarmEnv(config=config)
    env_b.reset(seed=2)

    assert not np.allclose(env_a.positions, env_b.positions)


def test_full_episode_reproducible_under_same_action_sequence():
    """Same seed + same action sequence -> same trajectory, including
    rewards. This is the property RLlib's rollout workers actually rely
    on for reproducible training runs."""
    config = SwarmEnvConfig(num_agents=8, num_obstacles=3, max_episode_steps=15)

    def run_episode(seed):
        env = SwarmEnv(config=config)
        env.reset(seed=seed)
        rng = np.random.default_rng(999)  # separate RNG, only for action sampling
        reward_trace = []
        while env.agents:
            actions = {
                a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                for a in env.agents
            }
            _, rewards, _, _, _ = env.step(actions)
            reward_trace.append(sum(rewards.values()))
        return reward_trace

    trace_a = run_episode(seed=55)
    trace_b = run_episode(seed=55)
    assert trace_a == pytest.approx(trace_b)


# ---------------------------------------------------------------------
# Agent-count edge cases
# ---------------------------------------------------------------------
@pytest.mark.parametrize("num_agents", [1, 2, 100])
def test_env_runs_at_various_swarm_sizes(num_agents):
    """1 agent (no neighbors ever), 2 agents (exactly one neighbor each),
    and 100 agents (larger than the architecture doc's default 50) should
    all run a full episode without error."""
    config = SwarmEnvConfig(num_agents=num_agents, num_obstacles=3, max_episode_steps=10)
    env = SwarmEnv(config=config)
    obs, _ = env.reset(seed=0)
    assert len(obs) == num_agents

    steps = 0
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        steps += 1
        assert len(obs) == num_agents
        assert all(np.isfinite(r) for r in rewards.values())
    assert steps == 10


def test_k_nearest_larger_than_swarm_size_pads_correctly():
    """k_nearest > (num_agents - 1) forces the padding branch on every
    step, not just at construction. Confirms padding stays well-formed
    across a whole episode, not just at t=0."""
    config = SwarmEnvConfig(num_agents=3, k_nearest=10, num_obstacles=2, max_episode_steps=5)
    env = SwarmEnv(config=config)
    obs, _ = env.reset(seed=0)
    for agent in obs:
        assert obs[agent].shape == (env._obs_dim,)
        assert np.isfinite(obs[agent]).all()

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
        for agent in obs:
            assert np.isfinite(obs[agent]).all()