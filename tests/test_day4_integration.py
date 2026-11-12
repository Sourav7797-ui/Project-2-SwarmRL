"""Day-4 integration test: the final health check for Task 1 before the
env's API is frozen for Task 2 (reward tuning) and Task 3 (RLlib wrapper)
to build against."""
from backend.env import config
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from tests.test_swarm_env_smoke import env


def test_full_scale_episode_is_numerically_clean():
    """50 agents, 1000 steps, default config — the actual target scale
    from the architecture doc."""
    config = SwarmEnvConfig()  # defaults: num_agents=50, max_episode_steps=1000
    env = SwarmEnv(config=config)
    obs, _ = env.reset(seed=0)

    for agent, o in obs.items():
        assert np.isfinite(o).all(), f"Non-finite value in initial obs for {agent}"

    coverage_trace = [env.coverage_grid.mean()]
    rng = np.random.default_rng(0)
    steps = 0

    while env.agents:
        actions = {
            a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
            for a in env.agents
        }
        obs, rewards, terms, truncs, infos = env.step(actions)
        steps += 1

        assert np.isfinite(env.positions).all(), f"Non-finite position at step {steps}"
        assert np.isfinite(env.speed).all(), f"Non-finite speed at step {steps}"
        for agent, o in obs.items():
            assert np.isfinite(o).all(), f"Non-finite obs for {agent} at step {steps}"
        for agent, r in rewards.items():
            assert np.isfinite(r), f"Non-finite reward for {agent} at step {steps}"

        coverage_trace.append(env.coverage_grid.mean())

    assert steps == config.max_episode_steps
    assert all(b >= a - 1e-9 for a, b in zip(coverage_trace, coverage_trace[1:]))
    assert coverage_trace[-1] > coverage_trace[0], "No exploration happened over 1000 steps"


def test_env_is_reusable_across_multiple_episodes():
    """reset() after a completed episode should produce a clean new
    episode with no leaked state from the previous one."""
    config = SwarmEnvConfig(num_agents=10, max_episode_steps=20)
    env = SwarmEnv(config=config)

    def run_one_episode(seed):
        env.reset(seed=seed)
        while env.agents:
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)

    run_one_episode(seed=1)
    assert env.coverage_grid.mean() > 0

    obs, _ = env.reset(seed=2)
    assert env._step_count == 0
    assert np.all(env.speed == 0.0)
    assert env.coverage_grid.sum() > 0
    assert env.coverage_grid.sum() <= np.prod(config.voxel_grid_shape)
    assert len(env.agents) == config.num_agents
    for agent, o in obs.items():
        assert np.isfinite(o).all()


def test_observation_and_action_space_contracts_are_stable():
    """Freezes the shape/dtype/bounds contract that Task 3's RLlib
    wrapper and Task 4's PyTorch networks will be built against."""
    config = SwarmEnvConfig(num_agents=50, k_nearest=6)
    env = SwarmEnv(config=config)
    env.reset(seed=0)

    agent = env.agents[0]
    obs_space = env.observation_space(agent)
    act_space = env.action_space(agent)

    expected_obs_dim = config.k_nearest * 3 + 8
    assert obs_space.shape == (expected_obs_dim,)
    assert obs_space.dtype == np.float32

    assert act_space.shape == (3,)
    assert act_space.dtype == np.float32
    assert act_space.low[0] == 0.0
    assert act_space.high[0] == config.max_velocity