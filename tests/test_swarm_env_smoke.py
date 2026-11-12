import numpy as np
import pytest
from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


@pytest.fixture
def env():
    return SwarmEnv(config=SwarmEnvConfig(num_agents=10, max_episode_steps=20))


def test_reset_shapes(env):
    obs, infos = env.reset(seed=0)
    assert len(obs) == 10
    for agent in env.agents:
        assert obs[agent].shape == (env._obs_dim,)
        assert np.isfinite(obs[agent]).all()


def test_step_runs_full_episode(env):
    env.reset(seed=0)
    total_steps = 0
    coverage_over_time = []
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        total_steps += 1
        coverage_over_time.append(env.coverage_grid.mean())
        assert set(rewards.keys()) == set(obs.keys())
        for r in rewards.values():
            assert np.isfinite(r)
    assert total_steps == 20
    assert all(b >= a - 1e-9 for a, b in zip(coverage_over_time, coverage_over_time[1:]))


def test_collision_triggers_penalty():
    config = SwarmEnvConfig(num_agents=2, max_episode_steps=5, min_separation=50.0)
    env = SwarmEnv(config=config)
    env.reset(seed=1)
    env.positions[:] = np.array([[10.0, 10.0, 10.0], [10.5, 10.0, 10.0]], dtype=np.float32)
    actions = {agent: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent in env.agents}
    _, rewards, _, _, infos = env.step(actions)
    assert any(info["collision"] for info in infos.values())
    assert min(rewards.values()) <= -95.0