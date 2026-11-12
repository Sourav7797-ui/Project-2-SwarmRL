"""Day-2 tests: obstacle collision physics and the velocity/drag momentum model."""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def test_drone_pushed_out_of_obstacle():
    config = SwarmEnvConfig(num_agents=1, num_obstacles=1, max_episode_steps=5)
    env = SwarmEnv(config=config)
    env.reset(seed=0)

    env.obstacles[:] = np.array([[50.0, 50.0, 10.0]], dtype=np.float32)
    env.positions[:] = np.array([[50.0, 50.0, 10.0]], dtype=np.float32)  # dead center = max penetration

    actions = {"drone_0": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    _, rewards, _, _, infos = env.step(actions)

    dist_to_obstacle = np.linalg.norm(env.positions[0] - env.obstacles[0])
    expected_min_dist = config.drone_radius + config.obstacle_radius

    assert infos["drone_0"]["obstacle_collision"] is True
    assert dist_to_obstacle == pytest.approx(expected_min_dist, abs=1e-3)
    assert rewards["drone_0"] <= -70.0


def test_no_obstacle_collision_when_clear():
    config = SwarmEnvConfig(num_agents=1, num_obstacles=2, max_episode_steps=5)
    env = SwarmEnv(config=config)
    env.reset(seed=0)

    env.obstacles[:] = np.array([[0.0, 0.0, 0.0], [200.0, 200.0, 40.0]], dtype=np.float32)
    env.positions[:] = np.array([[100.0, 100.0, 20.0]], dtype=np.float32)

    actions = {"drone_0": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    _, rewards, _, _, infos = env.step(actions)

    assert infos["drone_0"]["obstacle_collision"] is False


def test_speed_ramps_up_under_acceleration_cap():
    config = SwarmEnvConfig(
        num_agents=1, num_obstacles=0, max_episode_steps=50,
        max_acceleration=2.0, drag_coefficient=0.0, dt=0.1,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=0)
    env.positions[:] = np.array([[100.0, 100.0, 20.0]], dtype=np.float32)

    max_command = env.action_space("drone_0").high[0]
    actions = {"drone_0": np.array([max_command, 0.0, 0.0], dtype=np.float32)}

    env.step(actions)
    speed_after_1_tick = env.speed[0]

    assert speed_after_1_tick == pytest.approx(0.2, abs=1e-3)  # max_accel * dt
    assert speed_after_1_tick < max_command  # did NOT snap to full speed


def test_speed_never_exceeds_max_velocity():
    config = SwarmEnvConfig(
        num_agents=1, num_obstacles=0, max_episode_steps=200,
        max_acceleration=50.0, drag_coefficient=0.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=0)
    env.positions[:] = np.array([[100.0, 100.0, 20.0]], dtype=np.float32)

    max_command = env.action_space("drone_0").high[0]
    actions = {"drone_0": np.array([max_command, 0.0, 0.0], dtype=np.float32)}

    for _ in range(20):
        if not env.agents:
            break
        env.step(actions)

    assert env.speed[0] <= config.max_velocity + 1e-3


def test_drag_decays_speed_with_no_command():
    config = SwarmEnvConfig(
        num_agents=1, num_obstacles=0, max_episode_steps=50,
        max_acceleration=100.0, drag_coefficient=0.1,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=0)
    env.positions[:] = np.array([[100.0, 100.0, 20.0]], dtype=np.float32)
    env.speed[:] = 5.0

    actions = {"drone_0": np.array([0.0, 0.0, 0.0], dtype=np.float32)}
    env.step(actions)

    assert env.speed[0] < 5.0
    assert env.speed[0] >= 0.0