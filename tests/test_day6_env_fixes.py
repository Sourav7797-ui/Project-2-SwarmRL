"""Day-6 tests for the two env fixes the reward analysis uncovered:

  1. pitch is bounded (it used to accumulate without limit, so drones flew
     upside-down and got pinned against the floor/ceiling)
  2. collisions are charged per EVENT (the tick contact begins), not per
     tick of overlap (a persistent overlap used to re-bill -100 every tick,
     up to 169 times in a row for a single agent)
"""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from backend.reward.reward_logger import RewardComponentLogger


def _step_idle(env, n_agents):
    """Step with every agent commanding zero velocity and zero turn."""
    actions = {f"drone_{i}": np.zeros(3, dtype=np.float32) for i in range(n_agents)}
    return env.step(actions)


# ------------------------------- pitch bound -------------------------------
def test_pitch_is_clamped_at_the_positive_limit():
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=0, max_episode_steps=300)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.positions[:] = [[100.0, 100.0, 20.0]]

    pitch_up = np.array([0.0, cfg.max_pitch_rate, 0.0], dtype=np.float32)
    for _ in range(100):  # 100 ticks * 1 rad/s * 0.1 s = 10 rad requested, far past the limit
        env.step({"drone_0": pitch_up})
        env.positions[:] = [[100.0, 100.0, 20.0]]  # keep it away from walls; we only test heading
        assert env.headings[0, 0] <= cfg.max_pitch_angle + 1e-6

    assert env.headings[0, 0] == pytest.approx(cfg.max_pitch_angle, abs=1e-5)


def test_pitch_is_clamped_at_the_negative_limit():
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=0, max_episode_steps=300)
    env = SwarmEnv(cfg)
    env.reset(seed=0)

    pitch_down = np.array([0.0, -cfg.max_pitch_rate, 0.0], dtype=np.float32)
    for _ in range(100):
        env.step({"drone_0": pitch_down})
        env.positions[:] = [[100.0, 100.0, 20.0]]
        assert env.headings[0, 0] >= -cfg.max_pitch_angle - 1e-6

    assert env.headings[0, 0] == pytest.approx(-cfg.max_pitch_angle, abs=1e-5)


def test_drones_never_fly_upside_down_under_random_actions():
    """The original symptom: random pitch commands drifted past +-90 degrees."""
    cfg = SwarmEnvConfig(num_agents=10, num_obstacles=0, max_episode_steps=500)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    while env.agents:
        actions = {
            a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
            for a in env.agents
        }
        env.step(actions)
        assert np.abs(env.headings[:, 0]).max() <= np.pi / 2


# ------------------------- drone collisions: per event -------------------------
def _two_overlapping_drones():
    cfg = SwarmEnvConfig(num_agents=2, num_obstacles=0, max_episode_steps=50)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.positions[:] = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]  # 0.5 m apart < min_separation
    return env


def test_persistent_overlap_is_charged_once_not_every_tick():
    env = _two_overlapping_drones()
    logger = RewardComponentLogger()
    env.reward_logger = logger

    states, events = [], []
    for _ in range(6):
        _, _, _, _, infos = _step_idle(env, 2)
        states.append(infos["drone_0"]["collision"])
        events.append(infos["drone_0"]["collision_event"])

    assert states == [True] * 6                      # still overlapping every tick
    assert events == [True] + [False] * 5            # but only the first tick is an event
    # Two agents, each charged exactly once: 2 * -100. Per-tick billing would be -1200.
    assert logger.episode_totals()["drone_collision"] == pytest.approx(-200.0)


def test_recontact_after_separating_is_charged_again():
    env = _two_overlapping_drones()

    _, _, _, _, infos = _step_idle(env, 2)
    assert infos["drone_0"]["collision_event"] is True           # first contact

    # Pull them apart for a full cooldown (Day 8: a brief separation is the same contact).
    env.positions[:] = [[100.0, 100.0, 20.0], [150.0, 100.0, 20.0]]
    for _ in range(env.config.contact_cooldown_ticks):
        _, _, _, _, infos = _step_idle(env, 2)
        assert infos["drone_0"]["collision"] is False
        assert infos["drone_0"]["collision_event"] is False

    env.positions[:] = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]   # collide a second time
    _, _, _, _, infos = _step_idle(env, 2)
    assert infos["drone_0"]["collision_event"] is True           # a NEW event is charged


def test_collision_flags_do_not_leak_across_reset():
    """A collision still 'active' at the end of one episode must not suppress
    the first collision of the next episode."""
    env = _two_overlapping_drones()
    _step_idle(env, 2)                                  # leaves the collision flags set

    env.reset(seed=1)
    env.positions[:] = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]
    _, rewards, _, _, infos = _step_idle(env, 2)
    assert infos["drone_0"]["collision_event"] is True
    assert rewards["drone_0"] <= -95.0


# ------------------------ obstacle collisions: per event ------------------------
def test_pressing_into_an_obstacle_is_charged_once_not_every_tick():
    """A drone pointed at an obstacle re-penetrates every tick (the push-out
    zeroes its speed, then it re-accelerates inward). That is one contact,
    not six."""
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=1, max_episode_steps=50)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.obstacles[:] = [[50.0, 50.0, 10.0]]
    # 3.49 m from the centre (collision radius is 3.5), heading +x = straight at the obstacle
    env.headings[:] = 0.0  # explicit: face +x. (Resets now spawn random headings.)
    env.positions[:] = [[50.0 - 3.49, 50.0, 10.0]]

    logger = RewardComponentLogger()
    env.reward_logger = logger
    drive_in = np.array([cfg.max_velocity, 0.0, 0.0], dtype=np.float32)

    states, events = [], []
    for _ in range(6):
        _, _, _, _, infos = env.step({"drone_0": drive_in})
        states.append(infos["drone_0"]["obstacle_collision"])
        events.append(infos["drone_0"]["obstacle_collision_event"])

    assert states == [True] * 6
    assert events == [True] + [False] * 5
    assert logger.episode_totals()["obstacle_collision"] == pytest.approx(cfg.obstacle_collision_penalty)


def test_info_keys_for_events_are_present_every_step():
    cfg = SwarmEnvConfig(num_agents=3, num_obstacles=2, max_episode_steps=5)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    _, _, _, _, infos = _step_idle(env, 3)
    for info in infos.values():
        for key in ("collision", "collision_event", "obstacle_collision",
                    "obstacle_collision_event", "battery", "speed"):
            assert key in info