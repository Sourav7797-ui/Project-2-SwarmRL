"""Day-5 tests: reward weights are now config-driven (not module
constants), compute_with_breakdown() must sum to the same total compute()
returns, and RewardComponentLogger must accumulate correctly across an
episode."""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from backend.reward.reward_shaping import RewardShaper, ALL_COMPONENTS
from backend.reward.reward_logger import RewardComponentLogger


# ---------------------------------------------------------------------
# Config-driven weights
# ---------------------------------------------------------------------
def test_exploration_reward_weight_is_configurable():
    """Doubling exploration_reward in config should exactly double the
    exploration component, checked via the breakdown so other components
    (like the team bonus, which has its own separate scale) can't blur it."""
    config_default = SwarmEnvConfig(num_agents=3, num_obstacles=0)
    config_doubled = SwarmEnvConfig(num_agents=3, num_obstacles=0, exploration_reward=2.0)

    newly_covered = np.array([True, True, False])
    collisions = np.array([False, False, False])
    positions = np.array([[100.0, 100.0, 20.0]] * 3, dtype=np.float32)  # world center, no boundary penalty

    _, breakdown_default = RewardShaper(config_default).compute_with_breakdown(
        newly_covered_by_agent=newly_covered, collisions_by_agent=collisions,
        positions=positions, world_size_xy=config_default.world_size_xy,
        world_height=config_default.world_height,
        obstacle_collisions_by_agent=np.zeros(3, dtype=bool),
    )
    _, breakdown_doubled = RewardShaper(config_doubled).compute_with_breakdown(
        newly_covered_by_agent=newly_covered, collisions_by_agent=collisions,
        positions=positions, world_size_xy=config_doubled.world_size_xy,
        world_height=config_doubled.world_height,
        obstacle_collisions_by_agent=np.zeros(3, dtype=bool),
    )

    np.testing.assert_allclose(
        breakdown_doubled["exploration"], breakdown_default["exploration"] * 2.0
    )


def test_collision_penalty_weight_is_configurable():
    config = SwarmEnvConfig(num_agents=2, num_obstacles=0, collision_penalty=-50.0)
    shaper = RewardShaper(config)

    newly_covered = np.array([False, False])
    collisions = np.array([True, True])
    positions = np.array([[100.0, 100.0, 20.0]] * 2, dtype=np.float32)

    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=newly_covered, collisions_by_agent=collisions,
        positions=positions, world_size_xy=config.world_size_xy,
        world_height=config.world_height,
        obstacle_collisions_by_agent=np.zeros(2, dtype=bool),
    )
    np.testing.assert_allclose(breakdown["drone_collision"], [-50.0, -50.0])


def test_obstacle_collision_penalty_weight_is_configurable():
    config = SwarmEnvConfig(num_agents=1, num_obstacles=0, obstacle_collision_penalty=-20.0)
    shaper = RewardShaper(config)

    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=np.array([False]),
        collisions_by_agent=np.array([False]),
        positions=np.array([[100.0, 100.0, 20.0]], dtype=np.float32),
        world_size_xy=config.world_size_xy, world_height=config.world_height,
        obstacle_collisions_by_agent=np.array([True]),
    )
    assert breakdown["obstacle_collision"][0] == pytest.approx(-20.0)


def test_env_wires_custom_reward_weights_end_to_end():
    """A full SwarmEnv built with a custom config should hand those
    weights to its RewardShaper, not just the standalone class."""
    config = SwarmEnvConfig(
        num_agents=1, num_obstacles=0, max_episode_steps=5,
        collision_penalty=-999.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=0)
    assert env.reward_shaper.config.collision_penalty == -999.0


# ---------------------------------------------------------------------
# Breakdown correctness
# ---------------------------------------------------------------------
def test_breakdown_components_sum_to_total():
    """compute_with_breakdown()'s five components must sum (per agent)
    to exactly what compute() returns as the total."""
    config = SwarmEnvConfig(num_agents=6, num_obstacles=3)
    shaper = RewardShaper(config)

    rng = np.random.default_rng(1)
    newly_covered = rng.random(6) > 0.5
    collisions = rng.random(6) > 0.8
    obstacle_collisions = rng.random(6) > 0.9
    positions = rng.uniform(0, config.world_size_xy, size=(6, 3)).astype(np.float32)
    positions[:, 2] = rng.uniform(0, config.world_height, size=6)

    total_rewards = shaper.compute(
        newly_covered_by_agent=newly_covered, collisions_by_agent=collisions,
        positions=positions, world_size_xy=config.world_size_xy,
        world_height=config.world_height,
        obstacle_collisions_by_agent=obstacle_collisions,
    )
    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=newly_covered, collisions_by_agent=collisions,
        positions=positions, world_size_xy=config.world_size_xy,
        world_height=config.world_height,
        obstacle_collisions_by_agent=obstacle_collisions,
    )

    reconstructed = sum(breakdown[c] for c in ALL_COMPONENTS)
    for i in range(6):
        assert reconstructed[i] == pytest.approx(total_rewards[f"drone_{i}"], abs=1e-4)


def test_breakdown_has_all_expected_components():
    config = SwarmEnvConfig(num_agents=2, num_obstacles=1)
    shaper = RewardShaper(config)
    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=np.array([True, False]),
        collisions_by_agent=np.array([False, False]),
        positions=np.array([[100.0, 100.0, 20.0]] * 2, dtype=np.float32),
        world_size_xy=config.world_size_xy, world_height=config.world_height,
        obstacle_collisions_by_agent=np.array([False, False]),
    )
    assert set(breakdown.keys()) == set(ALL_COMPONENTS)
    for component, values in breakdown.items():
        assert values.shape == (2,)


# ---------------------------------------------------------------------
# RewardComponentLogger
# ---------------------------------------------------------------------
def test_logger_accumulates_across_steps():
    logger = RewardComponentLogger()
    logger.log_step({"exploration": np.array([1.0, 1.0, 0.0]), "drone_collision": np.array([0.0, 0.0, 0.0])})
    logger.log_step({"exploration": np.array([0.0, 1.0, 1.0]), "drone_collision": np.array([-100.0, 0.0, 0.0])})

    assert logger.num_steps == 2
    totals = logger.episode_totals()
    assert totals["exploration"] == pytest.approx(4.0)  # (1+1+0) + (0+1+1)
    assert totals["drone_collision"] == pytest.approx(-100.0)


def test_logger_means_per_step():
    logger = RewardComponentLogger()
    logger.log_step({"exploration": np.array([2.0, 2.0])})
    logger.log_step({"exploration": np.array([4.0, 4.0])})
    means = logger.episode_means_per_step()
    # total = 4 + 8 = 12, over 2 steps -> mean 6.0 per step
    assert means["exploration"] == pytest.approx(6.0)


def test_logger_component_trace():
    logger = RewardComponentLogger()
    logger.log_step({"exploration": np.array([1.0, 1.0])})  # sum = 2.0
    logger.log_step({"exploration": np.array([3.0, 3.0])})  # sum = 6.0
    trace = logger.component_trace("exploration")
    assert trace == pytest.approx([2.0, 6.0])


def test_logger_reset_clears_state():
    logger = RewardComponentLogger()
    logger.log_step({"exploration": np.array([5.0])})
    logger.reset()
    assert logger.num_steps == 0
    assert logger.episode_totals() == {}


def test_logger_summary_string_does_not_crash():
    logger = RewardComponentLogger()
    logger.log_step({"exploration": np.array([1.0]), "drone_collision": np.array([-100.0])})
    summary = logger.summary_string()
    assert "exploration" in summary
    assert "drone_collision" in summary
    assert "NET" in summary


def test_env_feeds_logger_automatically_during_step():
    """Attaching a logger to env.reward_logger should make step() call
    it automatically, with no other code changes needed."""
    config = SwarmEnvConfig(num_agents=5, num_obstacles=2, max_episode_steps=10)
    env = SwarmEnv(config=config)
    env.reset(seed=0)

    logger = RewardComponentLogger()
    env.reward_logger = logger

    rng = np.random.default_rng(0)
    steps_run = 0
    while env.agents:
        actions = {
            a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
            for a in env.agents
        }
        env.step(actions)
        steps_run += 1

    assert logger.num_steps == steps_run == config.max_episode_steps
    totals = logger.episode_totals()
    assert set(totals.keys()) == set(ALL_COMPONENTS)


def test_env_without_logger_attached_runs_unaffected():
    """reward_logger defaults to None; step() must not crash when
    nobody's using it."""
    config = SwarmEnvConfig(num_agents=5, num_obstacles=2, max_episode_steps=5)
    env = SwarmEnv(config=config)
    env.reset(seed=0)
    assert env.reward_logger is None

    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, _, _, _ = env.step(actions)
        assert all(np.isfinite(r) for r in rewards.values())