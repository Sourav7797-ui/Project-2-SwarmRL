"""Day-9 tests: the policy-facing observation scaling and action mapping.
Pure numpy: these need neither Ray nor torch."""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.policy_io import (
    denormalize_action,
    normalize_action,
    observation_indices,
    observation_scale,
    scale_observation,
)
from backend.env.swarm_env import SwarmEnv
from backend.reward.curriculum import STAGE_DYNAMIC, STAGE_STATIC, STAGE_WIND


def _bounds(config):
    env = SwarmEnv(config)
    space = env.action_space("drone_0")
    return space.low, space.high


# ------------------------------ observation scale ------------------------------
@pytest.mark.parametrize("k", [1, 3, 6, 10])
def test_scale_vector_matches_the_observation_dimension(k):
    cfg = SwarmEnvConfig(num_agents=4, k_nearest=k)
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=0)
    scale = observation_scale(cfg)
    assert scale.shape == obs["drone_0"].shape == (3 * k + 8,)
    assert scale.dtype == np.float32


def test_observation_indices_cover_the_tail_exactly():
    k = 6
    idx = observation_indices(k)
    assert sorted(idx.values()) == list(range(3 * k, 3 * k + 8))     # contiguous, no gaps/overlap


def test_scale_values_follow_the_documented_layout():
    cfg = SwarmEnvConfig(k_nearest=2)
    s = observation_scale(cfg)
    L, pi = cfg.lidar_range, np.pi
    np.testing.assert_allclose(s[:6], [1 / L, 1 / pi, 1 / pi, 1 / L, 1 / pi, 1 / pi], rtol=1e-6)
    idx = observation_indices(2)
    assert s[idx["nearest_obstacle_dist"]] == pytest.approx(1 / L)
    assert s[idx["coverage_ahead"]] == 1.0
    assert s[idx["battery"]] == 1.0
    assert s[idx["wall_dist"]] == pytest.approx(1 / L)
    assert s[idx["wall_bearing"]] == pytest.approx(1 / pi)
    assert s[idx["altitude"]] == 1.0
    assert s[idx["pitch"]] == pytest.approx(1 / cfg.max_pitch_angle)
    assert s[idx["obstacle_bearing"]] == pytest.approx(1 / pi)


def test_scale_adapts_to_the_config():
    a = observation_scale(SwarmEnvConfig(lidar_range=30.0))
    b = observation_scale(SwarmEnvConfig(lidar_range=60.0))
    assert b[0] == pytest.approx(a[0] / 2)                          # distances scale with lidar range


def test_scaling_is_invertible_so_no_information_is_lost():
    cfg = SwarmEnvConfig(num_agents=5)
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=1)
    scale = observation_scale(cfg)
    raw = obs["drone_0"]
    np.testing.assert_allclose(scale_observation(raw, scale) / scale, raw, rtol=1e-5, atol=1e-6)


def test_scale_observation_accepts_a_batch_and_returns_float32():
    cfg = SwarmEnvConfig(num_agents=5)
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=1)
    batch = np.stack([obs[a] for a in env.agents])
    out = scale_observation(batch, observation_scale(cfg))
    assert out.shape == batch.shape and out.dtype == np.float32


@pytest.mark.parametrize("stage", [None, STAGE_STATIC, STAGE_DYNAMIC, STAGE_WIND],
                         ids=["default", "static", "dynamic", "wind"])
def test_scaled_observations_stay_within_minus_one_to_one_in_real_rollouts(stage):
    """The property the whole scaler exists for, checked on actual trajectories in
    every curriculum stage, not on hand-picked values."""
    cfg = SwarmEnvConfig(num_agents=20, max_episode_steps=300)
    env = SwarmEnv(cfg)
    if stage is not None:
        env.set_stage(stage)
    obs, _ = env.reset(seed=3)
    scale = observation_scale(env.config)
    rng = np.random.default_rng(3)

    worst = np.zeros(scale.shape[0])
    while env.agents:
        actions = {a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                   for a in env.agents}
        obs, *_ = env.step(actions)
        scaled = scale_observation(np.stack([obs[a] for a in obs]), scale)
        assert np.isfinite(scaled).all()
        worst = np.maximum(worst, np.abs(scaled).max(axis=0))

    assert worst.max() <= 1.0 + 1e-4, f"feature {int(worst.argmax())} reached {worst.max():.4f}"


def test_scaled_features_actually_use_their_range():
    """Guard against a scale that is far too small, which would make a feature
    invisible to the network: the distance features must reach well past 0.5."""
    cfg = SwarmEnvConfig(num_agents=30, max_episode_steps=200)
    env = SwarmEnv(cfg)
    obs, _ = env.reset(seed=0)
    scale = observation_scale(cfg)
    seen = []
    rng = np.random.default_rng(0)
    while env.agents:
        actions = {a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                   for a in env.agents}
        obs, *_ = env.step(actions)
        seen.append(scale_observation(np.stack([obs[a] for a in obs]), scale))
    seen = np.concatenate(seen)
    assert seen[:, 0].max() > 0.9                                   # nearest-neighbour distance
    assert np.abs(seen[:, 1]).max() > 0.5                           # neighbour bearing
    assert seen[:, observation_indices(6)["wall_dist"]].max() > 0.9


# ------------------------------ action mapping ------------------------------
def test_minus_one_plus_one_and_zero_map_to_low_high_and_midpoint():
    low, high = _bounds(SwarmEnvConfig(num_agents=1))
    np.testing.assert_allclose(denormalize_action(-np.ones(3), low, high), low, atol=1e-6)
    np.testing.assert_allclose(denormalize_action(np.ones(3), low, high), high, atol=1e-6)
    np.testing.assert_allclose(denormalize_action(np.zeros(3), low, high), (low + high) / 2, atol=1e-6)


def test_zero_action_means_mid_speed_and_no_turning():
    """A freshly initialised Gaussian policy outputs ~0: that must be a sensible
    action (half speed, straight, level), not 'hover'."""
    cfg = SwarmEnvConfig(num_agents=1)
    low, high = _bounds(cfg)
    velocity, pitch_rate, yaw_rate = denormalize_action(np.zeros(3), low, high)
    assert velocity == pytest.approx(cfg.max_velocity / 2)
    assert pitch_rate == pytest.approx(0.0, abs=1e-6) and yaw_rate == pytest.approx(0.0, abs=1e-6)


def test_out_of_range_policy_actions_are_clipped_to_the_physical_limits():
    low, high = _bounds(SwarmEnvConfig(num_agents=1))
    rng = np.random.default_rng(0)
    wild = rng.normal(0.0, 5.0, size=(1000, 3))                    # a badly-behaved Gaussian policy
    mapped = denormalize_action(wild, low, high)
    assert (mapped >= low - 1e-6).all() and (mapped <= high + 1e-6).all()
    assert mapped[:, 0].min() >= 0.0                                # never a negative velocity


def test_action_mapping_round_trips_inside_the_range():
    low, high = _bounds(SwarmEnvConfig(num_agents=1))
    rng = np.random.default_rng(1)
    unit = rng.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
    back = normalize_action(denormalize_action(unit, low, high), low, high)
    np.testing.assert_allclose(back, unit, atol=1e-5)


def test_mapped_actions_are_float32_and_batchable():
    low, high = _bounds(SwarmEnvConfig(num_agents=1))
    out = denormalize_action(np.zeros((7, 3)), low, high)
    assert out.shape == (7, 3) and out.dtype == np.float32


def test_mapping_respects_a_custom_config():
    cfg = SwarmEnvConfig(num_agents=1, max_velocity=10.0, max_yaw_rate=2.0, max_pitch_rate=0.5)
    low, high = _bounds(cfg)
    np.testing.assert_allclose(denormalize_action(np.ones(3), low, high), [10.0, 0.5, 2.0], atol=1e-6)
    np.testing.assert_allclose(denormalize_action(-np.ones(3), low, high), [0.0, -0.5, -2.0], atol=1e-6)