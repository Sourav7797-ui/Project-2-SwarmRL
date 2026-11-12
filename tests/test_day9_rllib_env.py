"""Day-9 tests: the RLlib adapter around SwarmEnv.

These exercise the adapter as a plain Python object (no Ray cluster is started),
so they are fast, but they do need `ray` importable for the MultiAgentEnv base
class. They are skipped automatically if Ray is not installed."""
import numpy as np
import pytest

pytest.importorskip("ray")
from ray.rllib.utils.pre_checks.env import check_multiagent_environments  # noqa: E402

from backend.env.config import SwarmEnvConfig  # noqa: E402
from backend.env.policy_io import denormalize_action, observation_scale, scale_observation  # noqa: E402
from backend.env.swarm_env import SwarmEnv  # noqa: E402
from backend.reward.curriculum import STAGE_DYNAMIC, STAGE_STATIC, STAGE_WIND  # noqa: E402
from backend.training.rllib_env import (  # noqa: E402
    ENV_NAME,
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
    env_creator,
    register_swarm_env,
)

SMALL = dict(num_agents=4, max_episode_steps=10, num_obstacles=3)


def _unit_actions(env, rng):
    return {a: rng.uniform(-1.0, 1.0, size=3).astype(np.float32) for a in env.agents}


# ------------------------------- RLlib's own checker -------------------------------
def test_passes_rllibs_own_multi_agent_env_checker():
    check_multiagent_environments(SwarmRLlibEnv(SMALL))     # raises if the contract is broken


# ------------------------------- spaces and agents -------------------------------
def test_per_agent_spaces():
    env = SwarmRLlibEnv(SMALL)
    k = SwarmEnvConfig().k_nearest
    for aid in env.possible_agents:
        obs_space, act_space = env.get_observation_space(aid), env.get_action_space(aid)
        assert obs_space.shape == (3 * k + 8,) and obs_space.dtype == np.float32
        assert act_space.shape == (3,) and act_space.dtype == np.float32
        np.testing.assert_array_equal(act_space.low, -1.0)       # the policy acts in [-1, 1]
        np.testing.assert_array_equal(act_space.high, 1.0)


def test_dict_spaces_cover_every_agent():
    env = SwarmRLlibEnv(SMALL)
    assert set(env.observation_space.spaces) == set(env.possible_agents)
    assert set(env.action_space.spaces) == set(env.possible_agents)


def test_agent_bookkeeping_is_consistent_and_stays_constant():
    env = SwarmRLlibEnv(SMALL)
    assert env.possible_agents == [f"drone_{i}" for i in range(4)]
    assert env.agents == env.possible_agents
    assert env._agent_ids == set(env.possible_agents)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(_unit_actions(env.env, rng))
    assert env.agents == env.possible_agents                  # not emptied when the episode ends


# ------------------------------- reset / step contract -------------------------------
def test_reset_returns_scaled_float32_observations_for_every_agent():
    env = SwarmRLlibEnv(SMALL)
    obs, infos = env.reset(seed=1)
    assert set(obs) == set(env.possible_agents) and isinstance(infos, dict)
    for o in obs.values():
        assert o.dtype == np.float32 and np.isfinite(o).all()
        assert np.abs(o).max() <= 1.0 + 1e-4                   # scaled


def test_step_returns_the_five_rllib_dicts():
    env = SwarmRLlibEnv(SMALL)
    env.reset(seed=1)
    out = env.step(_unit_actions(env.env, np.random.default_rng(0)))
    obs, rewards, terminateds, truncateds, infos = out
    agents = set(env.possible_agents)
    assert set(obs) == agents and set(rewards) == agents and set(infos) == agents
    assert set(terminateds) == agents | {"__all__"} and set(truncateds) == agents | {"__all__"}
    assert all(isinstance(r, float) for r in rewards.values())


def test_all_flag_is_false_until_the_final_step_then_truncated():
    env = SwarmRLlibEnv(dict(SMALL, max_episode_steps=5))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    flags = []
    for _ in range(5):
        _, _, term, trunc, _ = env.step(_unit_actions(env.env, rng))
        flags.append((term["__all__"], trunc["__all__"]))
    assert flags == [(False, False)] * 4 + [(False, True)]      # time-limit truncation, never termination


def test_step_rejects_actions_that_do_not_cover_the_live_agents():
    env = SwarmRLlibEnv(SMALL)
    env.reset(seed=0)
    actions = _unit_actions(env.env, np.random.default_rng(0))
    del actions["drone_0"]
    with pytest.raises(ValueError, match="live agents"):
        env.step(actions)


# ------------------------------- the policy-facing interface -------------------------------
def test_wrapper_matches_the_raw_env_driven_with_equivalent_physical_actions():
    """End to end: policy-space actions through the adapter must give exactly the
    trajectory the raw env gives for the mapped physical actions, with observations
    equal to the scaled raw ones. Catches any mapping or scaling mistake."""
    cfg = SwarmEnvConfig(**SMALL)
    wrapped = SwarmRLlibEnv(SMALL)
    raw = SwarmEnv(cfg)
    w_obs, _ = wrapped.reset(seed=5)
    r_obs, _ = raw.reset(seed=5)
    scale = observation_scale(cfg)
    low, high = raw.action_space("drone_0").low, raw.action_space("drone_0").high
    rng = np.random.default_rng(5)

    for _ in range(10):
        unit = _unit_actions(raw, rng)
        w_obs, w_rew, *_ = wrapped.step(unit)
        r_obs, r_rew, *_ = raw.step({a: denormalize_action(u, low, high) for a, u in unit.items()})
        for a in w_obs:
            np.testing.assert_allclose(w_obs[a], scale_observation(r_obs[a], scale), atol=1e-6)
            assert w_rew[a] == pytest.approx(r_rew[a])
    np.testing.assert_allclose(wrapped.env.positions, raw.positions, atol=1e-5)


def test_a_zero_action_flies_at_roughly_half_speed_not_hovering():
    env = SwarmRLlibEnv(dict(num_agents=1, num_obstacles=0, max_episode_steps=500))
    env.reset(seed=0)
    for _ in range(150):
        env.step({"drone_0": np.zeros(3, dtype=np.float32)})
    speed = float(env.env.speed[0])
    assert 3.0 < speed <= env.env.config.max_velocity / 2 + 1e-3


def test_out_of_range_policy_actions_cannot_break_the_physical_limits():
    env = SwarmRLlibEnv(dict(num_agents=3, num_obstacles=0, max_episode_steps=200))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.step({a: rng.normal(0.0, 10.0, size=3).astype(np.float32) for a in env.env.agents})
        assert env.env.speed.max() <= env.env.config.max_velocity + 1e-4
        assert np.abs(env.env.headings[:, 0]).max() <= env.env.config.max_pitch_angle + 1e-6


def test_wrapper_is_deterministic_for_a_seed():
    def run():
        env = SwarmRLlibEnv(SMALL)
        obs, _ = env.reset(seed=9)
        rng = np.random.default_rng(9)
        trace = [obs["drone_0"].copy()]
        for _ in range(10):
            obs, rew, *_ = env.step(_unit_actions(env.env, rng))
            trace.append(obs["drone_0"].copy())
        return np.stack(trace)

    np.testing.assert_array_equal(run(), run())


# ------------------------------- reward scaling -------------------------------
def test_reward_scale_multiplies_rewards_but_not_the_raw_return():
    def run(scale):
        env = SwarmRLlibEnv(dict(SMALL, reward_scale=scale))
        env.reset(seed=3)
        rng = np.random.default_rng(3)
        rewards = []
        for _ in range(10):
            _, rew, *_ = env.step(_unit_actions(env.env, rng))
            rewards.append(rew["drone_1"])
        return np.array(rewards), env.raw_episode_return

    base, raw_base = run(1.0)
    scaled, raw_scaled = run(0.1)
    np.testing.assert_allclose(scaled, base * 0.1, atol=1e-6)
    assert raw_scaled == pytest.approx(raw_base)               # unscaled units, unchanged


# ------------------------------- config handling -------------------------------
def test_unknown_env_config_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown env_config keys"):
        SwarmRLlibEnv(dict(SMALL, num_agnts=5))                 # typo


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError, match="Unknown stage"):
        SwarmRLlibEnv(dict(SMALL, stage="hurricane"))


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_reward_scale_is_rejected(bad):
    with pytest.raises(ValueError, match="reward_scale"):
        SwarmRLlibEnv(dict(SMALL, reward_scale=bad))


def test_env_config_overrides_reach_the_swarm_env():
    env = SwarmRLlibEnv(dict(num_agents=7, world_size_xy=120.0, collision_penalty=-50.0))
    assert env.env.config.num_agents == 7
    assert env.env.config.world_size_xy == 120.0
    assert env.env.config.collision_penalty == -50.0


def test_empty_or_missing_env_config_gives_the_defaults():
    assert SwarmRLlibEnv(None).env.config == SwarmEnvConfig()
    assert SwarmRLlibEnv({}).env.config == SwarmEnvConfig()


# ------------------------------- curriculum hooks -------------------------------
def test_stage_in_env_config_is_applied_at_the_first_reset():
    env = SwarmRLlibEnv(dict(SMALL, stage="dynamic_obstacles"))
    assert env.stage_name is None                               # nothing applied until reset
    env.reset(seed=0)
    assert env.stage_name == "dynamic_obstacles"
    assert env.env.config.obstacles_move and env.env.config.obstacle_speed > 0


def test_set_stage_passes_through_and_waits_for_the_next_reset():
    env = SwarmRLlibEnv(dict(SMALL, num_obstacles=8))
    env.reset(seed=0)
    env.set_stage(STAGE_STATIC)
    env.step(_unit_actions(env.env, np.random.default_rng(0)))
    assert len(env.env.obstacles) == 8                          # running episode untouched
    env.reset(seed=1)
    assert len(env.env.obstacles) == 0 and env.stage_name == "static_world"


def test_observation_and_action_spaces_are_identical_in_every_stage():
    shapes = set()
    for stage in (STAGE_STATIC, STAGE_DYNAMIC, STAGE_WIND):
        env = SwarmRLlibEnv(SMALL)
        env.set_stage(stage)
        obs, _ = env.reset(seed=0)
        shapes.add((obs["drone_0"].shape, env.get_action_space("drone_0").shape))
    assert len(shapes) == 1


def test_episode_stats_are_captured_when_the_episode_ends():
    env = SwarmRLlibEnv(dict(SMALL, max_episode_steps=8))           # reward_scale 1.0: rewards are raw
    assert env.last_episode_stats is None and env.raw_episode_return is None
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    summed = 0.0
    for _ in range(8):
        _, rew, _, _, _ = env.step(_unit_actions(env.env, rng))
        summed += sum(rew.values())
    stats = env.last_episode_stats
    assert stats is not None
    assert stats.agent_steps == 4 * 8
    assert stats.coverage_pct == pytest.approx(env.env.coverage_grid.mean() * 100)
    assert env.raw_episode_return == pytest.approx(summed)           # swarm total over the episode


def test_episode_stats_do_not_leak_between_episodes():
    env = SwarmRLlibEnv(dict(SMALL, max_episode_steps=6))
    rng = np.random.default_rng(0)
    for episode in range(2):
        env.reset(seed=episode)
        for _ in range(6):
            env.step(_unit_actions(env.env, rng))
        assert env.last_episode_stats.agent_steps == 4 * 6      # not 2 * 24 on the second episode


# ------------------------------- registration + scale -------------------------------
def test_env_creator_builds_the_adapter():
    assert isinstance(env_creator({"num_agents": 2}), SwarmRLlibEnv)


def test_registration_does_not_raise():
    register_swarm_env()
    assert ENV_NAME == "swarmrl_v0" and SHARED_POLICY_ID == "shared_policy"


def test_full_scale_episode_is_numerically_clean_through_the_adapter():
    env = SwarmRLlibEnv(dict(max_episode_steps=150))           # 50 agents, default world
    obs, _ = env.reset(seed=0)
    rng = np.random.default_rng(0)
    done = False
    while not done:
        obs, rew, term, trunc, _ = env.step({a: rng.uniform(-1, 1, 3).astype(np.float32) for a in env.env.agents})
        assert all(np.isfinite(o).all() and np.abs(o).max() <= 1.0 + 1e-4 for o in obs.values())
        assert all(np.isfinite(r) for r in rew.values())
        done = trunc["__all__"]