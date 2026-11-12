"""Day-8 contract tests: freeze the PUBLIC reward + curriculum interface that
Task 3 (RLlib) and Task 5 (metrics API) will build on, and pin actual
behaviour with golden-master runs.

If one of these fails you have changed something other code depends on. That
may be intentional, but make it a deliberate decision: update the test AND
the relevant API contract together, never just the test.
"""
import dataclasses

import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from backend.reward.curriculum import (
    DEFAULT_STAGES,
    CurriculumManager,
    CurriculumStage,
    EpisodeStats,
    EpisodeStatsAccumulator,
)
from backend.reward.reward_logger import RewardComponentLogger
from backend.reward.reward_shaping import ALL_COMPONENTS, RewardShaper


def _fields(cls):
    return {f.name for f in dataclasses.fields(cls)}


# ============================ reward interface ============================
def test_reward_component_names_and_order_are_frozen():
    assert ALL_COMPONENTS == ("exploration", "drone_collision", "obstacle_collision",
                              "boundary", "team_bonus")


def test_breakdown_contract():
    cfg = SwarmEnvConfig(num_agents=3)
    rewards, breakdown = RewardShaper(cfg).compute_with_breakdown(
        newly_covered_by_agent=np.array([True, False, False]),
        collisions_by_agent=np.array([False, False, False]),
        positions=np.array([[100.0, 100.0, 20.0]] * 3, dtype=np.float32),
        world_size_xy=cfg.world_size_xy,
        world_height=cfg.world_height,
        obstacle_collisions_by_agent=np.array([False, False, False]),
    )
    assert set(breakdown) == set(ALL_COMPONENTS)
    for values in breakdown.values():
        assert values.shape == (3,) and values.dtype == np.float32
    assert list(rewards) == ["drone_0", "drone_1", "drone_2"]
    assert all(isinstance(r, float) for r in rewards.values())


def test_reward_and_contact_weights_exist_in_the_config():
    required = {"exploration_reward", "collision_penalty", "obstacle_collision_penalty",
                "boundary_margin", "boundary_penalty_scale", "team_coverage_bonus_scale",
                "contact_cooldown_ticks"}
    assert required <= _fields(SwarmEnvConfig)     # names are frozen; the VALUES may be tuned


def test_dynamics_fields_exist_in_the_config():
    required = {"num_obstacles", "obstacles_move", "obstacle_speed", "wind_speed",
                "wind_direction_drift"}
    assert required <= _fields(SwarmEnvConfig)


def test_step_info_keys_are_exactly_these():
    env = SwarmEnv(SwarmEnvConfig(num_agents=3, num_obstacles=2, max_episode_steps=5))
    env.reset(seed=0)
    _, _, _, _, infos = env.step({a: np.zeros(3, dtype=np.float32) for a in env.agents})
    expected = {"collision", "collision_event", "obstacle_collision",
                "obstacle_collision_event", "battery", "speed"}
    for info in infos.values():
        assert set(info) == expected


def test_reward_logger_public_api():
    for name in ("log_step", "episode_totals", "episode_means_per_step",
                 "component_trace", "reset", "summary_string"):
        assert callable(getattr(RewardComponentLogger, name))


# ============================ env stage interface ============================
def test_env_stage_interface_exists():
    env = SwarmEnv(SwarmEnvConfig(num_agents=2))
    assert callable(env.set_stage)
    assert env.stage_name is None                   # nothing applied until a stage is set + reset
    env.reset(seed=0)
    assert env.obstacle_velocities.shape == (env.config.num_obstacles, 3)
    assert isinstance(env.wind_angle, float)


# ============================ curriculum interface ============================
def test_curriculum_stage_fields_are_frozen():
    assert _fields(CurriculumStage) == {
        "name", "num_obstacles", "obstacles_move", "obstacle_speed", "wind_speed",
        "min_coverage_pct", "max_collisions_per_1k",
    }
    assert callable(CurriculumStage.apply_to)


def test_episode_stats_fields_are_frozen():
    assert _fields(EpisodeStats) == {"coverage_pct", "collision_events", "agent_steps"}
    assert EpisodeStats(10.0, 1, 1000).collisions_per_1k == pytest.approx(1.0)


def test_episode_stats_accumulator_api():
    for name in ("add_step", "finalize", "reset"):
        assert callable(getattr(EpisodeStatsAccumulator, name))


def test_curriculum_manager_public_api():
    for name in ("record_episode", "should_promote", "window_means", "state_dict", "load_state_dict"):
        assert callable(getattr(CurriculumManager, name))
    for name in ("stage", "stage_index", "is_final", "window_size", "episodes_total",
                 "episodes_in_stage", "promotions"):
        assert isinstance(getattr(CurriculumManager, name), property)


def test_curriculum_state_dict_keys_are_frozen():
    state = CurriculumManager(window_size=3).state_dict()
    assert set(state) == {"stage_names", "window_size", "stage_index", "episodes_total",
                          "episodes_in_stage", "window", "promotions"}


def test_default_curriculum_stage_names_and_order_are_frozen():
    assert [s.name for s in DEFAULT_STAGES] == ["static_world", "dynamic_obstacles", "wind"]


# ============================ golden-master behaviour ============================
# One fixed episode, replayed exactly. Integer counts must match exactly; the
# float sums get a small tolerance for cross-platform floating-point noise.
def _golden_run(**cfg_kw):
    cfg = SwarmEnvConfig(num_agents=20, num_obstacles=12, max_episode_steps=400, **cfg_kw)
    env = SwarmEnv(cfg)
    logger = RewardComponentLogger()
    env.reward_logger = logger
    env.reset(seed=123)
    rng = np.random.default_rng(7)
    drone_events = obstacle_events = 0
    while env.agents:
        actions = {a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                   for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        for info in infos.values():
            drone_events += info["collision_event"]
            obstacle_events += info["obstacle_collision_event"]
    t = logger.episode_totals()
    return {
        "voxels": int(env.coverage_grid.sum()),
        "drone_events": drone_events,
        "obstacle_events": obstacle_events,
        "exploration": t["exploration"],
        "boundary": t["boundary"],
        "team_bonus": t["team_bonus"],
        "net": sum(t.values()),
    }


GOLDEN_STATIC = dict(voxels=9761, drone_events=4, obstacle_events=1,
                     exploration=3814.0, boundary=-267.2687,
                     team_bonus=190.70, net=3262.4313)
GOLDEN_DYNAMIC_WIND = dict(voxels=10110, drone_events=4, obstacle_events=0,
                           exploration=3363.0, boundary=-284.1421,
                           team_bonus=168.15, net=2847.0079)


def _assert_matches(actual, golden):
    for key in ("voxels", "drone_events", "obstacle_events"):
        assert actual[key] == golden[key], f"{key}: {actual[key]} != {golden[key]}"
    for key in ("exploration", "boundary", "team_bonus", "net"):
        assert actual[key] == pytest.approx(golden[key], abs=0.05), f"{key}: {actual[key]} != {golden[key]}"


def test_golden_static_world():
    _assert_matches(_golden_run(), GOLDEN_STATIC)


def test_golden_dynamic_obstacles_and_wind():
    _assert_matches(_golden_run(obstacles_move=True, obstacle_speed=1.5, wind_speed=2.0),
                    GOLDEN_DYNAMIC_WIND)


def test_golden_run_is_deterministic():
    assert _golden_run() == _golden_run()