"""Day-8 tests: moving obstacles, wind, and wiring the curriculum into the env
through SwarmEnv.set_stage()."""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from backend.reward.curriculum import (
    STAGE_DYNAMIC,
    STAGE_STATIC,
    STAGE_WIND,
    CurriculumManager,
    CurriculumStage,
    EpisodeStats,
)
from backend.reward.reward_logger import RewardComponentLogger


def _idle(env):
    return {a: np.zeros(3, dtype=np.float32) for a in env.agents}


def _step_idle(env, n=1):
    for _ in range(n):
        env.step(_idle(env))


def _make(num_agents=1, num_obstacles=1, **kw):
    cfg = SwarmEnvConfig(num_agents=num_agents, num_obstacles=num_obstacles,
                         max_episode_steps=5000, **kw)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    return env


# =============================== moving obstacles ===============================
def test_obstacles_are_static_by_default():
    env = _make(num_agents=2, num_obstacles=4)
    before = env.obstacles.copy()
    _step_idle(env, 20)
    np.testing.assert_array_equal(env.obstacles, before)
    assert np.all(env.obstacle_velocities == 0.0)


def test_obstacles_stay_static_if_moving_is_on_but_speed_is_zero():
    env = _make(num_obstacles=3, obstacles_move=True, obstacle_speed=0.0)
    before = env.obstacles.copy()
    _step_idle(env, 20)
    np.testing.assert_array_equal(env.obstacles, before)


def test_each_obstacle_moves_by_speed_times_dt_per_step():
    env = _make(num_obstacles=5, obstacles_move=True, obstacle_speed=1.5)
    env.obstacles[:] = [100.0, 100.0, 20.0]              # interior: no bounce this step
    before = env.obstacles.copy()
    _step_idle(env)
    displacement = np.linalg.norm(env.obstacles - before, axis=1)
    np.testing.assert_allclose(displacement, 1.5 * env.config.dt, atol=1e-5)


def test_obstacle_velocities_all_have_the_configured_speed():
    env = _make(num_obstacles=20, obstacles_move=True, obstacle_speed=2.5)
    np.testing.assert_allclose(np.linalg.norm(env.obstacle_velocities, axis=1), 2.5, atol=1e-5)


def test_obstacles_point_in_different_directions():
    env = _make(num_obstacles=20, obstacles_move=True, obstacle_speed=1.0)
    assert np.unique(np.round(env.obstacle_velocities, 3), axis=0).shape[0] == 20


def test_moving_obstacles_bounce_and_never_leave_the_world():
    """A tiny world forces constant bouncing; speed must stay constant throughout."""
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=6, world_size_xy=20.0, world_height=10.0,
                         obstacles_move=True, obstacle_speed=3.0, max_episode_steps=5000)
    env = SwarmEnv(cfg)
    env.reset(seed=3)
    world_max = np.array([20.0, 20.0, 10.0])
    for _ in range(1500):
        _step_idle(env)
        assert np.all(env.obstacles >= -1e-4) and np.all(env.obstacles <= world_max + 1e-4)
        np.testing.assert_allclose(np.linalg.norm(env.obstacle_velocities, axis=1), 3.0, atol=1e-4)


def test_bounce_reflects_the_position_and_flips_that_velocity_component():
    env = _make(num_obstacles=1, obstacles_move=True, obstacle_speed=1.5)
    env.obstacles[:] = [199.95, 100.0, 20.0]             # 5 cm from the +x wall
    env.obstacle_velocities[:] = [1.5, 0.0, 0.0]         # heading into it, 0.15 m per step
    _step_idle(env)
    # unreflected x would be 200.10; reflected = 2*200 - 200.10 = 199.90
    assert env.obstacles[0, 0] == pytest.approx(199.90, abs=1e-3)
    assert env.obstacle_velocities[0, 0] == pytest.approx(-1.5)
    assert env.obstacle_velocities[0, 1] == 0.0          # other components untouched


def test_moving_obstacles_are_reproducible_for_a_seed():
    def run(seed):
        cfg = SwarmEnvConfig(num_agents=2, num_obstacles=5, obstacles_move=True,
                             obstacle_speed=2.0, max_episode_steps=5000)
        env = SwarmEnv(cfg)
        env.reset(seed=seed)
        _step_idle(env, 50)
        return env.obstacles.copy()

    np.testing.assert_array_equal(run(7), run(7))
    assert not np.allclose(run(7), run(8))


def test_turning_dynamics_on_does_not_change_the_initial_world():
    """Obstacle velocities are drawn last, so a seed gives the same positions,
    obstacles and headings whether or not obstacles move."""
    base = SwarmEnvConfig(num_agents=15, num_obstacles=6)
    moving = SwarmEnvConfig(num_agents=15, num_obstacles=6, obstacles_move=True,
                            obstacle_speed=1.5, wind_speed=2.0)
    a, b = SwarmEnv(base), SwarmEnv(moving)
    a.reset(seed=11)
    b.reset(seed=11)
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.obstacles, b.obstacles)
    np.testing.assert_array_equal(a.headings, b.headings)


def test_an_approaching_obstacle_hits_a_stationary_drone_and_pushes_it_out():
    env = _make(num_obstacles=1, obstacles_move=True, obstacle_speed=1.5)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    env.obstacles[:] = [[96.4, 100.0, 20.0]]             # 3.6 m away; contact is at 3.5 m
    env.obstacle_velocities[:] = [[1.5, 0.0, 0.0]]       # closing at 0.15 m per step
    _, _, _, _, infos = env.step(_idle(env))
    assert infos["drone_0"]["obstacle_collision"] is True
    assert infos["drone_0"]["obstacle_collision_event"] is True
    gap = np.linalg.norm(env.positions[0] - env.obstacles[0])
    assert gap == pytest.approx(env.config.drone_radius + env.config.obstacle_radius, abs=1e-3)


# ==================================== wind ====================================
def test_no_wind_by_default_so_an_idle_drone_stays_put():
    env = _make()
    env.positions[:] = [[100.0, 100.0, 20.0]]
    _step_idle(env, 20)
    np.testing.assert_allclose(env.positions[0], [100.0, 100.0, 20.0], atol=1e-5)


def test_wind_drifts_an_idle_drone_by_wind_speed_times_dt():
    env = _make(num_obstacles=0, wind_speed=2.0, wind_direction_drift=0.0)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    env.wind_angle = 0.0                                  # blowing toward +x
    _step_idle(env, 5)
    np.testing.assert_allclose(env.positions[0], [100.0 + 5 * 0.2, 100.0, 20.0], atol=1e-4)


def test_wind_direction_follows_the_wind_angle():
    env = _make(num_obstacles=0, wind_speed=2.0, wind_direction_drift=0.0)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    env.wind_angle = np.pi / 2                            # toward +y
    _step_idle(env, 5)
    np.testing.assert_allclose(env.positions[0], [100.0, 101.0, 20.0], atol=1e-4)


def test_wind_is_horizontal_only():
    env = _make(num_obstacles=0, wind_speed=3.0)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    _step_idle(env, 100)
    assert env.positions[0, 2] == pytest.approx(20.0, abs=1e-4)


def test_wind_does_not_change_the_drones_own_speed():
    env = _make(num_obstacles=0, wind_speed=3.0)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    _step_idle(env, 10)
    assert env.speed[0] == 0.0                            # drifting is not flying


def test_wind_pushes_every_drone_the_same_way():
    env = _make(num_agents=5, num_obstacles=0, wind_speed=2.0, wind_direction_drift=0.0)
    env.positions[:] = np.array([[50.0 + 20 * i, 100.0, 20.0] for i in range(5)])
    start = env.positions.copy()
    _step_idle(env, 10)
    drift = env.positions - start
    np.testing.assert_allclose(drift, np.tile(drift[0], (5, 1)), atol=1e-4)   # all rows identical
    assert np.linalg.norm(drift[0]) == pytest.approx(10 * 0.2, abs=1e-3)


def test_wind_cannot_push_a_drone_out_of_the_world():
    env = _make(num_obstacles=0, wind_speed=5.0, wind_direction_drift=0.0)
    env.positions[:] = [[199.9, 100.0, 20.0]]
    env.wind_angle = 0.0
    _step_idle(env, 30)
    assert env.positions[0, 0] <= env.config.world_size_xy


def test_wind_direction_random_walks():
    env = _make(num_obstacles=0, wind_speed=1.0, wind_direction_drift=0.1)
    env.positions[:] = [[100.0, 100.0, 20.0]]
    start = env.wind_angle
    angles = []
    for _ in range(200):
        env.positions[:] = [[100.0, 100.0, 20.0]]
        _step_idle(env)
        angles.append(env.wind_angle)
    assert len(set(np.round(angles, 6))) > 100            # it genuinely wanders
    assert abs(angles[-1] - start) > 0.0


def test_wind_is_reproducible_for_a_seed():
    def trajectory(seed):
        cfg = SwarmEnvConfig(num_agents=1, num_obstacles=0, wind_speed=2.0, max_episode_steps=5000)
        env = SwarmEnv(cfg)
        env.reset(seed=seed)
        env.positions[:] = [[100.0, 100.0, 20.0]]
        _step_idle(env, 100)
        return env.positions[0].copy()

    np.testing.assert_array_equal(trajectory(4), trajectory(4))
    assert not np.allclose(trajectory(4), trajectory(5))


def test_initial_wind_direction_varies_with_the_seed():
    angles = []
    for seed in range(20):
        env = SwarmEnv(SwarmEnvConfig(num_agents=1, num_obstacles=0, wind_speed=1.0))
        env.reset(seed=seed)
        angles.append(env.wind_angle)
    assert all(0.0 <= a < 2 * np.pi for a in angles)
    assert np.std(angles) > 0.5


def test_windless_run_is_identical_whatever_the_wind_drift_setting():
    """With wind off, the wind_direction_drift setting must have no effect at all."""
    def run(drift):
        cfg = SwarmEnvConfig(num_agents=3, num_obstacles=2, wind_speed=0.0,
                             wind_direction_drift=drift, max_episode_steps=5000)
        env = SwarmEnv(cfg)
        env.reset(seed=2)
        rng = np.random.default_rng(0)
        for _ in range(30):
            env.step({a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                      for a in env.agents})
        return env.positions.copy()

    np.testing.assert_array_equal(run(0.0), run(0.7))


# ============================ curriculum wiring: set_stage ============================
def _stage_env(num_obstacles=8):
    cfg = SwarmEnvConfig(num_agents=4, num_obstacles=num_obstacles, max_episode_steps=5000)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    return env


def test_set_stage_does_not_change_a_running_episode():
    env = _stage_env(num_obstacles=8)
    env.set_stage(STAGE_STATIC)                           # would remove every obstacle
    _step_idle(env, 5)
    assert len(env.obstacles) == 8
    assert env.config.num_obstacles == 8
    assert env.stage_name is None


def test_set_stage_takes_effect_at_the_next_reset():
    env = _stage_env(num_obstacles=8)
    env.set_stage(STAGE_STATIC)
    env.reset(seed=1)
    assert len(env.obstacles) == 0
    assert env.config.num_obstacles == 0
    assert env.stage_name == "static_world"


def test_dynamic_stage_switches_on_moving_obstacles():
    env = _stage_env(num_obstacles=0)
    env.set_stage(STAGE_DYNAMIC)
    env.reset(seed=1)
    assert env.config.obstacles_move and env.config.obstacle_speed == STAGE_DYNAMIC.obstacle_speed
    assert len(env.obstacles) == STAGE_DYNAMIC.num_obstacles
    np.testing.assert_allclose(np.linalg.norm(env.obstacle_velocities, axis=1),
                               STAGE_DYNAMIC.obstacle_speed, atol=1e-5)
    assert env.config.wind_speed == 0.0


def test_wind_stage_switches_on_wind_as_well():
    env = _stage_env(num_obstacles=0)
    env.set_stage(STAGE_WIND)
    env.reset(seed=1)
    assert env.config.wind_speed == STAGE_WIND.wind_speed
    assert env.config.obstacles_move
    env.positions[:] = np.array([[100.0, 100.0, 20.0]] * 4)
    _step_idle(env)
    assert np.linalg.norm(env.positions[0] - [100.0, 100.0, 20.0]) > 0.05   # it drifted


def test_a_pending_stage_is_applied_once():
    env = _stage_env()
    env.set_stage(STAGE_STATIC)
    env.reset(seed=1)
    assert env.stage_name == "static_world"
    env.reset(seed=2)                                     # nothing pending: stage persists
    assert env.stage_name == "static_world" and len(env.obstacles) == 0


def test_stage_change_preserves_everything_it_does_not_own():
    cfg = SwarmEnvConfig(num_agents=4, num_obstacles=8, world_size_xy=150.0,
                         exploration_reward=3.0, max_episode_steps=5000)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.set_stage(STAGE_WIND)
    env.reset(seed=1)
    assert env.config.num_agents == 4 and env.config.world_size_xy == 150.0
    assert env.config.exploration_reward == 3.0
    assert env.reward_shaper.config is env.config         # shaper follows the new config


def test_reward_logger_survives_a_stage_change():
    env = _stage_env()
    logger = RewardComponentLogger()
    env.reward_logger = logger
    _step_idle(env, 3)
    env.set_stage(STAGE_DYNAMIC)
    env.reset(seed=1)
    _step_idle(env, 3)
    assert env.reward_logger is logger and logger.num_steps == 6


def test_observation_shape_is_identical_across_every_stage():
    shapes = set()
    for stage in (STAGE_STATIC, STAGE_DYNAMIC, STAGE_WIND):
        env = _stage_env()
        env.set_stage(stage)
        obs, _ = env.reset(seed=0)
        shapes.add(obs["drone_0"].shape)
    assert len(shapes) == 1


def test_apply_to_sets_all_four_dynamics_parameters_and_nothing_else():
    base = SwarmEnvConfig(num_agents=9, world_size_xy=111.0)
    stage = CurriculumStage("s", num_obstacles=3, obstacles_move=True,
                            obstacle_speed=2.0, wind_speed=4.0)
    out = stage.apply_to(base)
    assert (out.num_obstacles, out.obstacles_move, out.obstacle_speed, out.wind_speed) == (3, True, 2.0, 4.0)
    assert out.num_agents == 9 and out.world_size_xy == 111.0
    assert (base.obstacles_move, base.obstacle_speed, base.wind_speed) == (False, 0.0, 0.0)


def test_full_ladder_manager_drives_the_env_through_all_three_stages():
    """CurriculumManager decides, set_stage applies it, the env obeys. A window
    of 1 with trivial thresholds promotes after every episode."""
    stages = (
        CurriculumStage("static_world", 0, min_coverage_pct=0.0, max_collisions_per_1k=1e9),
        CurriculumStage("dynamic_obstacles", 4, obstacles_move=True, obstacle_speed=1.5,
                        min_coverage_pct=0.0, max_collisions_per_1k=1e9),
        CurriculumStage("wind", 4, obstacles_move=True, obstacle_speed=1.5, wind_speed=2.0),
    )
    mgr = CurriculumManager(stages=stages, window_size=1)
    env = SwarmEnv(SwarmEnvConfig(num_agents=3, num_obstacles=8, max_episode_steps=10))

    seen = []
    for episode in range(4):
        env.set_stage(mgr.stage)
        env.reset(seed=episode)
        seen.append((env.stage_name, len(env.obstacles), env.config.wind_speed))
        while env.agents:
            env.step(_idle(env))
        mgr.record_episode(EpisodeStats(coverage_pct=10.0, collision_events=0, agent_steps=30))

    assert seen == [
        ("static_world", 0, 0.0),
        ("dynamic_obstacles", 4, 0.0),
        ("wind", 4, 2.0),
        ("wind", 4, 2.0),                                 # final stage holds
    ]


# ========================= contact cooldown (billing of flickering contact) =========================
def _pair(cooldown):
    cfg = SwarmEnvConfig(num_agents=2, num_obstacles=0, contact_cooldown_ticks=cooldown,
                         max_episode_steps=5000)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    return env


OVERLAP = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]
APART = [[100.0, 100.0, 20.0], [150.0, 100.0, 20.0]]


def _drive(env, layout):
    """Put the drones in `layout`, take one idle step, return drone_0's info."""
    env.positions[:] = layout
    _, _, _, _, infos = env.step(_idle(env))
    return infos["drone_0"]


def test_flickering_contact_is_billed_once():
    """In, out, in, out ... every tick for 8 ticks: one contact, one event."""
    env = _pair(cooldown=10)
    logger = RewardComponentLogger()
    env.reward_logger = logger
    infos = [_drive(env, OVERLAP if t % 2 == 0 else APART) for t in range(8)]
    assert [i["collision"] for i in infos] == [True, False] * 4          # it really flickers
    assert [i["collision_event"] for i in infos] == [True] + [False] * 7
    assert logger.episode_totals()["drone_collision"] == pytest.approx(2 * env.config.collision_penalty)


def test_a_new_event_needs_exactly_cooldown_clear_ticks():
    cooldown = 5
    for clear_ticks, expect_new_event in [(cooldown - 1, False), (cooldown, True)]:
        env = _pair(cooldown)
        assert _drive(env, OVERLAP)["collision_event"] is True           # first contact
        for _ in range(clear_ticks):
            _drive(env, APART)
        assert _drive(env, OVERLAP)["collision_event"] is expect_new_event


def test_cooldown_of_one_reproduces_plain_rising_edge_billing():
    env = _pair(cooldown=1)
    infos = [_drive(env, OVERLAP if t % 2 == 0 else APART) for t in range(8)]
    assert [i["collision_event"] for i in infos] == [True, False] * 4    # every re-contact is billed


def test_continuous_contact_never_re_bills_however_long_it_lasts():
    env = _pair(cooldown=3)
    events = [_drive(env, OVERLAP)["collision_event"] for _ in range(40)]
    assert events == [True] + [False] * 39


def test_cooldown_applies_to_obstacle_contact_too():
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=1, contact_cooldown_ticks=10, max_episode_steps=5000)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.obstacles[:] = [[50.0, 50.0, 10.0]]
    inside, outside = [[50.0, 50.0, 10.0]], [[80.0, 50.0, 10.0]]

    events = []
    for t in range(6):
        env.positions[:] = inside if t % 2 == 0 else outside
        _, _, _, _, infos = env.step(_idle(env))
        events.append(infos["drone_0"]["obstacle_collision_event"])
    assert events == [True] + [False] * 5


def test_cooldown_state_does_not_leak_across_reset():
    env = _pair(cooldown=10)
    _drive(env, OVERLAP)                                  # leaves the counters at 0
    env.reset(seed=1)
    assert _drive(env, OVERLAP)["collision_event"] is True