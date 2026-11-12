"""Day-7 tests: curriculum stages, episode statistics, and promotion logic."""
import json

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

# A tiny two-stage curriculum with round thresholds so expectations are obvious:
# promote when mean coverage >= 50% and collisions <= 1.0 per 1000 agent-steps.
EASY = CurriculumStage("easy", num_obstacles=0, min_coverage_pct=50.0, max_collisions_per_1k=1.0)
HARD = CurriculumStage("hard", num_obstacles=4)

GOOD = EpisodeStats(coverage_pct=60.0, collision_events=0, agent_steps=1000)
LOW_COVERAGE = EpisodeStats(coverage_pct=10.0, collision_events=0, agent_steps=1000)
CRASHY = EpisodeStats(coverage_pct=60.0, collision_events=5, agent_steps=1000)   # 5.0 per 1k


def _manager(window=3):
    return CurriculumManager(stages=(EASY, HARD), window_size=window)


def _feed(mgr, stats, n):
    """Record `n` episodes; return the list of promoted/not flags."""
    return [mgr.record_episode(stats) for _ in range(n)]


# ------------------------------- stages -------------------------------
@pytest.mark.parametrize("kwargs", [
    dict(name="", num_obstacles=0),
    dict(name="x", num_obstacles=-1),
    dict(name="x", num_obstacles=1, obstacle_speed=-1.0, obstacles_move=True),
    dict(name="x", num_obstacles=1, wind_speed=-0.1),
    dict(name="x", num_obstacles=1, obstacle_speed=2.0, obstacles_move=False),   # contradictory
    dict(name="x", num_obstacles=1, min_coverage_pct=101.0),
    dict(name="x", num_obstacles=1, max_collisions_per_1k=-0.5),
])
def test_invalid_stages_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CurriculumStage(**kwargs)


def test_apply_to_sets_obstacle_count_and_leaves_everything_else_alone():
    base = SwarmEnvConfig(num_agents=7, num_obstacles=8, world_size_xy=123.0)
    out = CurriculumStage("s", num_obstacles=3).apply_to(base)
    assert out.num_obstacles == 3
    assert out.num_agents == 7 and out.world_size_xy == 123.0
    assert base.num_obstacles == 8                       # original untouched


def test_default_curriculum_matches_the_architecture_doc():
    """static world -> dynamic obstacles -> wind, each step strictly harder."""
    names = [s.name for s in DEFAULT_STAGES]
    assert names == ["static_world", "dynamic_obstacles", "wind"]
    static, dynamic, wind = DEFAULT_STAGES
    assert static.num_obstacles == 0 and not static.obstacles_move and static.wind_speed == 0
    assert dynamic.num_obstacles > 0 and dynamic.obstacles_move and dynamic.wind_speed == 0
    assert wind.obstacles_move and wind.wind_speed > 0


# ------------------------------- stats -------------------------------
def test_collisions_per_1k_is_events_over_agent_steps():
    assert EpisodeStats(10.0, 5, 2000).collisions_per_1k == pytest.approx(2.5)
    assert EpisodeStats(10.0, 0, 2000).collisions_per_1k == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(coverage_pct=10.0, collision_events=0, agent_steps=0),
    dict(coverage_pct=10.0, collision_events=-1, agent_steps=10),
])
def test_invalid_episode_stats_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EpisodeStats(**kwargs)


def test_accumulator_counts_events_not_ticks_of_overlap():
    acc = EpisodeStatsAccumulator()
    # Tick 1: drone_0 starts a drone collision and drone_1 starts an obstacle collision.
    acc.add_step({
        "drone_0": {"collision": True, "collision_event": True,
                    "obstacle_collision": False, "obstacle_collision_event": False},
        "drone_1": {"collision": False, "collision_event": False,
                    "obstacle_collision": True, "obstacle_collision_event": True},
    })
    # Tick 2: both still overlapping, but no NEW events.
    acc.add_step({
        "drone_0": {"collision": True, "collision_event": False,
                    "obstacle_collision": False, "obstacle_collision_event": False},
        "drone_1": {"collision": False, "collision_event": False,
                    "obstacle_collision": True, "obstacle_collision_event": False},
    })
    stats = acc.finalize(coverage_pct=42.0)
    assert stats.collision_events == 2          # two events, not four overlapping ticks
    assert stats.agent_steps == 4               # 2 agents x 2 steps
    assert stats.coverage_pct == 42.0


def test_accumulator_reset_clears_counts():
    acc = EpisodeStatsAccumulator()
    acc.add_step({"a": {"collision_event": True}})
    acc.reset()
    acc.add_step({"a": {}})
    stats = acc.finalize(0.0)
    assert stats.collision_events == 0 and stats.agent_steps == 1


# ------------------------------- manager: construction -------------------------------
def test_manager_starts_at_the_first_stage():
    m = _manager()
    assert m.stage is EASY and m.stage_index == 0 and not m.is_final
    assert m.episodes_total == 0 and m.window_means() is None


@pytest.mark.parametrize("kwargs", [
    dict(stages=(), window_size=5),
    dict(stages=(EASY, HARD), window_size=0),
    dict(stages=(EASY, EASY), window_size=5),            # duplicate names
])
def test_invalid_manager_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        CurriculumManager(**kwargs)


# ------------------------------- manager: promotion -------------------------------
def test_no_promotion_until_the_window_is_full():
    m = _manager(window=3)
    flags = _feed(m, GOOD, 2)                            # excellent, but only 2 of 3 episodes
    assert flags == [False, False] and m.stage is EASY


def test_promotes_exactly_when_the_window_fills_with_good_episodes():
    m = _manager(window=3)
    flags = _feed(m, GOOD, 3)
    assert flags == [False, False, True]
    assert m.stage is HARD and m.is_final


def test_low_coverage_blocks_promotion():
    m = _manager(window=3)
    assert not any(_feed(m, LOW_COVERAGE, 10))
    assert m.stage is EASY


def test_too_many_collisions_block_promotion_even_with_good_coverage():
    m = _manager(window=3)
    assert not any(_feed(m, CRASHY, 10))
    assert m.stage is EASY


def test_both_criteria_must_hold_together():
    m = _manager(window=2)
    # Mean coverage (60+10)/2 = 35% < 50%: fails coverage although one episode was fine.
    m.record_episode(GOOD)
    assert m.record_episode(LOW_COVERAGE) is False


def test_threshold_is_inclusive():
    m = _manager(window=2)
    exactly = EpisodeStats(coverage_pct=50.0, collision_events=1, agent_steps=1000)   # 1.0 per 1k
    m.record_episode(exactly)
    assert m.record_episode(exactly) is True


def test_window_rolls_so_old_bad_episodes_age_out():
    m = _manager(window=3)
    _feed(m, LOW_COVERAGE, 3)                            # window is full of failures
    assert m.record_episode(GOOD) is False               # [low, low, good]  mean 26.7%
    assert m.record_episode(GOOD) is False               # [low, good, good] mean 43.3%
    assert m.record_episode(GOOD) is True                # [good, good, good]
    assert m.episodes_total == 6


def test_collision_rate_is_pooled_across_the_window_not_averaged_per_episode():
    """One short crashy episode must not outweigh a long clean one."""
    m = _manager(window=2)
    long_clean = EpisodeStats(coverage_pct=60.0, collision_events=0, agent_steps=9000)
    short_crashy = EpisodeStats(coverage_pct=60.0, collision_events=5, agent_steps=1000)
    m.record_episode(long_clean)
    # pooled: 5 events / 10000 steps = 0.5 per 1k <= 1.0 -> promote.
    # (A naive mean of per-episode rates would be (0 + 5)/2 = 2.5 per 1k -> blocked.)
    assert m.record_episode(short_crashy) is True


def test_window_is_cleared_on_promotion_and_evidence_does_not_carry_over():
    stages = (EASY, CurriculumStage("mid", num_obstacles=2, min_coverage_pct=50.0), HARD)
    m = CurriculumManager(stages=stages, window_size=3)
    _feed(m, GOOD, 3)                                    # easy -> mid
    assert m.stage.name == "mid" and m.window_means() is None and m.episodes_in_stage == 0
    # Needs a FRESH window of 3 good episodes: the 3 from "easy" must not count.
    assert _feed(m, GOOD, 2) == [False, False]
    assert m.record_episode(GOOD) is True and m.stage is HARD


def test_final_stage_never_promotes_and_keeps_counting():
    m = _manager(window=2)
    _feed(m, GOOD, 2)
    assert m.is_final
    assert not any(_feed(m, GOOD, 20))
    assert m.stage is HARD and m.episodes_total == 22


def test_promotion_history_is_recorded():
    m = _manager(window=2)
    _feed(m, GOOD, 2)
    (p,) = m.promotions
    assert p["from_stage"] == "easy" and p["to_stage"] == "hard" and p["at_episode"] == 2
    assert p["window"]["coverage_pct"] == pytest.approx(60.0)
    assert p["window"]["episodes"] == 2


def test_promotions_property_returns_a_copy():
    m = _manager(window=2)
    _feed(m, GOOD, 2)
    m.promotions.clear()
    assert len(m.promotions) == 1


# ------------------------------- manager: persistence -------------------------------
def test_state_roundtrip_through_json_preserves_behaviour():
    a = _manager(window=3)
    _feed(a, GOOD, 2)                                    # one episode short of promotion
    saved = json.loads(json.dumps(a.state_dict()))       # must be JSON-serialisable

    b = _manager(window=3)
    b.load_state_dict(saved)
    assert b.stage_index == 0 and b.episodes_total == 2
    assert b.window_means() == pytest.approx(a.window_means())
    assert b.record_episode(GOOD) is True                # resumes exactly where `a` left off
    assert b.stage is HARD


def test_state_roundtrip_after_promotion():
    a = _manager(window=2)
    _feed(a, GOOD, 2)
    b = _manager(window=2)
    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    assert b.stage is HARD and b.is_final and len(b.promotions) == 1


def test_loading_state_from_a_different_curriculum_is_rejected():
    saved = _manager().state_dict()
    other = CurriculumManager(stages=DEFAULT_STAGES, window_size=3)
    with pytest.raises(ValueError):
        other.load_state_dict(saved)


def test_loading_state_with_a_different_window_size_is_rejected():
    saved = _manager(window=3).state_dict()
    with pytest.raises(ValueError):
        _manager(window=5).load_state_dict(saved)


# ------------------------------- integration with the real env -------------------------------
def test_curriculum_pieces_work_with_the_real_environment():
    """Stage -> config -> env -> accumulator -> manager, using the env's real
    `infos` keys, so a renamed key would break this test."""
    mgr = CurriculumManager(stages=DEFAULT_STAGES, window_size=2)
    base = SwarmEnvConfig(num_agents=5, max_episode_steps=20)

    for episode in range(2):
        cfg = mgr.stage.apply_to(base)
        assert cfg.num_obstacles == mgr.stage.num_obstacles
        env = SwarmEnv(cfg)
        env.reset(seed=episode)
        acc = EpisodeStatsAccumulator()
        rng = np.random.default_rng(episode)
        while env.agents:
            actions = {a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
                       for a in env.agents}
            _, _, _, _, infos = env.step(actions)
            acc.add_step(infos)
        stats = acc.finalize(env.coverage_grid.mean() * 100)
        assert stats.agent_steps == 5 * 20
        mgr.record_episode(stats)

    assert mgr.episodes_total == 2
    # Either it promoted (window cleared, now on stage 1) or it did not (both
    # episodes still sit in the window). Nothing else is possible.
    promoted = mgr.stage_index == 1 and mgr.window_means() is None
    not_promoted = mgr.stage_index == 0 and mgr.window_means()["episodes"] == 2
    assert promoted or not_promoted