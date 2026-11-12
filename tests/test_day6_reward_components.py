"""Day-6 tests: each reward component checked in isolation, with expected
values small enough to verify by hand.

Weights are PINNED explicitly (see _cfg) instead of read from the defaults.
These tests check the reward *formula*; if they depended on the tuned
defaults, retuning a weight would break them without any bug existing.
  exploration_reward=1.0, collision_penalty=-100, obstacle_collision_penalty=-75,
  boundary_margin=5.0, boundary_penalty_scale=-0.1, team_coverage_bonus_scale=0.05
  world: 200 x 200 x 40 m
"""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.reward.reward_shaping import RewardShaper

CENTER = [100.0, 100.0, 20.0]  # far from every wall -> zero boundary penalty

PINNED_WEIGHTS = dict(
    exploration_reward=1.0,
    collision_penalty=-100.0,
    obstacle_collision_penalty=-75.0,
    boundary_margin=5.0,
    boundary_penalty_scale=-0.1,
    team_coverage_bonus_scale=0.05,
)


def _cfg(**overrides):
    """SwarmEnvConfig with the formula-test weights pinned; overrides win."""
    return SwarmEnvConfig(**{**PINNED_WEIGHTS, **overrides})


def _breakdown(config, positions, newly_covered=None, collisions=None, obstacle_collisions=None):
    """Helper: run compute_with_breakdown with all-False defaults."""
    positions = np.array(positions, dtype=np.float32)
    n = len(positions)
    newly_covered = np.zeros(n, dtype=bool) if newly_covered is None else np.array(newly_covered)
    collisions = np.zeros(n, dtype=bool) if collisions is None else np.array(collisions)
    _, breakdown = RewardShaper(config).compute_with_breakdown(
        newly_covered_by_agent=newly_covered,
        collisions_by_agent=collisions,
        positions=positions,
        world_size_xy=config.world_size_xy,
        world_height=config.world_height,
        obstacle_collisions_by_agent=None if obstacle_collisions is None else np.array(obstacle_collisions),
    )
    return breakdown


# ------------------------- exploration -------------------------
def test_exploration_only_rewards_agents_that_found_new_voxels():
    cfg = _cfg(num_agents=3)
    b = _breakdown(cfg, [CENTER] * 3, newly_covered=[True, False, True])
    np.testing.assert_allclose(b["exploration"], [1.0, 0.0, 1.0])


def test_exploration_is_zero_when_nothing_new_found():
    cfg = _cfg(num_agents=4)
    b = _breakdown(cfg, [CENTER] * 4)
    np.testing.assert_allclose(b["exploration"], 0.0)


# ------------------------- collisions -------------------------
def test_drone_collision_penalizes_only_flagged_agents():
    cfg = _cfg(num_agents=3)
    b = _breakdown(cfg, [CENTER] * 3, collisions=[False, True, False])
    np.testing.assert_allclose(b["drone_collision"], [0.0, -100.0, 0.0])


def test_obstacle_collision_penalizes_only_flagged_agents():
    cfg = _cfg(num_agents=3)
    b = _breakdown(cfg, [CENTER] * 3, obstacle_collisions=[True, False, False])
    np.testing.assert_allclose(b["obstacle_collision"], [-75.0, 0.0, 0.0])


def test_obstacle_collision_defaults_to_zero_when_not_provided():
    """Backward-compat path: callers that omit obstacle_collisions get zeros."""
    cfg = _cfg(num_agents=2)
    b = _breakdown(cfg, [CENTER] * 2, obstacle_collisions=None)
    np.testing.assert_allclose(b["obstacle_collision"], 0.0)


def test_drone_and_obstacle_penalties_stack_for_the_same_agent():
    cfg = _cfg(num_agents=1)
    positions = np.array([CENTER], dtype=np.float32)
    total, _ = RewardShaper(cfg).compute_with_breakdown(
        newly_covered_by_agent=np.array([False]),
        collisions_by_agent=np.array([True]),
        positions=positions,
        world_size_xy=cfg.world_size_xy,
        world_height=cfg.world_height,
        obstacle_collisions_by_agent=np.array([True]),
    )
    assert total["drone_0"] == pytest.approx(-175.0)


# ------------------------- boundary -------------------------
def test_boundary_penalty_is_zero_in_safe_interior():
    cfg = _cfg(num_agents=1)
    b = _breakdown(cfg, [CENTER])
    assert b["boundary"][0] == pytest.approx(0.0)


def test_boundary_penalty_is_zero_exactly_at_the_margin():
    """x = margin (5m from wall) is the edge of the safe zone: no penalty yet."""
    cfg = _cfg(num_agents=1)
    b = _breakdown(cfg, [[5.0, 100.0, 20.0]])
    assert b["boundary"][0] == pytest.approx(0.0)


def test_boundary_penalty_grows_linearly_past_the_margin():
    """2m from the low-x wall -> 3m violation -> 3 * -0.1 = -0.3."""
    cfg = _cfg(num_agents=1)
    b = _breakdown(cfg, [[2.0, 100.0, 20.0]])
    assert b["boundary"][0] == pytest.approx(-0.3, abs=1e-5)


def test_boundary_penalty_applies_at_the_high_side_walls_too():
    """x=198 on a 200m world -> violation = 198 - 195 = 3 -> -0.3."""
    cfg = _cfg(num_agents=1)
    b = _breakdown(cfg, [[198.0, 100.0, 20.0]])
    assert b["boundary"][0] == pytest.approx(-0.3, abs=1e-5)


def test_boundary_penalty_covers_altitude_floor_and_ceiling():
    cfg = _cfg(num_agents=2)
    # z=1 -> violation 4 -> -0.4 ; z=39 -> violation 39-35=4 -> -0.4
    b = _breakdown(cfg, [[100.0, 100.0, 1.0], [100.0, 100.0, 39.0]])
    np.testing.assert_allclose(b["boundary"], [-0.4, -0.4], atol=1e-5)


def test_boundary_penalty_stacks_across_axes_at_a_corner():
    """Corner (0,0,0): 5 + 5 + 5 = 15m total violation -> -1.5."""
    cfg = _cfg(num_agents=1)
    b = _breakdown(cfg, [[0.0, 0.0, 0.0]])
    assert b["boundary"][0] == pytest.approx(-1.5, abs=1e-5)


def test_boundary_penalty_scale_and_margin_are_configurable():
    cfg = _cfg(num_agents=1, boundary_margin=10.0, boundary_penalty_scale=-1.0)
    # x=4 -> violation = 10 - 4 = 6 -> -6.0
    b = _breakdown(cfg, [[4.0, 100.0, 20.0]])
    assert b["boundary"][0] == pytest.approx(-6.0, abs=1e-5)


# ------------------------- team bonus -------------------------
def test_team_bonus_is_identical_for_every_agent():
    cfg = _cfg(num_agents=4)
    b = _breakdown(cfg, [CENTER] * 4, newly_covered=[True, False, False, False])
    assert np.unique(b["team_bonus"]).size == 1


def test_team_bonus_value_matches_fraction_of_swarm_that_explored():
    """2 of 4 agents explored -> 0.5 * 0.05 = 0.025 for everyone."""
    cfg = _cfg(num_agents=4)
    b = _breakdown(cfg, [CENTER] * 4, newly_covered=[True, True, False, False])
    np.testing.assert_allclose(b["team_bonus"], 0.025, atol=1e-6)


def test_team_bonus_is_zero_when_nobody_explored():
    cfg = _cfg(num_agents=4)
    b = _breakdown(cfg, [CENTER] * 4)
    np.testing.assert_allclose(b["team_bonus"], 0.0)


def test_team_bonus_reaches_agents_that_did_not_explore_themselves():
    """The whole point of the shared bonus: credit assignment across the swarm."""
    cfg = _cfg(num_agents=2)
    b = _breakdown(cfg, [CENTER] * 2, newly_covered=[True, False])
    assert b["exploration"][1] == 0.0       # agent 1 found nothing itself...
    assert b["team_bonus"][1] > 0.0         # ...but still shares in the team bonus


# ------------------------- combined -------------------------
def test_all_components_combine_into_hand_computed_total():
    """One agent, everything firing at once:
         exploration        +1.0
         drone collision  -100.0
         obstacle col.     -75.0
         boundary (x=2)     -0.3
         team bonus (1/1)   +0.05
         total            = -174.25
    """
    cfg = _cfg(num_agents=1)
    total, _ = RewardShaper(cfg).compute_with_breakdown(
        newly_covered_by_agent=np.array([True]),
        collisions_by_agent=np.array([True]),
        positions=np.array([[2.0, 100.0, 20.0]], dtype=np.float32),
        world_size_xy=cfg.world_size_xy,
        world_height=cfg.world_height,
        obstacle_collisions_by_agent=np.array([True]),
    )
    assert total["drone_0"] == pytest.approx(-174.25, abs=1e-4)
