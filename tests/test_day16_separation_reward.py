
"""Day 16 tests for optional soft drone separation reward."""

import numpy as np
import pytest

from dataclasses import replace

from backend.env.config import SwarmEnvConfig
from backend.reward.reward_shaping import (
    RewardShaper,
    COMPONENT_SEPARATION,
    COMPONENT_DRONE_COLLISION,
)


def calculate_reward(positions, enabled=True, collisions=None):
    """Calculate reward for a simple swarm arrangement."""
    positions = np.asarray(positions, dtype=np.float32)
    n = len(positions)

    config = replace(
        SwarmEnvConfig(),
        separation_reward_enabled=enabled,
        preferred_separation=4.0,
        separation_penalty_scale=-1.0,
        min_separation=2.0,
    )

    if collisions is None:
        collisions = np.zeros(n, dtype=np.float32)

    shaper = RewardShaper(config)

    _rewards, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=np.zeros(n, dtype=np.float32),
        collisions_by_agent=np.asarray(collisions, dtype=np.float32),
        positions=positions,
        world_size_xy=200.0,
        world_height=40.0,
    )

    return breakdown


def test_no_separation_penalty_when_feature_disabled():
    positions = [
        [50, 50, 20],
        [53, 50, 20],
    ]

    breakdown = calculate_reward(positions, enabled=False)

    # Disabled mode preserves the original five-component contract.
    assert COMPONENT_SEPARATION not in breakdown

def test_no_penalty_at_or_above_preferred_distance():
    positions = [
        [50, 50, 20],
        [54, 50, 20],
    ]

    breakdown = calculate_reward(positions, enabled=True)

    np.testing.assert_allclose(
        breakdown[COMPONENT_SEPARATION],
        [0.0, 0.0],
    )


@pytest.mark.parametrize(
    "distance, expected_penalty",
    [
        (3.0, -0.25),
        (2.5, -0.5625),
        (2.1, -0.9025),
    ],
)
def test_penalty_increases_as_drones_get_closer(
    distance, expected_penalty
):
    positions = [
        [50, 50, 20],
        [50 + distance, 50, 20],
    ]

    breakdown = calculate_reward(positions, enabled=True)

    np.testing.assert_allclose(
        breakdown[COMPONENT_SEPARATION],
        [expected_penalty, expected_penalty],
        atol=1e-5,
    )


def test_each_drone_uses_its_nearest_neighbour():
    positions = [
        [50, 50, 20],
        [53, 50, 20],
        [60, 50, 20],
    ]

    breakdown = calculate_reward(positions, enabled=True)

    # Nearest-neighbour distances are 3 m, 3 m and 7 m.
    expected = [-0.25, -0.25, 0.0]

    np.testing.assert_allclose(
        breakdown[COMPONENT_SEPARATION],
        expected,
        atol=1e-5,
    )


def test_single_drone_has_zero_separation_penalty():
    positions = [
        [50, 50, 20],
    ]

    breakdown = calculate_reward(positions, enabled=True)

    np.testing.assert_allclose(
        breakdown[COMPONENT_SEPARATION],
        [0.0],
    )


def test_hard_collision_penalty_remains_unchanged():
    positions = [
        [50, 50, 20],
        [51.5, 50, 20],
    ]

    breakdown = calculate_reward(
        positions,
        enabled=True,
        collisions=[1.0, 0.0],
    )

    # Existing hard collision penalty remains -100.
    np.testing.assert_allclose(
        breakdown[COMPONENT_DRONE_COLLISION],
        [-100.0, 0.0],
    )

    # Soft separation is calculated independently.
    np.testing.assert_allclose(
        breakdown[COMPONENT_SEPARATION],
        [-1.5625, -1.5625],
        atol=1e-5,
    )