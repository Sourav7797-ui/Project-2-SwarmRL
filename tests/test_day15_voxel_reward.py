
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.reward.reward_shaping import RewardShaper


def make_inputs():
    positions = np.array(
        [[50.0, 50.0, 20.0],
         [60.0, 60.0, 20.0]],
        dtype=np.float32,
    )
    newly_covered = np.array([True, False])
    collisions = np.array([False, False])
    return newly_covered, collisions, positions


def test_binary_mode_preserves_original_reward():
    config = SwarmEnvConfig()
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    rewards, breakdown = shaper.compute_with_breakdown(
        newly_covered,
        collisions,
        positions,
        200.0,
        40.0,
    )

    assert breakdown["exploration"][0] == pytest.approx(1.0)
    assert breakdown["exploration"][1] == pytest.approx(0.0)
    assert rewards["drone_0"] > rewards["drone_1"]


def test_voxel_log_reward_matches_expected_formula():
    config = SwarmEnvConfig(
        exploration_reward_mode="voxel_log",
        exploration_voxel_reference=5.0,
    )
    shaper = RewardShaper(config)

    counts = np.array([0, 1, 5, 10, 24], dtype=np.float32)
    positions = np.tile(
        np.array([[50.0, 50.0, 20.0]], dtype=np.float32),
        (5, 1),
    )

    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=counts > 0,
        collisions_by_agent=np.zeros(5, dtype=bool),
        positions=positions,
        world_size_xy=200.0,
        world_height=40.0,
        new_voxel_counts_by_agent=counts,
    )

    expected = np.log1p(counts) / np.log1p(5.0)

    np.testing.assert_allclose(
        breakdown["exploration"],
        expected,
        rtol=1e-6,
    )


def test_voxel_log_zero_count_gives_zero_reward():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    _, collisions, positions = make_inputs()

    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=np.array([False, False]),
        collisions_by_agent=collisions,
        positions=positions,
        world_size_xy=200.0,
        world_height=40.0,
        new_voxel_counts_by_agent=np.array([0, 0]),
    )

    np.testing.assert_array_equal(
        breakdown["exploration"],
        np.array([0.0, 0.0], dtype=np.float32),
    )


def test_voxel_log_requires_counts():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    with pytest.raises(ValueError, match="required"):
        shaper.compute_with_breakdown(
            newly_covered,
            collisions,
            positions,
            200.0,
            40.0,
        )


def test_voxel_log_rejects_wrong_count_shape():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    with pytest.raises(ValueError, match="shape"):
        shaper.compute_with_breakdown(
            newly_covered,
            collisions,
            positions,
            200.0,
            40.0,
            new_voxel_counts_by_agent=np.array([1, 2, 3]),
        )


def test_voxel_log_rejects_negative_counts():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    with pytest.raises(ValueError, match="negative"):
        shaper.compute_with_breakdown(
            newly_covered,
            collisions,
            positions,
            200.0,
            40.0,
            new_voxel_counts_by_agent=np.array([1, -2]),
        )



def test_voxel_log_has_diminishing_marginal_returns():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    # Equal-sized intervals: 1 -> 5 -> 9
    counts = np.array([1, 5, 9], dtype=np.float32)

    positions = np.tile(
        np.array([[50.0, 50.0, 20.0]], dtype=np.float32),
        (3, 1),
    )

    _, breakdown = shaper.compute_with_breakdown(
        newly_covered_by_agent=np.ones(3, dtype=bool),
        collisions_by_agent=np.zeros(3, dtype=bool),
        positions=positions,
        world_size_xy=200.0,
        world_height=40.0,
        new_voxel_counts_by_agent=counts,
    )

    rewards = breakdown["exploration"]

    assert rewards[0] < rewards[1] < rewards[2]

    gain_1_to_5 = rewards[1] - rewards[0]
    gain_5_to_9 = rewards[2] - rewards[1]

    # Equal increases in voxel count must yield smaller reward gains.
    assert gain_1_to_5 > gain_5_to_9



def test_compute_forwards_voxel_counts():
    config = SwarmEnvConfig(exploration_reward_mode="voxel_log")
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    rewards = shaper.compute(
        newly_covered,
        collisions,
        positions,
        200.0,
        40.0,
        new_voxel_counts_by_agent=np.array([5, 0]),
    )

    assert rewards["drone_0"] > rewards["drone_1"]


def test_unknown_exploration_mode_is_rejected():
    config = SwarmEnvConfig(exploration_reward_mode="invalid")
    shaper = RewardShaper(config)

    newly_covered, collisions, positions = make_inputs()

    with pytest.raises(ValueError, match="Unknown"):
        shaper.compute_with_breakdown(
            newly_covered,
            collisions,
            positions,
            200.0,
            40.0,
        )
