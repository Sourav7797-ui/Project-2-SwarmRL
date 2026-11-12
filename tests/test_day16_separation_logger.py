
import numpy as np

from backend.reward.reward_logger import RewardComponentLogger
from backend.reward.reward_shaping import COMPONENT_SEPARATION


def test_logger_records_separation_when_enabled():
    logger = RewardComponentLogger()

    breakdown = {
        "exploration": np.array([1.0, 2.0], dtype=np.float32),
        "drone_collision": np.array([0.0, 0.0], dtype=np.float32),
        "obstacle_collision": np.array([0.0, 0.0], dtype=np.float32),
        "boundary": np.array([0.0, 0.0], dtype=np.float32),
        "team_bonus": np.array([0.5, 0.5], dtype=np.float32),
        COMPONENT_SEPARATION: np.array([-0.25, -0.5625], dtype=np.float32),
    }

    logger.log_step(breakdown)

    totals = logger.episode_totals()

    assert COMPONENT_SEPARATION in totals
    assert np.isclose(totals[COMPONENT_SEPARATION], -0.8125)
    assert logger.num_steps == 1


def test_logger_preserves_legacy_components_when_disabled():
    logger = RewardComponentLogger()

    breakdown = {
        "exploration": np.array([1.0, 2.0], dtype=np.float32),
        "drone_collision": np.array([0.0, 0.0], dtype=np.float32),
        "obstacle_collision": np.array([0.0, 0.0], dtype=np.float32),
        "boundary": np.array([0.0, 0.0], dtype=np.float32),
        "team_bonus": np.array([0.5, 0.5], dtype=np.float32),
    }

    logger.log_step(breakdown)

    totals = logger.episode_totals()

    assert len(totals) == 5
    assert COMPONENT_SEPARATION not in totals


def test_logger_separation_trace_and_reset():
    logger = RewardComponentLogger()

    logger.log_step({
        COMPONENT_SEPARATION: np.array([-0.25, -0.25])
    })
    logger.log_step({
        COMPONENT_SEPARATION: np.array([-0.5, -0.5])
    })

    assert np.allclose(
        logger.component_trace(COMPONENT_SEPARATION),
        [-0.5, -1.0],
    )

    logger.reset()

    assert logger.num_steps == 0
    assert logger.episode_totals() == {}
    assert logger.component_trace(COMPONENT_SEPARATION) == []
