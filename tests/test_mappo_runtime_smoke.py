"""MAPPO runtime smoke test: build the real algorithm and train one iteration.

OPT-IN because starting Ray and running an RLlib training iteration
is resource-intensive.

Enable in PowerShell:

    $env:SWARMRL_MAPPO_TESTS = "1"
    python -m pytest tests/test_mappo_runtime_smoke.py -v -rs
"""

import math
import os

import pytest

pytest.importorskip("ray")
pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    os.environ.get("SWARMRL_MAPPO_TESTS") != "1",
    reason=(
        "opt-in: set SWARMRL_MAPPO_TESTS=1 "
        "to run the real MAPPO training test"
    ),
)


def _flatten(data, prefix=""):
    """Flatten nested RLlib result dictionaries into slash-separated keys."""
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", value


def test_one_real_mappo_training_iteration_runs_and_reports_finite_losses():
    import ray

    from backend.training.mappo_config import build_mappo_config
    from backend.training.ray_runtime import (
        InsufficientMemoryError,
        init_ray,
    )

    try:
        init_ray()
    except InsufficientMemoryError as exc:
        pytest.skip(
            "not enough free RAM to start Ray right now "
            f"(environmental, not a code failure): {exc}"
        )

    algo = None

    try:
        config = build_mappo_config(
            num_agents=4,
            max_episode_steps=40,
            num_env_runners=0,
            train_batch_size=40,
            minibatch_size=40,
            num_epochs=1,
            hidden=(16, 16),
            seed=0,
        )

        algo = config.build_algo()

        results = algo.train()
        rows = dict(_flatten(results))

        # The iteration must collect real environment experience.
        assert rows["env_runners/num_env_steps_sampled"] > 0

        # The shared MAPPO module must report finite PPO losses.
        for key in ("policy_loss", "vf_loss", "entropy"):
            metric = f"learners/shared_policy/{key}"
            assert metric in rows, (
                f"Missing MAPPO learner metric: {metric}. "
                f"Available metrics: {sorted(rows)}"
            )
            assert math.isfinite(rows[metric]), (
                f"Non-finite MAPPO metric: {metric}={rows[metric]}"
            )

        # MAPPO must use the shared policy, not a default PPO policy.
        assert "learners/default_policy/policy_loss" not in rows

    finally:
        if algo is not None:
            algo.stop()

        ray.shutdown()