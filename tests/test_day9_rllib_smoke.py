"""Day-9 integration test: build a real PPO algorithm and train one iteration.

OPT-IN, because starting Ray takes tens of seconds and is the slowest, most
platform-sensitive thing in the suite. Enable with:

    PowerShell:  $env:SWARMRL_RAY_TESTS = "1"; python -m pytest tests/test_day9_rllib_smoke.py -q
    bash:        SWARMRL_RAY_TESTS=1 python -m pytest tests/test_day9_rllib_smoke.py -q
"""
import math
import os

import pytest

pytest.importorskip("ray")
pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    os.environ.get("SWARMRL_RAY_TESTS") != "1",
    reason="opt-in: set SWARMRL_RAY_TESTS=1 to run the real Ray/RLlib training test",
)


def _flatten(d, prefix=""):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", value


def test_one_real_training_iteration_runs_and_reports_finite_losses():
    import ray

    from backend.training.ppo_config import build_ppo_config
    from backend.training.ray_runtime import InsufficientMemoryError, init_ray

    try:
        init_ray()
    except InsufficientMemoryError as exc:
        pytest.skip(f"not enough free RAM to start Ray right now (environmental, not a code failure): {exc}")
    algo = None
    try:
        config = build_ppo_config(num_agents=4, max_episode_steps=40, train_batch_size=40,
                                  minibatch_size=40, num_epochs=2, hidden=(16, 16), seed=0)
        algo = config.build_algo()
        rows = dict(_flatten(algo.train()))

        assert rows["env_runners/num_env_steps_sampled"] == 40
        for key in ("policy_loss", "vf_loss", "entropy"):
            assert math.isfinite(rows[f"learners/shared_policy/{key}"]), key
        assert "learners/default_policy/policy_loss" not in rows      # only the shared policy exists
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()