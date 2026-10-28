"""PPO configuration for Phase 1: independent learners sharing ONE policy
(Task 3, Day 9; architecture doc section 2.3).

Every drone is a separate RLlib agent that trains on its own reward, but all 50
map to the same policy ("shared_policy"), so the network sees 50x the data and
the swarm behaves homogeneously. That is "independent learning with parameter
sharing". Phase 2 (MAPPO) adds a centralised critic on top of this.

UNITS: the sizes below are easy to get wrong, because RLlib mixes two units.
  * `train_batch_size`  counts ENVIRONMENT steps. With N drones one env step is
                        N samples for the shared policy, so an iteration trains
                        on  train_batch_size * N  samples.
  * `minibatch_size`    counts SAMPLES (one drone-step each).
  * RLlib still validates  minibatch_size <= train_batch_size  numerically, so at
    N = 50 a minibatch can never be larger than the number of env steps.
(Both were measured on Ray 2.59.0, not assumed: see scripts/rllib_smoke.py.)

Call build_ppo_config() AFTER ray.init() so the env registration reaches the
worker processes.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ray.rllib.algorithms.ppo import PPOConfig

from backend.training.rllib_env import (
    ENV_NAME,
    SHARED_POLICY_ID,
    _split_env_config,
    register_swarm_env,
)


def map_all_agents_to_shared_policy(agent_id, episode=None, **kwargs) -> str:
    """Every drone uses the one shared policy. Module-level (not a lambda) so it
    pickles cleanly to Ray worker processes on Windows."""
    return SHARED_POLICY_ID


def samples_per_iteration(num_agents: int, train_batch_size: int) -> int:
    """Samples the shared policy trains on per iteration (one per drone per env step)."""
    return num_agents * train_batch_size


def approx_gradient_updates_per_iteration(num_agents: int, train_batch_size: int,
                                          minibatch_size: int, num_epochs: int) -> int:
    """Approximate number of SGD updates per iteration: minibatches per epoch x epochs."""
    return math.ceil(samples_per_iteration(num_agents, train_batch_size) / minibatch_size) * num_epochs


def build_ppo_config(
    *,
    num_agents: int = 50,
    max_episode_steps: int = 1000,
    stage: str | None = None,
    env_overrides: Mapping[str, Any] | None = None,
    reward_scale: float = 0.05,
    num_env_runners: int = 0,
    train_batch_size: int = 1000,
    minibatch_size: int = 1000,
    num_epochs: int = 5,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lambda_: float = 0.95,
    clip_param: float = 0.2,
    entropy_coeff: float = 0.0,
    vf_clip_param: float = 10.0,
    grad_clip: float | None = 0.5,
    hidden: Sequence[int] = (256, 256),
    seed: int | None = None,
) -> PPOConfig:
    """Build the Phase 1 PPO config.

    The defaults are PROVISIONAL starting points chosen for stability, not tuned
    values: they get checked and adjusted against real training curves in
    Days 10-12. In particular `reward_scale=0.05` is there so collision penalties
    (-100) land near -5 instead of far outside what `vf_clip_param` lets the
    value network learn.
    """
    for name, value in (("num_agents", num_agents), ("max_episode_steps", max_episode_steps),
                        ("train_batch_size", train_batch_size), ("minibatch_size", minibatch_size),
                        ("num_epochs", num_epochs)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    if num_env_runners < 0:
        raise ValueError("num_env_runners must be >= 0 (0 = sample in the driver process)")
    if minibatch_size > train_batch_size:
        raise ValueError(
            f"minibatch_size ({minibatch_size}, counted in samples) must be <= train_batch_size "
            f"({train_batch_size}, counted in env steps): RLlib enforces this numerically. "
            f"With {num_agents} drones one iteration is {samples_per_iteration(num_agents, train_batch_size)} samples."
        )

    env_config = dict(env_overrides or {})
    env_config.update(num_agents=num_agents, max_episode_steps=max_episode_steps,
                      reward_scale=reward_scale)
    if stage is not None:
        env_config["stage"] = stage
    _split_env_config(env_config)       # fail fast on a typo, before any Ray worker starts

    register_swarm_env()

    config = (
        PPOConfig()
        .environment(
            ENV_NAME,
            env_config=env_config,
            # The adapter already maps the policy's [-1, 1] actions to physical units
            # (backend/env/policy_io.py). RLlib must NOT rescale them a second time.
            normalize_actions=False,
            clip_actions=True,
        )
        .framework("torch")
        .multi_agent(policies={SHARED_POLICY_ID}, policy_mapping_fn=map_all_agents_to_shared_policy)
        .env_runners(num_env_runners=num_env_runners, num_envs_per_env_runner=1)
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=lambda_,
            clip_param=clip_param,
            entropy_coeff=entropy_coeff,
            vf_clip_param=vf_clip_param,
            grad_clip=grad_clip,
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
        )
        .rl_module(model_config={
            "fcnet_hiddens": list(hidden),
            "fcnet_activation": "tanh",
            "vf_share_layers": False,       # separate value network: more stable with large penalties
        })
    )
    if seed is not None:
        config = config.debugging(seed=seed)
    return config