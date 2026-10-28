"""RLlib adapter for SwarmEnv (Task 3, Day 9).

Wraps the PettingZoo-style SwarmEnv as an RLlib `MultiAgentEnv` and applies the
policy-facing interface from backend/env/policy_io.py:

  * observations are SCALED to roughly [-1, 1]
  * actions are received in [-1, 1] and mapped to the env's physical bounds
  * rewards can be multiplied by `reward_scale` (see below)

Why not the shipped ParallelPettingZooEnv? It would pass raw observations and
raw actions straight through, and it knows nothing about curriculum stages or
the per-episode statistics the curriculum needs.

Usage with RLlib: `register_swarm_env()` once, then `.environment("swarmrl_v0",
env_config={...})`. env_config keys are `SwarmEnvConfig` field overrides plus two
adapter options:
    "stage":        name of a curriculum stage in DEFAULT_STAGES (applied at the
                    first reset), or omitted/None
    "reward_scale": multiplier applied to every reward RLlib sees (default 1.0)
Unknown keys raise, so a typo fails loudly instead of silently training the
wrong thing.

About `reward_scale`: collisions cost -100 while exploring a voxel pays +1.
PPO's value-function loss is clipped (`vf_clip_param`), and targets of that
magnitude would be clipped away. Scaling the rewards that RLlib optimises brings
them into a range the value network can fit. It does NOT change what the env
measures: `raw_episode_return` below is always in unscaled units.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from backend.env.config import SwarmEnvConfig
from backend.env.policy_io import denormalize_action, observation_scale, scale_observation
from backend.env.swarm_env import SwarmEnv
from backend.reward.curriculum import (
    DEFAULT_STAGES,
    CurriculumStage,
    EpisodeStats,
    EpisodeStatsAccumulator,
)

ENV_NAME = "swarmrl_v0"
SHARED_POLICY_ID = "shared_policy"

_ADAPTER_KEYS = ("stage", "reward_scale")


def _split_env_config(config: Mapping[str, Any] | None) -> tuple[SwarmEnvConfig, CurriculumStage | None, float]:
    """Turn an RLlib env_config mapping into (SwarmEnvConfig, stage, reward_scale)."""
    overrides = dict(config or {})
    stage_name = overrides.pop("stage", None)
    reward_scale = float(overrides.pop("reward_scale", 1.0))

    valid = {f.name for f in dataclasses.fields(SwarmEnvConfig)}
    unknown = sorted(set(overrides) - valid)
    if unknown:
        raise ValueError(
            f"Unknown env_config keys {unknown}. Valid keys are SwarmEnvConfig fields "
            f"plus {list(_ADAPTER_KEYS)}."
        )
    if reward_scale <= 0:
        raise ValueError(f"reward_scale must be positive, got {reward_scale}")

    stage = None
    if stage_name is not None:
        by_name = {s.name: s for s in DEFAULT_STAGES}
        if stage_name not in by_name:
            raise ValueError(f"Unknown stage {stage_name!r}. Choose from {list(by_name)}.")
        stage = by_name[stage_name]

    return SwarmEnvConfig(**overrides), stage, reward_scale


class SwarmRLlibEnv(MultiAgentEnv):
    """SwarmEnv as an RLlib MultiAgentEnv. One policy-facing agent per drone."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__()
        swarm_config, stage, self.reward_scale = _split_env_config(config)

        self.env = SwarmEnv(swarm_config)
        if stage is not None:
            self.env.set_stage(stage)          # applied at the first reset()

        self._obs_scale = observation_scale(swarm_config)
        probe = self.env.possible_agents[0]
        self._act_low = self.env.action_space(probe).low
        self._act_high = self.env.action_space(probe).high

        # RLlib reads these. `agents` is kept constant (like RLlib's own
        # PettingZoo wrapper) rather than shrinking when the episode ends.
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.possible_agents)
        self._agent_ids = set(self.possible_agents)

        obs_space = gym.spaces.Box(-np.inf, np.inf, (self.env._obs_dim,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)       # policy acts in [-1, 1]
        self.observation_spaces = {aid: obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: act_space for aid in self.possible_agents}
        self.observation_space = gym.spaces.Dict(dict(self.observation_spaces))
        self.action_space = gym.spaces.Dict(dict(self.action_spaces))

        # Per-episode bookkeeping for the curriculum and for honest (unscaled) metrics.
        self._stats = EpisodeStatsAccumulator()
        self._raw_return = 0.0
        self.last_episode_stats: EpisodeStats | None = None
        self.raw_episode_return: float | None = None      # swarm total, unscaled units

    # ------------------------------------------------------------------ helpers
    def _scale_obs(self, raw_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {aid: scale_observation(o, self._obs_scale) for aid, o in raw_obs.items()}

    @property
    def stage_name(self) -> str | None:
        """Name of the curriculum stage currently applied (None = none set)."""
        return self.env.stage_name

    def set_stage(self, stage: CurriculumStage) -> None:
        """Queue a curriculum stage; it takes effect at the next reset()."""
        self.env.set_stage(stage)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._stats.reset()
        self._raw_return = 0.0
        return self._scale_obs(raw_obs), infos

    def step(self, action_dict: dict[str, np.ndarray]):
        expected = set(self.env.agents)
        got = set(action_dict)
        if got != expected:
            missing, extra = sorted(expected - got), sorted(got - expected)
            raise ValueError(f"action_dict must cover exactly the live agents; missing={missing[:3]}, extra={extra[:3]}")

        env_actions = {aid: denormalize_action(a, self._act_low, self._act_high)
                       for aid, a in action_dict.items()}
        raw_obs, rewards, terminateds, truncateds, infos = self.env.step(env_actions)

        self._stats.add_step(infos)
        self._raw_return += float(sum(rewards.values()))

        terminateds = dict(terminateds)
        truncateds = dict(truncateds)
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        if terminateds["__all__"] or truncateds["__all__"]:
            self.last_episode_stats = self._stats.finalize(self.env.coverage_grid.mean() * 100.0)
            self.raw_episode_return = self._raw_return

        scaled_rewards = {aid: r * self.reward_scale for aid, r in rewards.items()}
        return self._scale_obs(raw_obs), scaled_rewards, terminateds, truncateds, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def env_creator(env_config: Mapping[str, Any] | None = None) -> SwarmRLlibEnv:
    return SwarmRLlibEnv(env_config)


def register_swarm_env() -> None:
    """Register the env with RLlib under ENV_NAME. Call once per process before
    building an algorithm (Ray workers pick the registration up from the driver)."""
    from ray.tune.registry import register_env

    register_env(ENV_NAME, env_creator)