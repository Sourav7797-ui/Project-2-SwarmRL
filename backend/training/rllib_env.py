
"""RLlib adapter for SwarmEnv.

Supports two observation modes:

1. Existing PPO mode (default):
   Each drone receives its scaled local observation.

2. MAPPO mode (opt-in):
   Each drone receives:
       - local_obs: scaled local observation (26 features)
       - global_state: shared global state

   Global-state dimension:
       7 * num_agents + 548

   The 548 fixed features consist of:
       - 8 obstacle position slots (24 features)
       - 8 obstacle velocity slots (24 features)
       - Downsampled coverage grid (500 features)

The default swarmrl_v0 PPO behavior remains unchanged.

Adapter options:
    stage: curriculum stage name, or None
    reward_scale: multiplier applied to RLlib rewards (default 1.0)
    include_global_state: include centralized state in observations
                          (default False)

The global state is intended for centralized training. A custom RLModule
must use local_obs for the actor and global_state for the centralized critic.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from backend.env.config import SwarmEnvConfig
from backend.env.policy_io import (
    denormalize_action,
    observation_scale,
    scale_observation,
)
from backend.env.swarm_env import (
    GLOBAL_STATE_COVERAGE_FEATURES,
    GLOBAL_STATE_MAX_OBSTACLES,
    SwarmEnv,
)
from backend.reward.curriculum import (
    DEFAULT_STAGES,
    CurriculumStage,
    EpisodeStats,
    EpisodeStatsAccumulator,
)

ENV_NAME = "swarmrl_v0"
SHARED_POLICY_ID = "shared_policy"

_ADAPTER_KEYS = (
    "stage",
    "reward_scale",
    "include_global_state",
)


def _split_env_config(
    config: Mapping[str, Any] | None,
) -> tuple[SwarmEnvConfig, CurriculumStage | None, float, bool]:
    """Convert RLlib env_config into environment and adapter settings."""

    overrides = dict(config or {})

    stage_name = overrides.pop("stage", None)
    reward_scale = float(overrides.pop("reward_scale", 1.0))
    include_global_state = overrides.pop("include_global_state", False)

    if not isinstance(include_global_state, bool):
        raise ValueError(
            "include_global_state must be a boolean, "
            f"got {include_global_state!r}"
        )

    valid = {field.name for field in dataclasses.fields(SwarmEnvConfig)}
    unknown = sorted(set(overrides) - valid)

    if unknown:
        raise ValueError(
            f"Unknown env_config keys {unknown}. "
            f"Valid keys are SwarmEnvConfig fields plus "
            f"{list(_ADAPTER_KEYS)}."
        )

    if reward_scale <= 0:
        raise ValueError(
            f"reward_scale must be positive, got {reward_scale}"
        )

    stage = None

    if stage_name is not None:
        stages_by_name = {
            stage.name: stage
            for stage in DEFAULT_STAGES
        }

        if stage_name not in stages_by_name:
            raise ValueError(
                f"Unknown stage {stage_name!r}. "
                f"Choose from {list(stages_by_name)}."
            )

        stage = stages_by_name[stage_name]

    swarm_config = SwarmEnvConfig(**overrides)

    return (
        swarm_config,
        stage,
        reward_scale,
        include_global_state,
    )


class SwarmRLlibEnv(MultiAgentEnv):
    """SwarmEnv wrapped as an RLlib MultiAgentEnv."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
    ):
        super().__init__()

        (
            swarm_config,
            stage,
            self.reward_scale,
            self.include_global_state,
        ) = _split_env_config(config)

        self.env = SwarmEnv(swarm_config)

        if stage is not None:
            self.env.set_stage(stage)

        self._obs_scale = observation_scale(swarm_config)

        probe = self.env.possible_agents[0]

        self._act_low = self.env.action_space(probe).low
        self._act_high = self.env.action_space(probe).high

        # Keep the agent list constant for RLlib.
        self.possible_agents = list(self.env.possible_agents)
        self.agents = list(self.env.possible_agents)
        self._agent_ids = set(self.possible_agents)

        # --------------------------------------------------------------
        # Observation spaces
        # --------------------------------------------------------------

        local_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.env._obs_dim,),
            dtype=np.float32,
        )

        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        if self.include_global_state:
            # Fixed global-state layout:
            #
            # Agent positions:       3 * num_agents
            # Agent headings:        2 * num_agents
            # Agent speeds:          1 * num_agents
            # Agent battery:         1 * num_agents
            # Obstacle positions:    3 * 8 slots
            # Obstacle velocities:   3 * 8 slots
            # Downsampled coverage:  500
            #
            # Unused obstacle slots are zero-padded by SwarmEnv.

            global_state_dim = (
                7 * swarm_config.num_agents
                + 6 * GLOBAL_STATE_MAX_OBSTACLES
                + GLOBAL_STATE_COVERAGE_FEATURES
            )

            global_state_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(global_state_dim,),
                dtype=np.float32,
            )

            # Every agent receives its own local observation and
            # the same shared global state.
            agent_obs_space = gym.spaces.Dict(
                {
                    "local_obs": local_obs_space,
                    "global_state": global_state_space,
                }
            )

        else:
            # Preserve the original PPO observation space.
            agent_obs_space = local_obs_space

        self.observation_spaces = {
            aid: agent_obs_space
            for aid in self.possible_agents
        }

        self.action_spaces = {
            aid: action_space
            for aid in self.possible_agents
        }

        self.observation_space = gym.spaces.Dict(
            dict(self.observation_spaces)
        )

        self.action_space = gym.spaces.Dict(
            dict(self.action_spaces)
        )

        # --------------------------------------------------------------
        # Episode bookkeeping
        # --------------------------------------------------------------

        self._stats = EpisodeStatsAccumulator()
        self._raw_return = 0.0

        self.last_episode_stats: EpisodeStats | None = None
        self.raw_episode_return: float | None = None

    # ---------------------------------------------------------------------
    # Observation helpers
    # ---------------------------------------------------------------------

    def _scale_obs(
        self,
        raw_obs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Scale each agent's local observation."""

        return {
            aid: scale_observation(obs, self._obs_scale)
            for aid, obs in raw_obs.items()
        }

    def _format_observations(
        self,
        raw_obs: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Format observations for PPO or centralized-state mode."""

        local_obs = self._scale_obs(raw_obs)

        # Existing PPO mode: return only local observations.
        if not self.include_global_state:
            return local_obs

        # The environment must have been reset before this method
        # is used.
        global_state = self.env.get_global_state()

        if global_state.dtype != np.float32:
            global_state = global_state.astype(np.float32)

        # Use the same fixed-capacity dimension as the environment
        # and the declared RLlib observation space.
        expected_dim = (
            7 * len(self.possible_agents)
            + 6 * GLOBAL_STATE_MAX_OBSTACLES
            + GLOBAL_STATE_COVERAGE_FEATURES
        )

        if global_state.shape != (expected_dim,):
            raise ValueError(
                "Unexpected global state shape: "
                f"expected {(expected_dim,)}, "
                f"received {global_state.shape}"
            )

        if not np.isfinite(global_state).all():
            raise ValueError(
                "Global state contains NaN or infinite values."
            )

        # Every agent receives the same global state.
        # Copy it to avoid accidental cross-agent mutation.
        return {
            aid: {
                "local_obs": obs.astype(
                    np.float32,
                    copy=False,
                ),
                "global_state": global_state.copy(),
            }
            for aid, obs in local_obs.items()
        }

    # ---------------------------------------------------------------------
    # Curriculum helpers
    # ---------------------------------------------------------------------

    @property
    def stage_name(self) -> str | None:
        """Name of the currently applied curriculum stage."""

        return self.env.stage_name

    def set_stage(self, stage: CurriculumStage) -> None:
        """Queue a curriculum stage for the next reset."""

        self.env.set_stage(stage)

    # ---------------------------------------------------------------------
    # Gymnasium / RLlib API
    # ---------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        raw_obs, infos = self.env.reset(
            seed=seed,
            options=options,
        )

        # Start a clean episode.
        self._stats.reset()
        self._raw_return = 0.0
        self.last_episode_stats = None
        self.raw_episode_return = None

        observations = self._format_observations(raw_obs)

        return observations, infos

    def step(
        self,
        action_dict: dict[str, np.ndarray],
    ):
        expected = set(self.env.agents)
        got = set(action_dict)

        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)

            raise ValueError(
                "action_dict must cover exactly the live agents; "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

        # Convert policy actions from [-1, 1] into physical env bounds.
        env_actions = {
            aid: denormalize_action(
                action,
                self._act_low,
                self._act_high,
            )
            for aid, action in action_dict.items()
        }

        (
            raw_obs,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(env_actions)

        self._stats.add_step(infos)
        self._raw_return += float(sum(rewards.values()))

        terminateds = dict(terminateds)
        truncateds = dict(truncateds)

        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = all(truncateds.values())

        if terminateds["__all__"] or truncateds["__all__"]:
            self.last_episode_stats = self._stats.finalize(
                self.env.coverage_grid.mean() * 100.0
            )

            self.raw_episode_return = self._raw_return

        scaled_rewards = {
            aid: reward * self.reward_scale
            for aid, reward in rewards.items()
        }

        observations = self._format_observations(raw_obs)

        return (
            observations,
            scaled_rewards,
            terminateds,
            truncateds,
            infos,
        )

    # ---------------------------------------------------------------------
    # Rendering and cleanup
    # ---------------------------------------------------------------------

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def env_creator(
    env_config: Mapping[str, Any] | None = None,
) -> SwarmRLlibEnv:
    """Create an RLlib-compatible SwarmEnv instance."""

    return SwarmRLlibEnv(env_config)


def register_swarm_env() -> None:
    """Register the environment with RLlib."""

    from ray.tune.registry import register_env

    register_env(ENV_NAME, env_creator)
