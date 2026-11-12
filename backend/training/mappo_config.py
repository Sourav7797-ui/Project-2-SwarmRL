
"""MAPPO configuration: shared decentralized actor + centralized critic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.learner import (
    AddColumnsFromEpisodesToTrainBatch,
    AddObservationsFromEpisodesToBatch,
    AddStatesFromEpisodesToBatch,
    AddTimeDimToBatchAndZeroPad,
    AgentToModuleMapping,
    BatchIndividualItems,
    NumpyToTensor,
)
from ray.rllib.connectors.learner.add_one_ts_to_episodes_and_truncate import (
    AddOneTsToEpisodesAndTruncate,
)
from ray.rllib.connectors.learner.general_advantage_estimation import (
    GeneralAdvantageEstimation,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec

from backend.training.mappo_learner import MAPPOTorchLearner
from backend.training.mappo_module import MAPPOActorCriticModule
from backend.training.ppo_config import map_all_agents_to_shared_policy
from backend.training.rllib_env import (
    ENV_NAME,
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
    _split_env_config,
    register_swarm_env,
)


def samples_per_iteration(num_agents: int, train_batch_size: int) -> int:
    """Number of shared-policy agent samples per iteration."""
    return num_agents * train_batch_size


def approx_gradient_updates_per_iteration(
    num_agents: int,
    train_batch_size: int,
    minibatch_size: int,
    num_epochs: int,
) -> int:
    """Estimate optimizer updates per iteration."""
    return (
        math.ceil(
            samples_per_iteration(num_agents, train_batch_size)
            / minibatch_size
        )
        * num_epochs
    )


def build_mappo_config(
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
    """Build a separate MAPPO configuration."""

    # ---------------------------------------------------------
    # 1. Validate configuration parameters
    # ---------------------------------------------------------

    for name, value in (
        ("num_agents", num_agents),
        ("max_episode_steps", max_episode_steps),
        ("train_batch_size", train_batch_size),
        ("minibatch_size", minibatch_size),
        ("num_epochs", num_epochs),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{name} must be a positive integer, got {value!r}"
            )

    if (
        isinstance(num_env_runners, bool)
        or not isinstance(num_env_runners, int)
        or num_env_runners < 0
    ):
        raise ValueError("num_env_runners must be a non-negative integer.")

    if minibatch_size > train_batch_size:
        raise ValueError(
            f"minibatch_size ({minibatch_size}) must be <= "
            f"train_batch_size ({train_batch_size}). "
            f"With {num_agents} agents, one iteration contains "
            f"{samples_per_iteration(num_agents, train_batch_size)} samples."
        )

    if not isinstance(hidden, (list, tuple)) or not hidden:
        raise ValueError("hidden must be a non-empty list or tuple.")

    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in hidden
    ):
        raise ValueError(
            "Every hidden layer size must be a positive integer."
        )

    # ---------------------------------------------------------
    # 2. Prepare environment configuration
    # ---------------------------------------------------------

    env_config = dict(env_overrides or {})

    env_config.update(
        num_agents=num_agents,
        max_episode_steps=max_episode_steps,
        reward_scale=reward_scale,
        include_global_state=True,
    )

    if stage is not None:
        env_config["stage"] = stage

    _split_env_config(env_config)
    register_swarm_env()

    # ---------------------------------------------------------
    # 3. Obtain actual per-agent observation/action spaces
    # ---------------------------------------------------------

    probe_env = SwarmRLlibEnv(env_config)

    try:
        observation_space = probe_env.get_observation_space("drone_0")
        action_space = probe_env.get_action_space("drone_0")
    finally:
        probe_env.close()

    if observation_space["local_obs"].shape != (26,):
        raise ValueError(
            "Expected local observation shape (26,), received "
            f"{observation_space['local_obs'].shape}."
        )

    # Global state:
    # 7 features per agent + 548 fixed features.
    # This gives 898 features for the default 50-agent configuration.
    expected_global_dim = 7 * num_agents + 548
    expected_global_shape = (expected_global_dim,)

    if observation_space["global_state"].shape != expected_global_shape:
        raise ValueError(
            f"Expected global state shape {expected_global_shape}, received "
            f"{observation_space['global_state'].shape}."
        )

    if action_space.shape != (3,):
        raise ValueError(
            f"Expected action shape (3,), received {action_space.shape}."
        )

    # ---------------------------------------------------------
    # 4. Configure MAPPO actor-critic RLModule
    # ---------------------------------------------------------

    module_spec = RLModuleSpec(
        module_class=MAPPOActorCriticModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={
            "fcnet_hiddens": list(hidden),
            "fcnet_activation": "tanh",
        },
    )

    # Explicit per-agent spaces are essential when there are no EnvRunners.
    shared_policy_spec = PolicySpec(
        observation_space=observation_space,
        action_space=action_space,
    )

    # ---------------------------------------------------------
    # 5. Build custom learner connector list
    # ---------------------------------------------------------

    def build_mappo_learner_connector(
        input_observation_space,
        input_action_space,
    ):
        """Return individual connectors for RLlib's learner pipeline."""

        # IMPORTANT:
        # RLlib 2.59.0 creates the outer LearnerConnectorPipeline.
        # Return individual connectors to avoid nested pipeline wrapping.

        connectors = []

        # 1. Add bootstrap timestep before assembling training batches.
        connectors.append(
            AddOneTsToEpisodesAndTruncate()
        )

        # 2. Collect observations and remaining episode columns.
        connectors.append(
            AddObservationsFromEpisodesToBatch(
                as_learner_connector=True
            )
        )

        connectors.append(
            AddColumnsFromEpisodesToTrainBatch()
        )

        # 3. Prepare time dimensions and recurrent state columns.
        connectors.append(
            AddTimeDimToBatchAndZeroPad(
                as_learner_connector=True
            )
        )

        connectors.append(
            AddStatesFromEpisodesToBatch(
                as_learner_connector=True
            )
        )

        # 4. Map every drone agent to the shared MAPPO module.
        connectors.append(
            AgentToModuleMapping(
                rl_module_specs={
                    SHARED_POLICY_ID: module_spec
                },
                agent_to_module_mapping_fn=(
                    map_all_agents_to_shared_policy
                ),
                as_learner_connector=True,
            )
        )

        # 5. Assemble module batches and convert to tensors.
        connectors.append(
            BatchIndividualItems(
                multi_agent=True
            )
        )

        connectors.append(
            NumpyToTensor(
                as_learner_connector=True
            )
        )

        # 6. Calculate GAE using the centralized critic.
        connectors.append(
            GeneralAdvantageEstimation(
                gamma=gamma,
                lambda_=lambda_,
            )
        )

        return connectors

    # ---------------------------------------------------------
    # 6. Build MAPPO configuration
    # ---------------------------------------------------------

    config = (
        PPOConfig()

        .environment(
            ENV_NAME,
            env_config=env_config,
            normalize_actions=False,
            clip_actions=True,
        )

        .framework("torch")

        .multi_agent(
            policies={
                SHARED_POLICY_ID: shared_policy_spec
            },
            policy_mapping_fn=map_all_agents_to_shared_policy,
        )

        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
        )

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

        .learners(
            num_learners=0,
            learner_class=MAPPOTorchLearner,
            learner_connector=build_mappo_learner_connector,
            add_default_connectors_to_learner_pipeline=False,
        )

        .rl_module(
            rl_module_spec=module_spec
        )
    )

    if seed is not None:
        config = config.debugging(seed=seed)

    return config
