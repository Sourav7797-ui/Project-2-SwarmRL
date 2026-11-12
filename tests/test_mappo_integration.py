
"""MAPPO integration regression tests.

These tests do not initialize Ray, construct an Algorithm, or train.
They validate configuration, module behavior, connectors, and checkpoints.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("ray")

from ray.rllib.core.columns import Columns
from ray.rllib.connectors.learner.learner_connector_pipeline import (
    LearnerConnectorPipeline,
)
from ray.rllib.connectors.module_to_env import NormalizeAndClipActions
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

from backend.training.mappo_config import (
    approx_gradient_updates_per_iteration,
    build_mappo_config,
    samples_per_iteration,
)
from backend.training.mappo_learner import MAPPOTorchLearner
from backend.training.mappo_module import MAPPOActorCriticModule
from backend.training.ppo_config import map_all_agents_to_shared_policy
from backend.training.rllib_env import (
    ENV_NAME,
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
)


# ------------------------------- dimension helpers -------------------------------

def expected_global_dim(num_agents: int) -> int:
    """Return the environment's global-state dimension."""
    return 7 * num_agents + 548


# ------------------------------- configuration -------------------------------

def test_default_mappo_config():
    config = build_mappo_config()

    assert config.env == ENV_NAME
    assert config.framework_str == "torch"
    assert config.env_config["num_agents"] == 50
    assert config.env_config["max_episode_steps"] == 1000
    assert config.env_config["reward_scale"] == pytest.approx(0.05)
    assert config.env_config["include_global_state"] is True
    assert "stage" not in config.env_config


def test_mappo_uses_one_shared_policy():
    config = build_mappo_config()

    assert set(config.policies) == {SHARED_POLICY_ID}

    for agent_id in ("drone_0", "drone_49", "anything_else"):
        assert (
            config.policy_mapping_fn(agent_id, None)
            == SHARED_POLICY_ID
        )

    assert (
        map_all_agents_to_shared_policy("drone_3", None)
        == SHARED_POLICY_ID
    )


def test_mappo_action_handling_configuration():
    config = build_mappo_config()

    assert config.normalize_actions is False
    assert config.clip_actions is True


def test_mappo_default_training_parameters():
    config = build_mappo_config()

    assert config.lr == pytest.approx(3e-4)
    assert config.gamma == pytest.approx(0.99)
    assert config.lambda_ == pytest.approx(0.95)
    assert config.clip_param == pytest.approx(0.2)
    assert config.entropy_coeff == 0.0
    assert config.vf_clip_param == pytest.approx(10.0)
    assert config.grad_clip == pytest.approx(0.5)
    assert config.train_batch_size_per_learner == 1000
    assert config.minibatch_size == 1000
    assert config.num_epochs == 5


def test_mappo_module_spec_has_expected_network():
    config = build_mappo_config()
    spec = config.get_multi_rl_module_spec()

    assert set(spec.rl_module_specs) == {SHARED_POLICY_ID}

    module_spec = spec.rl_module_specs[SHARED_POLICY_ID]

    assert module_spec.module_class is MAPPOActorCriticModule
    assert module_spec.observation_space["local_obs"].shape == (26,)
    assert (
        module_spec.observation_space["global_state"].shape
        == (expected_global_dim(50),)
    )
    assert module_spec.action_space.shape == (3,)
    assert module_spec.model_config["fcnet_hiddens"] == [256, 256]
    assert module_spec.model_config["fcnet_activation"] == "tanh"


@pytest.mark.parametrize("num_agents", [4, 5, 9, 12, 50])
def test_mappo_global_state_dimension_scales_with_agent_count(num_agents):
    config = build_mappo_config(num_agents=num_agents)

    module_spec = config.get_multi_rl_module_spec().rl_module_specs[
        SHARED_POLICY_ID
    ]

    assert (
        module_spec.observation_space["global_state"].shape
        == (expected_global_dim(num_agents),)
    )


def test_mappo_config_passes_rllib_validation():
    build_mappo_config().validate()


def test_mappo_customization_is_applied():
    config = build_mappo_config(
        num_agents=12,
        max_episode_steps=300,
        stage="wind",
        reward_scale=0.1,
        num_env_runners=0,
        train_batch_size=600,
        minibatch_size=300,
        num_epochs=7,
        lr=1e-3,
        hidden=(32, 64),
        seed=11,
    )

    assert config.env_config["num_agents"] == 12
    assert config.env_config["max_episode_steps"] == 300
    assert config.env_config["stage"] == "wind"
    assert config.env_config["reward_scale"] == pytest.approx(0.1)
    assert config.env_config["include_global_state"] is True

    assert config.train_batch_size_per_learner == 600
    assert config.minibatch_size == 300
    assert config.num_epochs == 7
    assert config.lr == pytest.approx(1e-3)
    assert config.seed == 11

    spec = config.get_multi_rl_module_spec()
    module_spec = spec.rl_module_specs[SHARED_POLICY_ID]

    assert module_spec.model_config["fcnet_hiddens"] == [32, 64]
    assert (
        module_spec.observation_space["global_state"].shape
        == (expected_global_dim(12),)
    )


def test_env_overrides_are_merged_with_explicit_arguments_winning():
    config = build_mappo_config(
        num_agents=9,
        env_overrides={
            "num_agents": 77,
            "world_size_xy": 150.0,
        },
    )

    assert config.env_config["num_agents"] == 9
    assert config.env_config["world_size_xy"] == 150.0
    assert config.env_config["include_global_state"] is True


def test_environment_accepts_mappo_configuration():
    num_agents = 5
    config = build_mappo_config(
        num_agents=num_agents,
        max_episode_steps=20,
    )

    env = SwarmRLlibEnv(config.env_config)

    try:
        assert len(env.possible_agents) == num_agents

        observation_space = env.get_observation_space("drone_0")

        assert observation_space["local_obs"].shape == (26,)
        assert (
            observation_space["global_state"].shape
            == (expected_global_dim(num_agents),)
        )
    finally:
        env.close()


# ------------------------------- validation -------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_agents": 0},
        {"max_episode_steps": 0},
        {"train_batch_size": 0},
        {"minibatch_size": 0},
        {"num_epochs": 0},
        {"num_env_runners": -1},
        {"hidden": ()},
        {"hidden": (32, 0)},
        {"hidden": (32, -1)},
    ],
)
def test_invalid_mappo_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_mappo_config(**kwargs)


def test_minibatch_larger_than_train_batch_is_rejected():
    with pytest.raises(ValueError, match="samples"):
        build_mappo_config(
            num_agents=8,
            train_batch_size=100,
            minibatch_size=200,
        )


def test_unknown_environment_override_is_rejected():
    with pytest.raises(ValueError, match="Unknown env_config keys"):
        build_mappo_config(
            env_overrides={"num_obstacle": 3}
        )


# ------------------------------- arithmetic -------------------------------

def test_samples_per_iteration():
    assert samples_per_iteration(50, 1000) == 50_000
    assert samples_per_iteration(8, 100) == 800


@pytest.mark.parametrize(
    "agents,batch,minibatch,epochs,expected",
    [
        (50, 1000, 1000, 5, 250),
        (50, 200, 200, 2, 100),
        (8, 100, 100, 2, 16),
        (4, 200, 64, 2, 26),
    ],
)
def test_gradient_update_estimate(
    agents, batch, minibatch, epochs, expected
):
    assert approx_gradient_updates_per_iteration(
        agents, batch, minibatch, epochs
    ) == expected


# ------------------------------- module behavior -------------------------------

@pytest.fixture(scope="module")
def mappo_module():
    config = build_mappo_config(num_agents=4)
    multi_module = config.get_multi_rl_module_spec().build()
    return multi_module[SHARED_POLICY_ID]


@pytest.fixture
def module_batch():
    return {
        Columns.OBS: {
            "local_obs": torch.randn(4, 26),
            "global_state": torch.randn(4, expected_global_dim(4)),
        }
    }


def test_module_actor_and_critic_output_shapes(
    mappo_module,
    module_batch,
):
    with torch.no_grad():
        output = mappo_module._forward_train(module_batch)
        values = mappo_module.compute_values(module_batch)

    actor = output[Columns.ACTION_DIST_INPUTS]

    assert actor.shape == (4, 6)
    assert values.shape == (4,)
    assert torch.isfinite(actor).all()
    assert torch.isfinite(values).all()


def test_actor_is_independent_of_global_state(mappo_module):
    local_obs = torch.randn(4, 26)

    batch_a = {
        Columns.OBS: {
            "local_obs": local_obs,
            "global_state": torch.randn(4, expected_global_dim(4)),
        }
    }

    batch_b = {
        Columns.OBS: {
            "local_obs": local_obs,
            "global_state": torch.randn(4, expected_global_dim(4)),
        }
    }

    with torch.no_grad():
        actor_a = mappo_module._forward_train(batch_a)[
            Columns.ACTION_DIST_INPUTS
        ]
        actor_b = mappo_module._forward_train(batch_b)[
            Columns.ACTION_DIST_INPUTS
        ]

    assert torch.allclose(actor_a, actor_b)


def test_critic_is_independent_of_local_observation(mappo_module):
    global_state = torch.randn(4, expected_global_dim(4))

    batch_a = {
        Columns.OBS: {
            "local_obs": torch.randn(4, 26),
            "global_state": global_state,
        }
    }

    batch_b = {
        Columns.OBS: {
            "local_obs": torch.randn(4, 26),
            "global_state": global_state,
        }
    }

    with torch.no_grad():
        values_a = mappo_module.compute_values(batch_a)
        values_b = mappo_module.compute_values(batch_b)

    assert torch.allclose(values_a, values_b)


def test_actor_and_critic_receive_finite_gradients(
    mappo_module,
    module_batch,
):
    mappo_module.zero_grad(set_to_none=True)

    output = mappo_module._forward_train(module_batch)
    values = mappo_module.compute_values(module_batch)

    loss = (
        output[Columns.ACTION_DIST_INPUTS].square().mean()
        + values.square().mean()
    )

    loss.backward()

    actor_grads = [
        p.grad
        for p in mappo_module.actor.parameters()
        if p.grad is not None
    ]

    critic_grads = [
        p.grad
        for p in mappo_module.critic.parameters()
        if p.grad is not None
    ]

    assert actor_grads
    assert critic_grads

    assert all(torch.isfinite(g).all() for g in actor_grads)
    assert all(torch.isfinite(g).all() for g in critic_grads)

    assert sum(g.norm().item() for g in actor_grads) > 0
    assert sum(g.norm().item() for g in critic_grads) > 0


# ------------------------------- learner and connectors -------------------------------

def test_mappo_uses_custom_torch_learner():
    config = build_mappo_config()

    assert config.learner_class is MAPPOTorchLearner
    assert issubclass(MAPPOTorchLearner, torch.nn.Module.__mro__[0]) is False


def test_learner_connector_has_expected_structure():
    config = build_mappo_config(num_agents=4)
    env = SwarmRLlibEnv(config.env_config)

    try:
        connector = config.build_learner_connector(
            env.observation_space,
            env.action_space,
        )

        assert isinstance(connector, LearnerConnectorPipeline)
        assert len(connector.connectors) == 9

        names = [type(item).__name__ for item in connector.connectors]

        assert names == [
            "AddOneTsToEpisodesAndTruncate",
            "AddObservationsFromEpisodesToBatch",
            "AddColumnsFromEpisodesToTrainBatch",
            "AddTimeDimToBatchAndZeroPad",
            "AddStatesFromEpisodesToBatch",
            "AgentToModuleMapping",
            "BatchIndividualItems",
            "NumpyToTensor",
            "GeneralAdvantageEstimation",
        ]
    finally:
        env.close()


def test_actual_learner_connector_batch_and_actor_critic_gradients():
    """Exercise real episode-to-batch connectors without an optimizer update."""
    num_agents = 4
    config = build_mappo_config(num_agents=num_agents)
    env = SwarmRLlibEnv(config.env_config)

    try:
        multi_module = config.get_multi_rl_module_spec().build()
        module = multi_module[SHARED_POLICY_ID]

        connector = config.build_learner_connector(
            env.observation_space,
            env.action_space,
        )

        agent_ids = list(env.possible_agents)

        observations = {
            agent_id: {
                "local_obs": np.zeros(26, dtype=np.float32),
                "global_state": np.zeros(
                    expected_global_dim(num_agents),
                    dtype=np.float32,
                ),
            }
            for agent_id in agent_ids
        }

        actions = {
            agent_id: np.zeros(3, dtype=np.float32)
            for agent_id in agent_ids
        }

        episode = MultiAgentEpisode(
            observation_space=env.observation_spaces,
            action_space=env.action_spaces,
            agent_to_module_mapping_fn=map_all_agents_to_shared_policy,
        )

        episode.add_env_reset(observations=observations)

        for step in range(2):
            episode.add_env_step(
                observations=observations,
                actions=actions,
                rewards={
                    agent_id: 0.1
                    for agent_id in agent_ids
                },
                terminateds={"__all__": step == 1},
                truncateds={"__all__": False},
            )

        batch = connector(
            rl_module=multi_module,
            episodes=[episode],
            explore=False,
        )

        assert SHARED_POLICY_ID in batch

        policy_batch = batch[SHARED_POLICY_ID]
        assert "obs" in policy_batch
        assert "advantages" in policy_batch
        assert "value_targets" in policy_batch

        observation_batch = policy_batch[Columns.OBS]

        assert set(observation_batch) == {
            "local_obs",
            "global_state",
        }
        assert observation_batch["local_obs"].shape == (12, 26)
        assert observation_batch["global_state"].shape == (
            12,
            expected_global_dim(num_agents),
        )

        advantages = policy_batch["advantages"]
        value_targets = policy_batch["value_targets"]

        assert advantages.shape[0] == 12
        assert value_targets.shape[0] == 12
        assert torch.isfinite(advantages).all()
        assert torch.isfinite(value_targets).all()

        # Check module-level gradient flow on the actual connector output.
        # Deliberately do not construct a Learner or call learner.update().
        module.zero_grad(set_to_none=True)

        output = module._forward_train(policy_batch)
        values = module.compute_values(policy_batch)

        loss = (
            output[Columns.ACTION_DIST_INPUTS].square().mean()
            + values.square().mean()
        )

        loss.backward()

        actor_grads = [
            parameter.grad
            for parameter in module.actor.parameters()
            if parameter.grad is not None
        ]

        critic_grads = [
            parameter.grad
            for parameter in module.critic.parameters()
            if parameter.grad is not None
        ]

        assert actor_grads
        assert critic_grads

        assert all(torch.isfinite(grad).all() for grad in actor_grads)
        assert all(torch.isfinite(grad).all() for grad in critic_grads)

        actor_norm = sum(grad.norm().item() for grad in actor_grads)
        critic_norm = sum(grad.norm().item() for grad in critic_grads)

        assert np.isfinite(actor_norm)
        assert np.isfinite(critic_norm)
        assert actor_norm > 0
        assert critic_norm > 0

    finally:
        env.close()


def test_module_to_env_connector_has_action_clipping():
    config = build_mappo_config(num_agents=4)
    env = SwarmRLlibEnv(config.env_config)

    try:
        connector = config.build_module_to_env_connector(env=env)

        clips = [
            item
            for item in connector.connectors
            if isinstance(item, NormalizeAndClipActions)
        ]

        assert len(clips) == 1
        assert clips[0].clip_actions is True
        assert clips[0].normalize_actions is False
    finally:
        env.close()


# ------------------------------- checkpoint -------------------------------

def test_multi_module_checkpoint_round_trip():
    config = build_mappo_config(num_agents=4)
    multi_module = config.get_multi_rl_module_spec().build()

    module = multi_module[SHARED_POLICY_ID]
    module.eval()

    batch = {
        Columns.OBS: {
            "local_obs": torch.randn(4, 26),
            "global_state": torch.randn(4, expected_global_dim(4)),
        }
    }

    with torch.no_grad():
        original_actor = module._forward_train(batch)[
            Columns.ACTION_DIST_INPUTS
        ].clone()

        original_critic = module.compute_values(batch).clone()

    with tempfile.TemporaryDirectory(
        prefix="swarmrl_mappo_test_"
    ) as tmp:
        checkpoint_path = str(Path(tmp) / "multi_module")

        saved_path = multi_module.save_to_path(checkpoint_path)

        assert Path(saved_path).is_dir()
        assert (Path(saved_path) / "metadata.json").exists()

        restored = type(multi_module).from_checkpoint(saved_path)
        restored_module = restored[SHARED_POLICY_ID]
        restored_module.eval()

        with torch.no_grad():
            restored_actor = restored_module._forward_train(batch)[
                Columns.ACTION_DIST_INPUTS
            ]

            restored_critic = restored_module.compute_values(batch)

        assert torch.equal(original_actor, restored_actor)
        assert torch.equal(original_critic, restored_critic)
