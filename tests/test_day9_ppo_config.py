"""Day-9 tests: the Phase 1 PPO config builder. No Ray cluster is started and no
network is built; these check the configuration the builder produces."""
import pickle

import pytest

pytest.importorskip("ray")

from backend.training.ppo_config import (  # noqa: E402
    approx_gradient_updates_per_iteration,
    build_ppo_config,
    map_all_agents_to_shared_policy,
    samples_per_iteration,
)
from backend.training.rllib_env import ENV_NAME, SHARED_POLICY_ID, SwarmRLlibEnv  # noqa: E402


# ------------------------------- defaults -------------------------------
def test_default_config_targets_the_swarm_env_with_torch():
    config = build_ppo_config()
    assert config.env == ENV_NAME
    assert config.framework_str == "torch"
    assert config.env_config["num_agents"] == 50
    assert config.env_config["max_episode_steps"] == 1000
    assert config.env_config["reward_scale"] == pytest.approx(0.05)
    assert "stage" not in config.env_config


def test_every_drone_maps_to_the_single_shared_policy():
    config = build_ppo_config()
    assert set(config.policies) == {SHARED_POLICY_ID}
    for aid in ("drone_0", "drone_49", "anything_else"):
        assert config.policy_mapping_fn(aid, None) == SHARED_POLICY_ID
        assert map_all_agents_to_shared_policy(aid, episode=object(), extra=1) == SHARED_POLICY_ID


def test_rllib_is_told_not_to_rescale_actions_a_second_time():
    config = build_ppo_config()
    assert config.normalize_actions is False          # the adapter already maps [-1, 1] -> physical units
    assert config.clip_actions is True


def test_default_training_hyperparameters():
    config = build_ppo_config()
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


def test_default_network_has_separate_value_head_and_tanh():
    mc = build_ppo_config().model_config
    assert list(mc["fcnet_hiddens"]) == [256, 256]
    assert mc["fcnet_activation"] == "tanh"
    assert mc["vf_share_layers"] is False


def test_rllibs_own_validation_accepts_the_defaults():
    build_ppo_config().validate()


# ------------------------------- customisation -------------------------------
def test_arguments_flow_through_to_the_config():
    config = build_ppo_config(
        num_agents=12, max_episode_steps=300, stage="wind", reward_scale=0.1,
        num_env_runners=3, train_batch_size=600, minibatch_size=300, num_epochs=7,
        lr=1e-3, hidden=(32, 32), seed=11,
    )
    assert config.env_config["stage"] == "wind"
    assert config.env_config["num_agents"] == 12 and config.env_config["reward_scale"] == 0.1
    assert config.num_env_runners == 3
    assert config.train_batch_size_per_learner == 600 and config.minibatch_size == 300
    assert config.num_epochs == 7 and config.lr == pytest.approx(1e-3)
    assert list(config.model_config["fcnet_hiddens"]) == [32, 32]
    assert config.seed == 11


def test_env_overrides_are_merged_but_explicit_arguments_win():
    config = build_ppo_config(num_agents=9, env_overrides={"num_agents": 77, "world_size_xy": 150.0})
    assert config.env_config["world_size_xy"] == 150.0
    assert config.env_config["num_agents"] == 9


def test_builder_output_is_accepted_by_the_environment():
    config = build_ppo_config(num_agents=5, max_episode_steps=20, stage="dynamic_obstacles")
    env = SwarmRLlibEnv(config.env_config)
    assert len(env.possible_agents) == 5


def test_each_call_returns_an_independent_config():
    a = build_ppo_config(num_agents=3)
    b = build_ppo_config(num_agents=4)
    assert a.env_config["num_agents"] == 3 and b.env_config["num_agents"] == 4


def test_policy_mapping_function_pickles_for_worker_processes():
    restored = pickle.loads(pickle.dumps(map_all_agents_to_shared_policy))
    assert restored("drone_3", None) == SHARED_POLICY_ID


# ------------------------------- validation -------------------------------
def test_minibatch_larger_than_batch_is_rejected_with_a_units_explanation():
    with pytest.raises(ValueError, match="env steps"):
        build_ppo_config(train_batch_size=400, minibatch_size=2500)


@pytest.mark.parametrize("kwargs", [
    dict(num_agents=0), dict(max_episode_steps=0), dict(train_batch_size=0),
    dict(minibatch_size=0), dict(num_epochs=0), dict(num_env_runners=-1),
])
def test_non_positive_sizes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        build_ppo_config(**kwargs)


def test_a_typo_in_env_overrides_fails_before_any_ray_worker_starts():
    with pytest.raises(ValueError, match="Unknown env_config keys"):
        build_ppo_config(env_overrides={"num_obstacle": 3})


def test_unknown_stage_is_rejected_up_front():
    with pytest.raises(ValueError, match="Unknown stage"):
        build_ppo_config(stage="hurricane")


# ------------------------------- the size arithmetic -------------------------------
def test_samples_per_iteration_is_agents_times_env_steps():
    assert samples_per_iteration(50, 1000) == 50_000
    assert samples_per_iteration(8, 100) == 800


@pytest.mark.parametrize("agents, batch, minibatch, epochs, expected", [
    (50, 1000, 1000, 5, 250),     # the defaults
    (50, 200, 200, 2, 100),       # the configuration measured on Ray 2.59.0 (100 updates)
    (8, 100, 100, 2, 16),         # what scripts/rllib_smoke.py prints
    (4, 200, 64, 2, 26),          # measured in the first exploratory run (13 minibatches x 2 epochs)
])
def test_gradient_update_estimate_matches_what_was_measured(agents, batch, minibatch, epochs, expected):
    assert approx_gradient_updates_per_iteration(agents, batch, minibatch, epochs) == expected