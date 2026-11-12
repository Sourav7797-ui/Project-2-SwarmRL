
"""Compare MAPPO policy outputs from iterations 5 and 10.

Read-only diagnostic:
- Collects 40 reset-only observations using the existing environment wrapper.
- Reads iteration 5 and iteration 10 module-state files.
- Loads the extracted parameters into a freshly constructed RLModule.
- Compares Gaussian means, standard deviations, entropy and action bounds.

It does NOT:
- Initialize Ray.
- Construct an RLlib Algorithm.
- Restore an RLlib checkpoint.
- Step or train the environment.
- Run evaluation episodes.
- Modify checkpoints or existing project files.
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from numpy._core.numeric import _frombuffer

# Ensure the project root is importable when run as a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ray.rllib.core.columns import Columns

from backend.training.mappo_config import build_mappo_config
from backend.training.rllib_env import (
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
)


DEFAULT_RUN = Path(
    "checkpoints/mappo_validation/run_20261108_002340_785217"
)

NUM_AGENTS = 4
OBSERVATION_SEEDS = list(range(42, 52))
HIDDEN_LAYERS = (64, 64)

RELATIVE_EPSILON = 1e-12


class RestrictedCheckpointUnpickler(pickle.Unpickler):
    """Allow only globals observed in the module-state pickle format."""

    ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"): OrderedDict,
        ("numpy", "dtype"): np.dtype,
        ("numpy._core.numeric", "_frombuffer"): _frombuffer,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)

        if key not in self.ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Blocked unexpected pickle global: {module}.{name}"
            )

        return self.ALLOWED_GLOBALS[key]


def checkpoint_state_path(run_dir: Path, iteration: int) -> Path:
    return (
        run_dir
        / f"iteration_{iteration:04d}"
        / "learner_group"
        / "learner"
        / "rl_module"
        / "shared_policy"
        / "module_state.pkl"
    )


def load_module_state(path: Path) -> dict[str, np.ndarray]:
    """Read module parameters using the restricted unpickler."""

    with path.open("rb") as handle:
        state = RestrictedCheckpointUnpickler(handle).load()

    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(
            f"Unexpected module state type: {type(state)}"
        )

    result = {}

    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError(
                f"Unexpected parameter name type: {type(name)}"
            )

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Parameter {name!r} is not a NumPy array: "
                f"{type(value)}"
            )

        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(
                f"Parameter {name!r} is not floating point: {value.dtype}"
            )

        if not np.isfinite(value).all():
            raise ValueError(
                f"Parameter {name!r} contains NaN or infinity."
            )

        result[name] = value

    return result


def collect_initial_observations(config):
    """Collect scaled observations through reset only."""

    env = SwarmRLlibEnv(config.env_config)

    local_batches = []
    global_batches = []

    try:
        for seed in OBSERVATION_SEEDS:
            observations, _ = env.reset(seed=seed)

            for agent_id in env.possible_agents:
                agent_obs = observations[agent_id]

                local_batches.append(agent_obs["local_obs"])
                global_batches.append(agent_obs["global_state"])

    finally:
        env.close()

    local_array = np.asarray(local_batches, dtype=np.float32)
    global_array = np.asarray(global_batches, dtype=np.float32)

    if local_array.shape != (40, 26):
        raise ValueError(
            f"Unexpected local observation shape: {local_array.shape}"
        )

    if not np.isfinite(local_array).all():
        raise ValueError("Local observations contain NaN or infinity.")

    if not np.isfinite(global_array).all():
        raise ValueError("Global observations contain NaN or infinity.")

    return local_array, global_array


def load_parameters_into_module(module, state: dict[str, np.ndarray]):
    """Validate names/shapes and load extracted parameters into the module."""

    current_state = module.state_dict()

    if current_state.keys() != state.keys():
        missing = sorted(set(current_state) - set(state))
        unexpected = sorted(set(state) - set(current_state))

        raise ValueError(
            "Checkpoint/module parameter names do not match.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    tensors = {}

    for name, current_tensor in current_state.items():
        array = state[name]

        if tuple(array.shape) != tuple(current_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"checkpoint={array.shape}, "
                f"module={tuple(current_tensor.shape)}"
            )

        tensors[name] = torch.as_tensor(
            array,
            dtype=current_tensor.dtype,
            device=current_tensor.device,
        )

    module.load_state_dict(tensors, strict=True)


def gaussian_entropy(stds: torch.Tensor) -> torch.Tensor:
    """Differential entropy for each independent Gaussian dimension."""

    return 0.5 * torch.log(
        2.0 * torch.pi * torch.e * stds.square()
    )


def gaussian_probability_outside_bounds(
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """Analytical probability that Gaussian samples fall outside [-1, 1]."""

    sqrt_two = np.sqrt(2.0)

    lower_z = (-1.0 - means) / stds
    upper_z = (1.0 - means) / stds

    lower_cdf = 0.5 * (
        1.0 + torch.erf(lower_z / sqrt_two)
    )

    upper_cdf = 0.5 * (
        1.0 + torch.erf(upper_z / sqrt_two)
    )

    return lower_cdf + (1.0 - upper_cdf)


def get_policy_outputs(
    module,
    local_array: np.ndarray,
    global_array: np.ndarray,
):
    """Run inference-only forward pass on fixed observations."""

    batch = {
        Columns.OBS: {
            "local_obs": torch.from_numpy(local_array),
            "global_state": torch.from_numpy(global_array),
        }
    }

    module.eval()

    with torch.no_grad():
        output = module._forward_train(batch)

        distribution_inputs = output[Columns.ACTION_DIST_INPUTS]

        means, log_stds = torch.split(
            distribution_inputs,
            module.action_dim,
            dim=-1,
        )

        log_stds = torch.clamp(
            log_stds,
            min=module.MIN_LOG_STD,
            max=module.MAX_LOG_STD,
        )

        stds = torch.exp(log_stds)

        entropy = gaussian_entropy(stds)

        outside_probability = gaussian_probability_outside_bounds(
            means,
            stds,
        )

    outputs = {
        "means": means.cpu().numpy().copy(),
        "log_stds": log_stds.cpu().numpy().copy(),
        "stds": stds.cpu().numpy().copy(),
        "entropy": entropy.cpu().numpy().copy(),
        "outside_probability": outside_probability.cpu().numpy().copy(),
    }

    for name, values in outputs.items():
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in policy output: {name}"
            )

    return outputs


def summarize_policy(label: str, outputs: dict) -> None:
    means = outputs["means"]
    stds = outputs["stds"]
    entropy = outputs["entropy"]
    outside = outputs["outside_probability"]

    print()
    print("=" * 88)
    print(label)
    print("=" * 88)

    print(
        f"{'Action':<10} {'Mean μ':>12} {'Mean σ':>12} "
        f"{'Std μ':>12} {'Entropy':>12} {'P(outside)':>14}"
    )
    print("-" * 88)

    for action_index in range(means.shape[1]):
        print(
            f"{action_index + 1:<10} "
            f"{np.mean(means[:, action_index]):>12.6f} "
            f"{np.std(means[:, action_index]):>12.6f} "
            f"{np.mean(stds[:, action_index]):>12.6f} "
            f"{np.mean(entropy[:, action_index]):>12.6f} "
            f"{100.0 * np.mean(outside[:, action_index]):>13.4f}%"
        )

    deterministic_outside = np.abs(means) > 1.0

    print("-" * 88)
    print(
        "Deterministic means outside [-1, 1]: "
        f"{100.0 * np.mean(deterministic_outside):.4f}%"
    )
    print(
        "Mean Gaussian entropy per action vector: "
        f"{np.mean(np.sum(entropy, axis=1)):.6f}"
    )


def symmetric_gaussian_kl(
    means_a: np.ndarray,
    stds_a: np.ndarray,
    means_b: np.ndarray,
    stds_b: np.ndarray,
):
    """Calculate both directions of diagonal-Gaussian KL divergence."""

    variance_a = stds_a ** 2
    variance_b = stds_b ** 2

    kl_a_b = (
        np.log(stds_b / stds_a)
        + (
            variance_a + (means_a - means_b) ** 2
        ) / (2.0 * variance_b)
        - 0.5
    )

    kl_b_a = (
        np.log(stds_a / stds_b)
        + (
            variance_b + (means_b - means_a) ** 2
        ) / (2.0 * variance_a)
        - 0.5
    )

    return kl_a_b, kl_b_a


def compare_outputs(
    iteration_5: dict,
    iteration_10: dict,
) -> None:
    means_5 = iteration_5["means"]
    means_10 = iteration_10["means"]

    stds_5 = iteration_5["stds"]
    stds_10 = iteration_10["stds"]

    entropy_5 = iteration_5["entropy"]
    entropy_10 = iteration_10["entropy"]

    outside_5 = iteration_5["outside_probability"]
    outside_10 = iteration_10["outside_probability"]

    mean_difference = means_10 - means_5
    std_difference = stds_10 - stds_5

    kl_5_10, kl_10_5 = symmetric_gaussian_kl(
        means_5,
        stds_5,
        means_10,
        stds_10,
    )

    print()
    print("=" * 104)
    print("POLICY OUTPUT CHANGE: ITERATION 5 -> ITERATION 10")
    print("=" * 104)

    print(
        f"{'Action':<9} {'Mean |Δμ|':>13} {'RMS Δμ':>13} "
        f"{'Max |Δμ|':>13} {'Mean |Δσ|':>13} "
        f"{'Entropy Δ':>13} {'KL 5->10':>13} {'KL 10->5':>13}"
    )
    print("-" * 104)

    for action_index in range(means_5.shape[1]):
        delta_mu = mean_difference[:, action_index]
        delta_std = std_difference[:, action_index]

        print(
            f"{action_index + 1:<9} "
            f"{np.mean(np.abs(delta_mu)):>13.7f} "
            f"{np.sqrt(np.mean(delta_mu ** 2)):>13.7f} "
            f"{np.max(np.abs(delta_mu)):>13.7f} "
            f"{np.mean(np.abs(delta_std)):>13.7f} "
            f"{np.mean(entropy_10[:, action_index] - entropy_5[:, action_index]):>13.7f} "
            f"{np.mean(kl_5_10[:, action_index]):>13.7f} "
            f"{np.mean(kl_10_5[:, action_index]):>13.7f}"
        )

    print("-" * 104)

    print(
        "Overall mean absolute action-mean change: "
        f"{np.mean(np.abs(mean_difference)):.8f}"
    )
    print(
        "Overall RMS action-mean change: "
        f"{np.sqrt(np.mean(mean_difference ** 2)):.8f}"
    )
    print(
        "Overall mean absolute standard-deviation change: "
        f"{np.mean(np.abs(std_difference)):.8f}"
    )
    print(
        "Mean Gaussian entropy, iteration 5: "
        f"{np.mean(np.sum(entropy_5, axis=1)):.8f}"
    )
    print(
        "Mean Gaussian entropy, iteration 10: "
        f"{np.mean(np.sum(entropy_10, axis=1)):.8f}"
    )
    print(
        "Mean probability outside [-1, 1], iteration 5: "
        f"{100.0 * np.mean(outside_5):.4f}%"
    )
    print(
        "Mean probability outside [-1, 1], iteration 10: "
        f"{100.0 * np.mean(outside_10):.4f}%"
    )

    print()
    print(
        "Note: Gaussian probabilities are analytical pre-clipping "
        "probabilities, not measured executed-action frequencies."
    )
    print(
        "Note: KL divergence compares the two Gaussian distributions "
        "on the same fixed observations."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MAPPO actor outputs across two checkpoints."
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help="MAPPO training run directory.",
    )

    args = parser.parse_args()

    path_5 = checkpoint_state_path(args.run_dir, 5)
    path_10 = checkpoint_state_path(args.run_dir, 10)

    for path in (path_5, path_10):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("=" * 88)
    print("MAPPO POLICY OUTPUT DIAGNOSTIC")
    print("=" * 88)
    print(f"Run: {args.run_dir}")
    print("Checkpoint comparison: iteration 5 -> iteration 10")
    print("Ray cluster: not initialized")
    print("RLlib Algorithm: not constructed")
    print("Checkpoint restore: not used")
    print("Environment interaction: reset only")
    print("Training/evaluation: not performed")

    print()
    print("Reading iteration 5 module parameters...")
    state_5 = load_module_state(path_5)

    print("Reading iteration 10 module parameters...")
    state_10 = load_module_state(path_10)

    config = build_mappo_config(
        num_agents=NUM_AGENTS,
        max_episode_steps=128,
        hidden=HIDDEN_LAYERS,
        num_env_runners=0,
    )

    print("Collecting fixed reset-only observations...")
    local_array, global_array = collect_initial_observations(config)

    print(f"Local observations: {local_array.shape}")
    print(f"Global states: {global_array.shape}")

    print("Building one fresh RLModule from the existing module specification...")
    module_spec = config.get_multi_rl_module_spec()
    multi_module = module_spec.build()
    module = multi_module[SHARED_POLICY_ID]

    print("Loading iteration 5 parameters...")
    load_parameters_into_module(module, state_5)

    outputs_5 = get_policy_outputs(
        module,
        local_array,
        global_array,
    )

    print("Loading iteration 10 parameters...")
    load_parameters_into_module(module, state_10)

    outputs_10 = get_policy_outputs(
        module,
        local_array,
        global_array,
    )

    summarize_policy("ITERATION 5 POLICY", outputs_5)
    summarize_policy("ITERATION 10 POLICY", outputs_10)

    compare_outputs(outputs_5, outputs_10)

    print()
    print("=" * 88)
    print("DIAGNOSTIC COMPLETE")
    print("No checkpoint files were modified.")
    print("=" * 88)


if __name__ == "__main__":
    main()