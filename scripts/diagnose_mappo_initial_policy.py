
"""Diagnose initial MAPPO Gaussian policy outputs.

This is an inference-only diagnostic.

It does NOT:
- Initialize a Ray cluster.
- Construct an RLlib Algorithm.
- Load checkpoints.
- Step or train the environment.
- Modify existing project files or models.

It builds fresh MAPPO RLModules and evaluates their initial
Gaussian action distributions on reset-only environment observations.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

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


NUM_AGENTS = 4

# These are network initialization seeds, not environment seeds.
INITIALIZATION_SEEDS = [42, 123, 2026, 31415, 7]

# Independent reset seeds provide a collection of initial observations.
OBSERVATION_SEEDS = list(range(42, 52))

HIDDEN_LAYERS = (64, 64)


def set_initialization_seed(seed: int) -> None:
    """Set random generators before constructing a fresh RLModule."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_initial_observations(config):
    """Collect scaled observations using reset only; never call step()."""

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

    if local_array.ndim != 2 or local_array.shape[1] != 26:
        raise ValueError(
            f"Unexpected local observation shape: {local_array.shape}"
        )

    if not np.isfinite(local_array).all():
        raise ValueError("Local observations contain NaN or infinity.")

    if not np.isfinite(global_array).all():
        raise ValueError("Global observations contain NaN or infinity.")

    return local_array, global_array


def gaussian_probability_outside_action_bounds(
    means: torch.Tensor,
    stds: torch.Tensor,
) -> torch.Tensor:
    """Calculate P(X < -1 or X > 1) for each Gaussian action."""

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


def describe(values: np.ndarray) -> str:
    """Compact distribution summary."""

    return (
        f"min={np.min(values):.4f}, "
        f"median={np.median(values):.4f}, "
        f"mean={np.mean(values):.4f}, "
        f"max={np.max(values):.4f}"
    )


def diagnose_module(
    initialization_seed: int,
    module_spec,
    local_array: np.ndarray,
    global_array: np.ndarray,
) -> None:
    """Build one fresh module and measure its initial policy."""

    set_initialization_seed(initialization_seed)

    # Build only the RLModule. No Algorithm or Ray cluster is created.
    multi_module = module_spec.build()
    module = multi_module[SHARED_POLICY_ID]
    module.eval()

    local_tensor = torch.from_numpy(local_array)
    global_tensor = torch.from_numpy(global_array)

    batch = {
        Columns.OBS: {
            "local_obs": local_tensor,
            "global_state": global_tensor,
        }
    }

    with torch.no_grad():
        output = module._forward_train(batch)

        distribution_inputs = output[Columns.ACTION_DIST_INPUTS]

        means, log_stds = torch.split(
            distribution_inputs,
            module.action_dim,
            dim=-1,
        )

        stds = torch.exp(log_stds)

        outside_probabilities = (
            gaussian_probability_outside_action_bounds(
                means,
                stds,
            )
        )

        # Differential entropy of each independent Gaussian dimension.
        entropies = 0.5 * torch.log(
            2.0 * torch.pi * torch.e * stds.square()
        )

    means_np = means.cpu().numpy()
    stds_np = stds.cpu().numpy()
    log_stds_np = log_stds.cpu().numpy()
    outside_np = outside_probabilities.cpu().numpy()
    entropy_np = entropies.cpu().numpy()

    deterministic_outside = np.abs(means_np) > 1.0

    print()
    print("=" * 76)
    print(f"NETWORK INITIALIZATION SEED: {initialization_seed}")
    print("=" * 76)

    print(f"Observation samples: {len(local_array)}")
    print(f"Actor hidden layers: {HIDDEN_LAYERS}")
    print(f"Actor output shape: {distribution_inputs.shape}")

    print()
    print("ACTION DISTRIBUTION")

    for action_index in range(module.action_dim):
        print()
        print(f"Action dimension {action_index + 1}")

        print("  Mean:             ", describe(means_np[:, action_index]))
        print("  Log standard dev: ", describe(log_stds_np[:, action_index]))
        print("  Standard dev:     ", describe(stds_np[:, action_index]))

        print(
            "  Gaussian P(outside [-1,1]): "
            f"{100.0 * np.mean(outside_np[:, action_index]):.2f}%"
        )

        print(
            "  Deterministic means outside [-1,1]: "
            f"{100.0 * np.mean(deterministic_outside[:, action_index]):.2f}%"
        )

        print(
            "  Mean Gaussian entropy: "
            f"{np.mean(entropy_np[:, action_index]):.4f}"
        )

    print()
    print("OVERALL POLICY")

    print(
        "  Total Gaussian entropy per action vector: "
        f"{np.mean(np.sum(entropy_np, axis=1)):.4f}"
    )

    print(
        "  Mean probability outside bounds per action dimension: "
        f"{100.0 * np.mean(outside_np):.2f}%"
    )

    print(
        "  Deterministic means outside bounds: "
        f"{100.0 * np.mean(deterministic_outside):.2f}%"
    )

    print(
        "  All distribution inputs finite: "
        f"{np.isfinite(distribution_inputs.cpu().numpy()).all()}"
    )


def main() -> None:
    print("=" * 76)
    print("MAPPO INITIAL POLICY DIAGNOSTIC")
    print("=" * 76)

    print("Mode: inference only")
    print("Ray cluster: not initialized")
    print("RLlib Algorithm: not constructed")
    print("Checkpoint: none")
    print("Environment steps: zero")
    print(f"Agents: {NUM_AGENTS}")
    print(f"Network seeds: {INITIALIZATION_SEEDS}")
    print(f"Observation reset seeds: {OBSERVATION_SEEDS}")
    print(f"Hidden layers: {HIDDEN_LAYERS}")

    # Construct the same module specification used by training.
    config = build_mappo_config(
        num_agents=NUM_AGENTS,
        max_episode_steps=128,
        hidden=HIDDEN_LAYERS,
        num_env_runners=0,
    )

    module_spec = config.get_multi_rl_module_spec()

    # Gather one fixed observation dataset for fair comparison
    # across all network initialization seeds.
    local_array, global_array = collect_initial_observations(config)

    print()
    print(f"Collected local observations: {local_array.shape}")
    print(f"Collected global states: {global_array.shape}")

    for initialization_seed in INITIALIZATION_SEEDS:
        diagnose_module(
            initialization_seed=initialization_seed,
            module_spec=module_spec,
            local_array=local_array,
            global_array=global_array,
        )

    print()
    print("=" * 76)
    print("DIAGNOSTIC COMPLETE")
    print("No training or evaluation was performed.")
    print("=" * 76)


if __name__ == "__main__":
    main()
