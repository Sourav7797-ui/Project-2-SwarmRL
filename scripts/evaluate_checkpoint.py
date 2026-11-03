
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------
# Project paths and existing SwarmRL APIs
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.ray_runtime import (
    InsufficientMemoryError,
    configure_environment,
    init_ray,
)
from backend.training.ppo_config import build_ppo_config
from backend.training.rllib_env import (
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
)


STAGE_NAMES = [
    "static_world",
    "dynamic_obstacles",
    "wind",
]


FIELDS = [
    "timestamp",
    "evaluation_number",
    "episode",
    "checkpoint_path",
    "stage",
    "stage_index",
    "reward_raw",
    "coverage_pct",
    "drone_drone_collisions",
    "drone_obstacle_collisions",
    "total_collision_events",
    "agent_steps",
    "episode_steps",
    "reward_scale",
    "exploration_reward_mode",
    "exploration_voxel_reference",
    "separation_reward_enabled",
    "preferred_separation_m",
    "separation_penalty_scale",
    "min_pair_distance_m",
    "steps_below_preferred_separation",
]


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved SwarmRL RLlib checkpoint and "
            "record rewards, coverage, collisions, separation, "
            "and evaluation stage."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the actual RLlib checkpoint directory.",
    )

    parser.add_argument(
        "--checkpoint-stage",
        choices=STAGE_NAMES,
        default="static_world",
        help="Stage used when the checkpoint was trained.",
    )

    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        default="static_world",
        help="Environment stage used during evaluation.",
    )

    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--exploration-reward-mode",
        choices=["binary", "voxel_log"],
        default="binary",
        help="Exploration reward calculation mode.",
    )

    parser.add_argument(
        "--exploration-voxel-reference",
        type=float,
        default=5.0,
        help="Reference voxel count for voxel_log reward mode.",
    )

    # Day 16: separation reward configuration
    parser.add_argument(
        "--separation-reward-enabled",
        action="store_true",
        help="Enable soft drone-separation reward shaping.",
    )

    parser.add_argument(
        "--preferred-separation",
        type=float,
        default=4.0,
        help="Preferred distance between drones in metres.",
    )

    parser.add_argument(
        "--separation-penalty-scale",
        type=float,
        default=-1.0,
        help="Negative reward scale for insufficient separation.",
    )

    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--output",
        default="checkpoints/day10/evaluation.csv",
        help="CSV path for evaluation results.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------

def read_existing_rows(path: Path) -> int:
    """Return the number of existing evaluation records."""

    if not path.exists():
        return 0

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return max(sum(1 for _ in csv.DictReader(file)), 0)


def append_result(
    writer: csv.DictWriter,
    *,
    evaluation_number: int,
    episode: int,
    checkpoint: Path,
    stage: str,
    reward_raw: float,
    coverage_pct: float,
    drone_collisions: int,
    obstacle_collisions: int,
    agent_steps: int,
    episode_steps: int,
    reward_scale: float,
    exploration_reward_mode: str,
    exploration_voxel_reference: float,
    separation_reward_enabled: bool,
    preferred_separation: float,
    separation_penalty_scale: float,
    min_pair_distance: float,
    steps_below_preferred_separation: int,
) -> None:
    """Append one episode's evaluation metrics to the CSV."""

    writer.writerow(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation_number": evaluation_number,
            "episode": episode,
            "checkpoint_path": str(checkpoint),
            "stage": stage,
            "stage_index": STAGE_NAMES.index(stage),
            "reward_raw": reward_raw,
            "coverage_pct": coverage_pct,
            "drone_drone_collisions": drone_collisions,
            "drone_obstacle_collisions": obstacle_collisions,
            "total_collision_events": (
                drone_collisions + obstacle_collisions
            ),
            "agent_steps": agent_steps,
            "episode_steps": episode_steps,
            "reward_scale": reward_scale,
            "exploration_reward_mode": exploration_reward_mode,
            "exploration_voxel_reference": exploration_voxel_reference,
            "separation_reward_enabled": separation_reward_enabled,
            "preferred_separation_m": preferred_separation,
            "separation_penalty_scale": separation_penalty_scale,
            "min_pair_distance_m": min_pair_distance,
            "steps_below_preferred_separation": (
                steps_below_preferred_separation
            ),
        }
    )


# ---------------------------------------------------------
# Drone separation metrics
# ---------------------------------------------------------

def calculate_min_pair_distance(positions: np.ndarray) -> float:
    """
    Return the minimum 3D distance between distinct drones.

    Expected input shape:
        (number_of_drones, 3)

    Returns infinity when fewer than two drones are present.
    """

    positions = np.asarray(positions, dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Expected drone positions with shape (N, 3), "
            f"got {positions.shape}."
        )

    if len(positions) < 2:
        return float("inf")

    differences = (
        positions[:, None, :] - positions[None, :, :]
    )

    distances = np.linalg.norm(differences, axis=-1)

    # Ignore each drone's distance from itself.
    np.fill_diagonal(distances, np.inf)

    return float(np.min(distances))


# ---------------------------------------------------------
# RLModule action selection for RLlib's new API stack
# ---------------------------------------------------------

def select_action(
    module: Any,
    observation: Any,
) -> Any:
    """
    Produce one deterministic action from the restored RLModule.

    Supports standard flat numeric observations and the default
    RLlib action distribution interface.
    """

    observation_array = np.asarray(
        observation,
        dtype=np.float32,
    )

    observation_tensor = torch.as_tensor(
        observation_array,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        output = module.forward_inference(
            {"obs": observation_tensor}
        )

        # Some RLModule implementations return actions directly.
        if "actions" in output:
            action_tensor = output["actions"]

        else:
            if "action_dist_inputs" not in output:
                raise RuntimeError(
                    "RLModule inference output contains neither "
                    "'actions' nor 'action_dist_inputs'. "
                    f"Available keys: {list(output.keys())}"
                )

            distribution_class = (
                module.get_inference_action_dist_cls()
            )

            distribution = distribution_class.from_logits(
                output["action_dist_inputs"]
            )

            action_tensor = (
                distribution.to_deterministic().sample()
            )

    if isinstance(action_tensor, torch.Tensor):
        action_array = (
            action_tensor.detach().cpu().numpy()
        )
    else:
        action_array = np.asarray(action_tensor)

    # Remove the batch dimension without collapsing legitimate
    # dimensions of the environment's action.
    if (
        action_array.ndim > 0
        and action_array.shape[0] == 1
    ):
        action_array = action_array[0]

    if action_array.ndim == 0:
        return action_array.item()

    return action_array


# ---------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------

def main() -> int:

    args = parse_args()

    # -----------------------------------------------------
    # Basic argument validation
    # -----------------------------------------------------

    if args.agents < 1:
        print(
            "ERROR: --agents must be at least 1.",
            file=sys.stderr,
        )
        return 2

    if args.steps < 1:
        print(
            "ERROR: --steps must be at least 1.",
            file=sys.stderr,
        )
        return 2

    if args.episodes < 1:
        print(
            "ERROR: --episodes must be at least 1.",
            file=sys.stderr,
        )
        return 2

    if (
        args.reward_scale <= 0
        or not math.isfinite(args.reward_scale)
    ):
        print(
            "ERROR: --reward-scale must be a finite value "
            "greater than zero.",
            file=sys.stderr,
        )
        return 2

    if (
        args.exploration_voxel_reference <= 0
        or not math.isfinite(
            args.exploration_voxel_reference
        )
    ):
        print(
            "ERROR: --exploration-voxel-reference must be "
            "a finite value greater than zero.",
            file=sys.stderr,
        )
        return 2

    # -----------------------------------------------------
    # Day 16: separation parameter validation
    # -----------------------------------------------------

    if (
        not math.isfinite(args.preferred_separation)
        or args.preferred_separation <= 2.0
    ):
        print(
            "ERROR: --preferred-separation must be greater "
            "than the hard collision distance of 2.0 metres.",
            file=sys.stderr,
        )
        return 2

    if (
        not math.isfinite(args.separation_penalty_scale)
        or args.separation_penalty_scale > 0
    ):
        print(
            "ERROR: --separation-penalty-scale must be finite "
            "and less than or equal to zero.",
            file=sys.stderr,
        )
        return 2

    # -----------------------------------------------------
    # Checkpoint validation
    # -----------------------------------------------------

    checkpoint = (
        Path(args.checkpoint)
        .expanduser()
        .resolve()
    )

    if not checkpoint.is_dir():
        print(
            f"ERROR: checkpoint directory not found: {checkpoint}",
            file=sys.stderr,
        )
        return 2

    if not (
        checkpoint / "algorithm_state.pkl"
    ).is_file():
        print(
            "ERROR: algorithm_state.pkl was not found directly "
            "inside the supplied checkpoint directory:\n"
            f"{checkpoint}\n\n"
            "For your checkpoint, the path should end with "
            "'iteration_0001', not its parent directory.",
            file=sys.stderr,
        )
        return 2

    # -----------------------------------------------------
    # Output CSV preparation
    # -----------------------------------------------------

    output_path = Path(args.output).expanduser()

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    algo = None
    env = None
    ray_started = False

    try:

        # -------------------------------------------------
        # Initialize Ray
        # -------------------------------------------------

        configure_environment()
        init_ray()
        ray_started = True

        print("\n========== EVALUATION CONFIGURATION ==========")
        print(f"Checkpoint: {checkpoint}")
        print(
            f"Checkpoint training stage: "
            f"{args.checkpoint_stage}"
        )
        print(f"Evaluation stage: {args.stage}")
        print(f"Agents: {args.agents}")
        print(f"Steps per episode: {args.steps}")
        print(f"Episodes: {args.episodes}")
        print(f"Reward scale: {args.reward_scale}")

        print(
            f"Exploration reward mode: "
            f"{args.exploration_reward_mode}"
        )

        print(
            "Exploration voxel reference: "
            f"{args.exploration_voxel_reference}"
        )

        print(
            "Separation reward enabled: "
            f"{args.separation_reward_enabled}"
        )

        print(
            f"Preferred separation: "
            f"{args.preferred_separation} m"
        )

        print(
            f"Separation penalty scale: "
            f"{args.separation_penalty_scale}"
        )

        print(f"Output CSV: {output_path}")
        print("==============================================\n")

        # -------------------------------------------------
        # Build PPO configuration
        # -------------------------------------------------

        config = build_ppo_config(
            num_agents=args.agents,
            max_episode_steps=args.steps,
            stage=args.checkpoint_stage,
            reward_scale=args.reward_scale,
            env_overrides={
                "exploration_reward_mode": (
                    args.exploration_reward_mode
                ),
                "exploration_voxel_reference": (
                    args.exploration_voxel_reference
                ),
                "separation_reward_enabled": (
                    args.separation_reward_enabled
                ),
                "preferred_separation": (
                    args.preferred_separation
                ),
                "separation_penalty_scale": (
                    args.separation_penalty_scale
                ),
            },
            num_env_runners=args.workers,
            train_batch_size=args.steps,
            minibatch_size=min(args.steps, 32),
            num_epochs=2,
            hidden=(64, 64),
            seed=args.seed,
        )

        algo = config.build_algo()

        # -------------------------------------------------
        # Restore checkpoint
        # -------------------------------------------------

        print("\nRestoring checkpoint...")

        algo.restore_from_path(str(checkpoint))

        print("Checkpoint restored.")

        effective_env_config = dict(
            algo.config.env_config
        )

        actual_mode = effective_env_config.get(
            "exploration_reward_mode"
        )

        actual_reference = effective_env_config.get(
            "exploration_voxel_reference"
        )

        print(
            "Restored algorithm environment configuration: "
            f"mode={actual_mode}, "
            f"reference={actual_reference}"
        )

        if actual_mode != args.exploration_reward_mode:
            raise RuntimeError(
                "Restored algorithm reward-mode mismatch: "
                f"expected {args.exploration_reward_mode!r}, "
                f"got {actual_mode!r}."
            )

        if (
            actual_reference is None
            or not math.isclose(
                float(actual_reference),
                args.exploration_voxel_reference,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "Restored algorithm voxel-reference mismatch: "
                f"expected {args.exploration_voxel_reference}, "
                f"got {actual_reference!r}."
            )
                # Verify restored Day 16 separation configuration.
        actual_separation_enabled = effective_env_config.get(
            "separation_reward_enabled"
        )
        actual_preferred_separation = effective_env_config.get(
            "preferred_separation"
        )
        actual_separation_scale = effective_env_config.get(
            "separation_penalty_scale"
        )

        if actual_separation_enabled != args.separation_reward_enabled:
            raise RuntimeError(
                "Restored algorithm separation-enabled mismatch: "
                f"expected {args.separation_reward_enabled!r}, "
                f"got {actual_separation_enabled!r}."
            )

        if (
            actual_preferred_separation is None
            or not math.isclose(
                float(actual_preferred_separation),
                args.preferred_separation,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "Restored algorithm preferred-separation mismatch: "
                f"expected {args.preferred_separation}, "
                f"got {actual_preferred_separation!r}."
            )

        if (
            actual_separation_scale is None
            or not math.isclose(
                float(actual_separation_scale),
                args.separation_penalty_scale,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                "Restored algorithm separation-penalty-scale mismatch: "
                f"expected {args.separation_penalty_scale}, "
                f"got {actual_separation_scale!r}."
            )

        print(
            "Restored separation configuration verified: "
            f"enabled={actual_separation_enabled}, "
            f"preferred={actual_preferred_separation} m, "
            f"scale={actual_separation_scale}"
        )

        # -------------------------------------------------
        # Obtain restored RLModule
        # -------------------------------------------------

        module = algo.get_module(
            module_id=SHARED_POLICY_ID
        )

        if module is None:
            raise RuntimeError(
                "The restored algorithm did not expose the "
                f"expected RLModule: {SHARED_POLICY_ID!r}. "
                "Inspect the checkpoint's RLModule IDs."
            )

        if hasattr(module, "eval"):
            module.eval()

        # -------------------------------------------------
        # Create direct evaluation environment
        # -------------------------------------------------

        env = SwarmRLlibEnv(
            {
                "num_agents": args.agents,
                "max_episode_steps": args.steps,
                "stage": args.stage,
                "reward_scale": args.reward_scale,
                "exploration_reward_mode": (
                    args.exploration_reward_mode
                ),
                "exploration_voxel_reference": (
                    args.exploration_voxel_reference
                ),
                "separation_reward_enabled": (
                    args.separation_reward_enabled
                ),
                "preferred_separation": (
                    args.preferred_separation
                ),
                "separation_penalty_scale": (
                    args.separation_penalty_scale
                ),
            }
        )

        # -------------------------------------------------
        # CSV initialization
        # -------------------------------------------------

        prior_rows = read_existing_rows(output_path)

        next_evaluation_number = prior_rows + 1

        write_header = (
            not output_path.exists()
            or output_path.stat().st_size == 0
        )

        # -------------------------------------------------
        # Evaluate episodes
        # -------------------------------------------------

        with output_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=FIELDS,
                extrasaction="ignore",
            )

            if write_header:
                writer.writeheader()

            for episode in range(
                1,
                args.episodes + 1,
            ):

                print(
                    f"\nEvaluation episode "
                    f"{episode}/{args.episodes}"
                )

                obs, _ = env.reset(
                    seed=args.seed + episode - 1
                )

                drone_collisions = 0
                obstacle_collisions = 0
                agent_steps = 0
                episode_steps = 0
                scaled_return_sum = 0.0

                # Day 16: episode separation metrics
                min_pair_distance = float("inf")

                steps_below_preferred_separation = 0

                # -----------------------------------------
                # Episode simulation
                # -----------------------------------------

                while True:

                    actions = {}

                    for agent_id, observation in obs.items():

                        actions[agent_id] = select_action(
                            module,
                            observation,
                        )

                    (
                        obs,
                        rewards,
                        terminateds,
                        truncateds,
                        infos,
                    ) = env.step(actions)

                    episode_steps += 1

                    scaled_return_sum += sum(
                        float(value)
                        for value in rewards.values()
                    )

                    # -------------------------------------
                    # Day 16: measure drone separation
                    # -------------------------------------

                    positions = getattr(
                        env.env,
                        "positions",
                        None,
                    )

                    if positions is None:
                        raise RuntimeError(
                            "Could not read drone positions "
                            "from env.env.positions."
                        )

                    current_min_distance = (
                        calculate_min_pair_distance(
                            positions
                        )
                    )

                    min_pair_distance = min(
                        min_pair_distance,
                        current_min_distance,
                    )

                    if (
                        current_min_distance
                        < args.preferred_separation
                    ):
                        steps_below_preferred_separation += 1

                    # -------------------------------------
                    # Collision metrics
                    # -------------------------------------

                    for info in infos.values():

                        if not isinstance(info, dict):
                            continue

                        drone_collisions += int(
                            bool(
                                info.get(
                                    "collision_event",
                                    False,
                                )
                            )
                        )

                        obstacle_collisions += int(
                            bool(
                                info.get(
                                    "obstacle_collision_event",
                                    False,
                                )
                            )
                        )

                    agent_steps += len(infos)

                    if (
                        terminateds.get("__all__", False)
                        or truncateds.get("__all__", False)
                    ):
                        break

                # -----------------------------------------
                # Calculate episode return
                # -----------------------------------------

                raw_return = getattr(
                    env,
                    "raw_episode_return",
                    None,
                )

                if raw_return is None:
                    raw_return = (
                        scaled_return_sum / args.reward_scale
                    )

                raw_return = float(raw_return)

                # -----------------------------------------
                # Calculate coverage
                # -----------------------------------------

                inner_env = getattr(
                    env,
                    "env",
                    None,
                )

                coverage_grid = getattr(
                    inner_env,
                    "coverage_grid",
                    None,
                )

                if coverage_grid is None:
                    raise RuntimeError(
                        "Could not read env.env.coverage_grid. "
                        "Please verify the SwarmRL environment's "
                        "coverage API before interpreting results."
                    )

                coverage_pct = float(
                    np.asarray(
                        coverage_grid
                    ).mean() * 100.0
                )

                # -----------------------------------------
                # Save episode result
                # -----------------------------------------

                append_result(
                    writer,
                    evaluation_number=next_evaluation_number,
                    episode=episode,
                    checkpoint=checkpoint,
                    stage=args.stage,
                    reward_raw=raw_return,
                    coverage_pct=coverage_pct,
                    drone_collisions=drone_collisions,
                    obstacle_collisions=obstacle_collisions,
                    agent_steps=agent_steps,
                    episode_steps=episode_steps,
                    reward_scale=args.reward_scale,
                    exploration_reward_mode=(
                        args.exploration_reward_mode
                    ),
                    exploration_voxel_reference=(
                        args.exploration_voxel_reference
                    ),
                    separation_reward_enabled=(
                        args.separation_reward_enabled
                    ),
                    preferred_separation=(
                        args.preferred_separation
                    ),
                    separation_penalty_scale=(
                        args.separation_penalty_scale
                    ),
                    min_pair_distance=min_pair_distance,
                    steps_below_preferred_separation=(
                        steps_below_preferred_separation
                    ),
                )

                csv_file.flush()

                # -----------------------------------------
                # Print episode summary
                # -----------------------------------------

                print(
                    f"Reward: {raw_return:.3f} | "
                    f"Coverage: {coverage_pct:.2f}% | "
                    f"Drone collisions: {drone_collisions} | "
                    f"Obstacle collisions: {obstacle_collisions} | "
                    f"Minimum drone distance: "
                    f"{min_pair_distance:.3f} m | "
                    f"Steps below preferred distance: "
                    f"{steps_below_preferred_separation} | "
                    f"Steps: {episode_steps}"
                )

                next_evaluation_number += 1

        print("\nEvaluation completed.")
        print(f"Results saved to: {output_path}")

        return 0

    # -----------------------------------------------------
    # Exception handling
    # -----------------------------------------------------

    except InsufficientMemoryError as exc:

        print(
            f"ERROR: Insufficient memory for evaluation: {exc}",
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:

        print(
            "\nEvaluation interrupted by user.",
            file=sys.stderr,
        )

        return 130

    except Exception as exc:

        print(
            f"\nERROR during evaluation: {exc}",
            file=sys.stderr,
        )

        import traceback
        traceback.print_exc()

        return 1

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    finally:

        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        if algo is not None:
            try:
                algo.stop()
            except Exception:
                pass

        if ray_started:
            try:
                import ray
                ray.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())