
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

from backend.reward.curriculum import CurriculumManager
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
]


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved SwarmRL RLlib checkpoint and "
            "record rewards, coverage, collisions, and stage."
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
) -> None:
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
        }
    )


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
    observation_array = np.asarray(observation, dtype=np.float32)

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
        action_array = action_tensor.detach().cpu().numpy()
    else:
        action_array = np.asarray(action_tensor)

    # Remove the batch dimension without collapsing legitimate
    # dimensions of the environment's action.
    if action_array.ndim > 0 and action_array.shape[0] == 1:
        action_array = action_array[0]

    if action_array.ndim == 0:
        return action_array.item()

    return action_array


# ---------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------

def main() -> int:
    args = parse_args()

    if args.agents < 1:
        print("ERROR: --agents must be at least 1.", file=sys.stderr)
        return 2

    if args.steps < 1:
        print("ERROR: --steps must be at least 1.", file=sys.stderr)
        return 2

    if args.episodes < 1:
        print("ERROR: --episodes must be at least 1.", file=sys.stderr)
        return 2

    if args.reward_scale <= 0:
        print(
            "ERROR: --reward-scale must be greater than zero.",
            file=sys.stderr,
        )
        return 2

    checkpoint = Path(args.checkpoint).expanduser().resolve()

    if not checkpoint.is_dir():
        print(
            f"ERROR: checkpoint directory not found: {checkpoint}",
            file=sys.stderr,
        )
        return 2

    if not (checkpoint / "algorithm_state.pkl").is_file():
        print(
            "ERROR: algorithm_state.pkl was not found directly inside "
            f"the supplied checkpoint directory:\n{checkpoint}\n\n"
            "For your checkpoint, the path should end with "
            "'iteration_0001', not its parent directory.",
            file=sys.stderr,
        )
        return 2

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    algo = None
    env = None
    ray_started = False

    try:
        # Keep the project's existing runtime setup.
        configure_environment()
        init_ray()
        ray_started = True

        print(f"Checkpoint: {checkpoint}")
        print(f"Checkpoint training stage: {args.checkpoint_stage}")
        print(f"Evaluation stage: {args.stage}")
        print(f"Agents: {args.agents}")
        print(f"Steps per episode: {args.steps}")
        print(f"Episodes: {args.episodes}")
        print(f"Output CSV: {output_path}")

        # Construct the configuration using the project's existing API.
        config = build_ppo_config(
            num_agents=args.agents,
            max_episode_steps=args.steps,
            stage=args.checkpoint_stage,
            reward_scale=args.reward_scale,
            num_env_runners=args.workers,
            train_batch_size=args.steps,
            minibatch_size=min(args.steps, 32),
            num_epochs=2,
            hidden=(64, 64),
            seed=args.seed,
        )

        algo = config.build_algo()

        print("\nRestoring checkpoint...")
        algo.restore_from_path(str(checkpoint))
        print("Checkpoint restored.")

        # New API stack: use the RLModule rather than
        # algo.compute_single_action(), which may rely on legacy policies.
        module = algo.get_module(module_id=SHARED_POLICY_ID)

        if module is None:
            raise RuntimeError(
                "The restored algorithm did not expose the expected "
                f"RLModule: {SHARED_POLICY_ID!r}. "
                "Inspect the checkpoint's RLModule IDs."
            )

        if hasattr(module, "eval"):
            module.eval()

        env = SwarmRLlibEnv(
            {
                "num_agents": args.agents,
                "max_episode_steps": args.steps,
                "stage": args.stage,
                "reward_scale": args.reward_scale,
            }
        )

        prior_rows = read_existing_rows(output_path)
        next_evaluation_number = prior_rows + 1

        write_header = (
            not output_path.exists()
            or output_path.stat().st_size == 0
        )

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

            for episode in range(1, args.episodes + 1):
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

                # Use the environment's raw return when available.
                raw_return = getattr(env, "raw_episode_return", None)

                if raw_return is None:
                    raw_return = (
                        scaled_return_sum / args.reward_scale
                    )

                raw_return = float(raw_return)

                inner_env = getattr(env, "env", None)
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
                    np.asarray(coverage_grid).mean() * 100.0
                )

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
                )

                csv_file.flush()

                print(
                    f"Reward: {raw_return:.3f} | "
                    f"Coverage: {coverage_pct:.2f}% | "
                    f"Drone collisions: {drone_collisions} | "
                    f"Obstacle collisions: {obstacle_collisions} | "
                    f"Steps: {episode_steps}"
                )

                next_evaluation_number += 1

        print("\nEvaluation completed.")
        print(f"Results saved to: {output_path}")
        return 0

    except InsufficientMemoryError as exc:
        print(
            f"ERROR: Insufficient memory for evaluation: {exc}",
            file=sys.stderr,
        )
        return 1

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(
            f"\nERROR during evaluation: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        return 1

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
