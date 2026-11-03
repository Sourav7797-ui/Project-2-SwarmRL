
"""Day 16: PPO fine-tuning with optional separation reward and checkpoint weights."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.ppo_config import build_ppo_config


def flatten_metrics(data, prefix=""):
    """Flatten nested RLlib result dictionaries into CSV-friendly keys."""
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            yield from flatten_metrics(value, name)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield name, float(value)


def finite_metric(metrics, *keys):
    """Return the first finite numeric metric found under supplied keys."""
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--object-store-mb", type=int, default=100)

    parser.add_argument(
        "--stage",
        choices=["static_world", "dynamic_obstacles", "wind"],
        default="static_world",
    )
    parser.add_argument("--reward-scale", type=float, default=0.05)

    parser.add_argument(
        "--exploration-reward-mode",
        choices=["binary", "voxel_log"],
        default="binary",
    )
    parser.add_argument(
        "--exploration-voxel-reference",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--separation-reward-enabled",
        action="store_true",
        help="Enable soft inter-drone separation reward shaping",
    )
    parser.add_argument(
        "--preferred-separation",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--separation-penalty-scale",
        type=float,
        default=-1.0,
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/day10",
    )
    parser.add_argument(
        "--log-file",
        default="progress.csv",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Existing RLlib checkpoint. Restores learner weights only; "
            "the new algorithm uses the requested environment settings."
        ),
    )

    args = parser.parse_args()

    if min(args.agents, args.steps, args.iterations, args.cpus) < 1:
        parser.error("agents, steps, iterations and cpus must be positive")

    if args.workers < 0:
        parser.error("workers cannot be negative")

    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")

    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error("reward-scale must be finite and positive")

    if (
        not math.isfinite(args.exploration_voxel_reference)
        or args.exploration_voxel_reference <= 0
    ):
        parser.error("exploration-voxel-reference must be finite and positive")

    if (
        not math.isfinite(args.preferred_separation)
        or args.preferred_separation <= 2.0
    ):
        parser.error("preferred-separation must be finite and greater than 2.0m")

    if (
        not math.isfinite(args.separation_penalty_scale)
        or args.separation_penalty_scale > 0
    ):
        parser.error("separation-penalty-scale must be finite and <= 0")

    if not args.log_file.lower().endswith(".csv"):
        parser.error("--log-file must end in .csv")

    resume_path = None
    learner_checkpoint_path = None

    if args.resume:
        resume_path = Path(args.resume).resolve()

        if not resume_path.is_dir():
            parser.error(f"Resume checkpoint directory not found: {resume_path}")

        learner_checkpoint_path = resume_path / "learner_group"

        if not learner_checkpoint_path.is_dir():
            parser.error(
                "The supplied checkpoint does not contain a learner_group "
                f"directory: {learner_checkpoint_path}"
            )

    from backend.training.ray_runtime import (
        InsufficientMemoryError,
        available_memory_gb,
        configure_environment,
        init_ray,
    )

    configure_environment()

    import ray

    output_dir = ROOT / args.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / args.log_file

    print(f"SwarmRL Day 16 | Ray {ray.__version__}")
    print(
        f"Agents={args.agents}, episode steps={args.steps}, "
        f"iterations={args.iterations}, workers={args.workers}, "
        f"stage={args.stage}"
    )
    print(
        "Requested separation configuration: "
        f"enabled={args.separation_reward_enabled}, "
        f"preferred={args.preferred_separation}m, "
        f"scale={args.separation_penalty_scale}"
    )

    if resume_path is not None:
        print(f"Initial learner checkpoint: {resume_path}")
        print("Restore mode: learner weights only; fresh optimizer and iteration.")

    free_gb = available_memory_gb()

    if free_gb is not None:
        print(f"Available RAM before startup: {free_gb:.2f} GB")

        if free_gb < 1.5:
            print(
                "WARNING: low available RAM. Close unnecessary applications "
                "before training."
            )

    try:
        init_ray(
            num_cpus=args.cpus,
            object_store_mb=args.object_store_mb,
        )
    except InsufficientMemoryError as exc:
        print(f"Ray could not start safely: {exc}")
        return 2

    algo = None

    try:
        # Always build a fresh algorithm using the requested environment config.
        config = build_ppo_config(
            num_agents=args.agents,
            max_episode_steps=args.steps,
            stage=args.stage,
            reward_scale=args.reward_scale,
            env_overrides={
                "exploration_reward_mode": args.exploration_reward_mode,
                "exploration_voxel_reference": (
                    args.exploration_voxel_reference
                ),
                "separation_reward_enabled": args.separation_reward_enabled,
                "preferred_separation": args.preferred_separation,
                "separation_penalty_scale": args.separation_penalty_scale,
            },
            num_env_runners=args.workers,
            train_batch_size=args.steps,
            minibatch_size=min(args.steps, 32),
            num_epochs=2,
            hidden=(64, 64),
            seed=args.seed,
        )

        algo = config.build_algo()
        print("Fresh PPO algorithm built with requested environment settings.")

        # Restore only the trained learner component from the older checkpoint.
        if learner_checkpoint_path is not None:
            print("\nRestoring Day 13 learner state...")

            algo.learner_group.restore_from_path(
                learner_checkpoint_path
            )

            # Confirm that the restored learner contains module weights.
            learner_weights = algo.learner_group.get_weights()

            if not learner_weights:
                raise RuntimeError(
                    "Learner restore completed, but get_weights() returned "
                    "an empty state. Refusing to start training."
                )

            print(
                "Learner weights restored. Module IDs: "
                f"{list(learner_weights.keys())}"
            )

            # Synchronize restored learner weights to the environment runner.
            algo.env_runner_group.sync_weights(
                from_worker_or_learner_group=algo.learner_group
            )

            print("Restored learner weights synchronized to environment runner.")

        # Verify the active environment configuration.
        effective_env_config = dict(algo.config.env_config)

        expected_settings = {
            "exploration_reward_mode": args.exploration_reward_mode,
            "exploration_voxel_reference": args.exploration_voxel_reference,
            "separation_reward_enabled": args.separation_reward_enabled,
            "preferred_separation": args.preferred_separation,
            "separation_penalty_scale": args.separation_penalty_scale,
            "num_agents": args.agents,
            "max_episode_steps": args.steps,
            "stage": args.stage,
            "reward_scale": args.reward_scale,
        }

        print("\nActive environment configuration:")

        for key, expected in expected_settings.items():
            actual = effective_env_config.get(key)
            print(f"  {key}: {actual}")

            if isinstance(expected, float):
                matches = (
                    actual is not None
                    and math.isclose(
                        float(actual),
                        expected,
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    )
                )
            else:
                matches = actual == expected

            if not matches:
                raise RuntimeError(
                    f"Environment configuration mismatch for {key}: "
                    f"expected {expected}, got {actual}"
                )

        print("\nEnvironment configuration verification passed.")

        if resume_path is not None:
            print("Checkpoint learner restoration verification passed.")

        fields = [
            "timestamp",
            "iteration",
            "iteration_seconds",
            "sampled_env_steps",
            "episode_return_mean_raw",
            "episode_length_mean",
            "policy_loss",
            "value_loss",
            "entropy",
            "value_explained_variance",
            "checkpoint_path",
        ]

        write_header = (
            not log_path.exists() or log_path.stat().st_size == 0
        )

        with log_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)

            if write_header:
                writer.writeheader()

            for iteration in range(1, args.iterations + 1):
                start = time.time()
                result = algo.train()
                elapsed = time.time() - start

                metrics = dict(flatten_metrics(result))

                scaled_return = finite_metric(
                    metrics, "env_runners/episode_return_mean"
                )

                raw_return = (
                    scaled_return / args.reward_scale
                    if scaled_return is not None
                    else None
                )

                sampled_steps = finite_metric(
                    metrics,
                    "env_runners/num_env_steps_sampled",
                    "env_runners/num_env_steps_sampled_lifetime",
                    "num_env_steps_sampled",
                    "num_env_steps_sampled_lifetime",
                )

                episode_length = finite_metric(
                    metrics,
                    "env_runners/episode_len_mean",
                    "env_runners/episode_length_mean",
                )

                policy_loss = finite_metric(
                    metrics,
                    "learners/shared_policy/policy_loss",
                )
                value_loss = finite_metric(
                    metrics,
                    "learners/shared_policy/vf_loss",
                )
                entropy = finite_metric(
                    metrics,
                    "learners/shared_policy/entropy",
                )
                explained_variance = finite_metric(
                    metrics,
                    "learners/shared_policy/vf_explained_var",
                )

                row = {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "iteration": iteration,
                    "iteration_seconds": round(elapsed, 2),
                    "sampled_env_steps": (
                        int(sampled_steps)
                        if sampled_steps is not None
                        else ""
                    ),
                    "episode_return_mean_raw": (
                        round(raw_return, 4)
                        if raw_return is not None
                        else ""
                    ),
                    "episode_length_mean": (
                        episode_length
                        if episode_length is not None
                        else ""
                    ),
                    "policy_loss": (
                        policy_loss if policy_loss is not None else ""
                    ),
                    "value_loss": (
                        value_loss if value_loss is not None else ""
                    ),
                    "entropy": entropy if entropy is not None else "",
                    "value_explained_variance": (
                        explained_variance
                        if explained_variance is not None
                        else ""
                    ),
                    "checkpoint_path": "",
                }

                checkpoint = None

                if (
                    iteration % args.checkpoint_every == 0
                    or iteration == args.iterations
                ):
                    checkpoint_dir = (
                        output_dir
                        / (
                            "checkpoint_"
                            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        )
                        / f"iteration_{iteration:04d}"
                    )

                    checkpoint_dir.parent.mkdir(
                        parents=True, exist_ok=True
                    )

                    saved = algo.save_to_path(str(checkpoint_dir))
                    checkpoint = str(
                        getattr(saved, "path", checkpoint_dir)
                    )
                    row["checkpoint_path"] = checkpoint

                writer.writerow(row)
                csv_file.flush()

                def show(value, fmt=".4f"):
                    return f"{value:{fmt}}" if value is not None else "n/a"

                steps_text = (
                    str(int(sampled_steps))
                    if sampled_steps is not None
                    else "n/a"
                )

                print(
                    f"Iteration {iteration}/{args.iterations} | "
                    f"{elapsed:.1f}s | "
                    f"env steps={steps_text} | "
                    f"raw return={show(raw_return)} | "
                    f"policy loss={show(policy_loss)} | "
                    f"value loss={show(value_loss)}"
                )

                if checkpoint:
                    print(f"  Checkpoint saved: {checkpoint}")

                if sampled_steps is None:
                    print(
                        "  NOTE: environment-step metric was not found."
                    )

        print(f"\nTraining run finished. CSV log: {log_path}")
        print(
            "A completed run does not by itself prove that the policy "
            "has learned."
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nTraining interrupted. Previously saved checkpoints "
            "remain available."
        )
        return 130

    finally:
        if algo is not None:
            algo.stop()

        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())