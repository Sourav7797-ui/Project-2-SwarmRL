
"""Day 10: Conservative, resumable PPO training for SwarmRL."""

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
    """Return the first finite numeric metric found under the supplied keys."""
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)

    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/day10",
    )
    parser.add_argument(
        "--log-file",
        default="progress.csv",
        help="CSV filename stored inside checkpoint-dir",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing RLlib checkpoint directory",
    )

    args = parser.parse_args()

    if min(args.agents, args.steps, args.iterations, args.cpus) < 1:
        parser.error("agents, steps, iterations and cpus must be positive")

    if args.workers < 0:
        parser.error("workers cannot be negative")

    if args.checkpoint_every < 1:
        parser.error("checkpoint-every must be positive")

    if args.reward_scale <= 0:
        parser.error("reward-scale must be positive")

    if not args.log_file.lower().endswith(".csv"):
        parser.error("--log-file must end in .csv")

    from backend.training.ray_runtime import (
        InsufficientMemoryError,
        available_memory_gb,
        configure_environment,
        init_ray,
    )

    configure_environment()

    import ray
    from backend.training.ppo_config import build_ppo_config

    output_dir = ROOT / args.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / args.log_file

    print(f"SwarmRL Day 10 | Ray {ray.__version__}")
    print(
        f"Agents={args.agents}, episode steps={args.steps}, "
        f"iterations={args.iterations}, workers={args.workers}, "
        f"stage={args.stage}"
    )

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
        config = build_ppo_config(
            num_agents=args.agents,
            max_episode_steps=args.steps,
            stage=args.stage,
            reward_scale=args.reward_scale,
            num_env_runners=args.workers,
            train_batch_size=args.steps,
            minibatch_size=min(args.steps, 32),
            num_epochs=2,
            hidden=(64, 64),
            seed=args.seed,
        )

        start_build = time.time()
        algo = config.build_algo()
        print(f"Algorithm built in {time.time() - start_build:.1f}s")

        if args.resume:
            checkpoint_path = Path(args.resume).resolve()

            if not checkpoint_path.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory does not exist: {checkpoint_path}"
                )

            algo.restore_from_path(str(checkpoint_path))
            print(f"Restored checkpoint: {checkpoint_path}")

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
                    checkpoint = str(getattr(saved, "path", checkpoint_dir))
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
                        "  NOTE: environment-step metric was not found. "
                        "Use --debug-metrics to inspect RLlib metric keys."
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
