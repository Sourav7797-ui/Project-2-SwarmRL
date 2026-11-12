
"""Dedicated MAPPO pilot training script.

Uses the shared decentralized actor and centralized critic configured in
backend.training.mappo_config. This script does not modify the PPO trainer.

Example small pilot:
    python -m scripts.train_mappo

Training starts only when this script is explicitly executed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def flatten_metrics(
    data: Any,
    prefix: str = "",
) -> Iterator[tuple[str, float]]:
    """Flatten numeric RLlib metrics into slash-separated keys."""

    if not isinstance(data, dict):
        return

    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)

        if isinstance(value, dict):
            yield from flatten_metrics(value, name)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            yield name, float(value)


def finite_metric(metrics: dict[str, float], *keys: str):
    """Return the first finite metric found under the supplied keys."""

    for key in keys:
        value = metrics.get(key)

        if value is not None and math.isfinite(value):
            return float(value)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # Environment and pilot duration.
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument(
        "--stage",
        choices=[
            "static_world",
            "dynamic_obstacles",
            "wind",
        ],
        default="static_world",
    )
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=10)

    # MAPPO optimizer settings.
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--seed", type=int, default=42)

    # Ray runtime.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--object-store-mb", type=int, default=100)

    # Output.
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--checkpoint-root",
        default="checkpoints/mappo_pilot",
    )
    parser.add_argument(
        "--results-root",
        default="results/mappo_pilot",
    )

    args = parser.parse_args()

    positive_values = {
        "agents": args.agents,
        "steps": args.steps,
        "iterations": args.iterations,
        "train_batch_size": args.train_batch_size,
        "minibatch_size": args.minibatch_size,
        "epochs": args.epochs,
        "cpus": args.cpus,
        "checkpoint_every": args.checkpoint_every,
    }

    for name, value in positive_values.items():
        if value < 1:
            parser.error(
                f"{name.replace('_', '-')} must be positive"
            )

    if args.workers < 0:
        parser.error("--workers cannot be negative")

    if args.object_store_mb < 80:
        parser.error("--object-store-mb must be at least 80")

    if args.minibatch_size > args.train_batch_size:
        parser.error(
            "--minibatch-size must be <= --train-batch-size "
            "for the current MAPPO configuration."
        )

    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error("--reward-scale must be finite and positive")

    if (
        not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("--learning-rate must be finite and positive")

    if not args.hidden or any(size < 1 for size in args.hidden):
        parser.error("--hidden must contain positive layer sizes")

    if not args.checkpoint_root or not args.results_root:
        parser.error("Output directory arguments cannot be empty")

    return args


def write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write run metadata without changing training state."""

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main() -> int:
    args = parse_args()

    from backend.training.ray_runtime import (
        InsufficientMemoryError,
        available_memory_gb,
        configure_environment,
        init_ray,
    )

    # Set worker environment before Ray initialization.
    configure_environment()

    import ray

    # Each run receives a unique directory. Existing pilot artifacts
    # are never overwritten by this script.
    run_id = datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    checkpoint_dir = ROOT / args.checkpoint_root / f"run_{run_id}"
    results_dir = ROOT / args.results_root / f"run_{run_id}"

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    results_dir.mkdir(parents=True, exist_ok=False)

    csv_path = results_dir / "training_metrics.csv"
    metadata_path = results_dir / "run_config.json"

    # RLlib counts this batch target in environment steps.
    # This is an estimated number of agent transitions, assuming
    # every agent contributes one transition per environment step.
    agent_samples_per_iteration = (
        args.agents * args.train_batch_size
    )

    # Estimated optimizer updates, assuming the learner receives
    # the expected number of agent transitions without padding
    # or filtering changing the effective batch size.
    approximate_updates = (
        math.ceil(
            agent_samples_per_iteration / args.minibatch_size
        )
        * args.epochs
    )

    metadata = {
        "algorithm": "MAPPO",
        "architecture": (
            "shared decentralized actor + centralized critic"
        ),
        "agents": args.agents,
        "max_episode_steps": args.steps,
        "stage": args.stage,
        "reward_scale": args.reward_scale,
        "iterations": args.iterations,
        "train_batch_size": args.train_batch_size,
        "train_batch_counting_unit": "environment_steps",
        "minibatch_size": args.minibatch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "hidden_layers": args.hidden,
        "seed": args.seed,
        "num_env_runners": args.workers,
        "num_cpus": args.cpus,
        "object_store_mb": args.object_store_mb,
        "agent_samples_per_iteration_estimate": (
            agent_samples_per_iteration
        ),
        "approx_optimizer_updates_per_iteration": (
            approximate_updates
        ),
        "checkpoint_every": args.checkpoint_every,
        "run_id": run_id,
        "diagnostic_metrics": [
            "kl_divergence",
            "current_kl_coeff",
            "value_loss_unclipped",
            "gradient_scaler_scale_if_available",
        ],
        "clip_fraction": (
            "not recorded: no confirmed source metric in "
            "the inspected PPO learner loss implementation"
        ),
    }

    # Preserve initial metadata even if Ray startup fails.
    write_metadata(metadata_path, metadata)

    print("\n========== SWARMRL MAPPO PILOT ==========")
    print(f"Ray version: {ray.__version__}")
    print(f"Run ID: {run_id}")
    print(
        f"Agents={args.agents}, episode limit={args.steps}, "
        f"iterations={args.iterations}, stage={args.stage}"
    )
    print(
        f"Train batch={args.train_batch_size} environment steps, "
        f"minibatch={args.minibatch_size}, epochs={args.epochs}"
    )
    print(
        f"Estimated agent samples/iteration="
        f"{agent_samples_per_iteration}"
    )
    print(
        f"Approximate optimizer updates/iteration="
        f"{approximate_updates}"
    )
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Results directory: {results_dir}")

    free_gb = available_memory_gb()

    if free_gb is not None:
        print(f"Available RAM before Ray startup: {free_gb:.2f} GB")

        if free_gb < 1.5:
            print(
                "WARNING: Available RAM is low. "
                "Close unnecessary applications before training."
            )

    algo = None

    fields = [
        "timestamp",
        "iteration",
        "iteration_seconds",
        "sampled_env_steps",
        "episode_return_mean_scaled",
        "episode_return_mean_raw_estimate",
        "episode_length_mean",
        "policy_loss",
        "value_loss",
        "value_loss_unclipped",
        "entropy",
        "value_explained_variance",
        "kl_divergence",
        "current_kl_coeff",
        "gradient_scaler_scale",
        "checkpoint_path",
    ]

    # Ray startup, Algorithm construction, and training all belong
    # inside this cleanup scope. This ensures shutdown is attempted
    # even if Ray startup or Algorithm construction fails.
    try:
        try:
            init_ray(
                num_cpus=args.cpus,
                object_store_mb=args.object_store_mb,
            )
        except InsufficientMemoryError as exc:
            print(f"\nRay startup safely refused: {exc}")
            return 2

        from backend.training.mappo_config import build_mappo_config

        config = build_mappo_config(
            num_agents=args.agents,
            max_episode_steps=args.steps,
            stage=args.stage,
            reward_scale=args.reward_scale,
            num_env_runners=args.workers,
            train_batch_size=args.train_batch_size,
            minibatch_size=args.minibatch_size,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            hidden=tuple(args.hidden),
            seed=args.seed,
        )

        # Record the resolved RLlib optimizer settings.
        # This only updates run metadata; it does not change config.
        metadata.update(
            {
                "use_kl_loss": bool(config.use_kl_loss),
                "kl_coeff_initial": float(config.kl_coeff),
                "kl_target": float(config.kl_target),
                "grad_clip": config.grad_clip,
                "clip_param": float(config.clip_param),
                "entropy_coeff": float(config.entropy_coeff),
            }
        )
        write_metadata(metadata_path, metadata)

        algo = config.build_algo()

        print(
            "\nMAPPO algorithm built successfully "
            "with the requested configuration."
        )
        print("Shared policy ID: shared_policy")
        print("Starting pilot iterations...\n")

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fields,
            )
            writer.writeheader()

            for iteration in range(1, args.iterations + 1):
                start = time.time()
                result = algo.train()
                elapsed = time.time() - start

                metrics = dict(flatten_metrics(result))

                scaled_return = finite_metric(
                    metrics,
                    "env_runners/episode_return_mean",
                )

                # Estimate raw return from RLlib's all-agent scaled
                # episode return. This is not the adapter's direct
                # raw_episode_return measurement.
                raw_return_estimate = (
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

                # PPO learner diagnostic metrics.
                # These exact names were confirmed against the
                # installed RLlib learner source.
                kl_divergence = finite_metric(
                    metrics,
                    "learners/shared_policy/mean_kl_loss",
                )

                current_kl_coeff = finite_metric(
                    metrics,
                    "learners/shared_policy/curr_kl_coeff",
                )

                value_loss_unclipped = finite_metric(
                    metrics,
                    "learners/shared_policy/vf_loss_unclipped",
                )

                gradient_scaler_scale = finite_metric(
                    metrics,
                    "learners/shared_policy/_torch_grad_scaler_current_scale",
                )

                saved_checkpoint = ""

                if (
                    iteration % args.checkpoint_every == 0
                    or iteration == args.iterations
                ):
                    target = (
                        checkpoint_dir
                        / f"iteration_{iteration:04d}"
                    )

                    saved = algo.save_to_path(str(target))
                    saved_checkpoint = str(
                        getattr(saved, "path", saved or target)
                    )

                row = {
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="seconds"
                    ),
                    "iteration": iteration,
                    "iteration_seconds": round(elapsed, 3),
                    "sampled_env_steps": (
                        int(sampled_steps)
                        if sampled_steps is not None
                        else ""
                    ),
                    "episode_return_mean_scaled": (
                        scaled_return
                        if scaled_return is not None
                        else ""
                    ),
                    "episode_return_mean_raw_estimate": (
                        raw_return_estimate
                        if raw_return_estimate is not None
                        else ""
                    ),
                    "episode_length_mean": (
                        episode_length
                        if episode_length is not None
                        else ""
                    ),
                    "policy_loss": (
                        policy_loss
                        if policy_loss is not None
                        else ""
                    ),
                    "value_loss": (
                        value_loss
                        if value_loss is not None
                        else ""
                    ),
                    "value_loss_unclipped": (
                        value_loss_unclipped
                        if value_loss_unclipped is not None
                        else ""
                    ),
                    "entropy": (
                        entropy
                        if entropy is not None
                        else ""
                    ),
                    "value_explained_variance": (
                        explained_variance
                        if explained_variance is not None
                        else ""
                    ),
                    "kl_divergence": (
                        kl_divergence
                        if kl_divergence is not None
                        else ""
                    ),
                    "current_kl_coeff": (
                        current_kl_coeff
                        if current_kl_coeff is not None
                        else ""
                    ),
                    "gradient_scaler_scale": (
                        gradient_scaler_scale
                        if gradient_scaler_scale is not None
                        else ""
                    ),
                    "checkpoint_path": saved_checkpoint,
                }

                writer.writerow(row)
                csv_file.flush()

                def show(value):
                    return (
                        f"{value:.5f}"
                        if value is not None
                        else "n/a"
                    )

                print(
                    f"Iteration {iteration}/{args.iterations} | "
                    f"{elapsed:.1f}s | "
                    f"env steps="
                    f"{sampled_steps if sampled_steps is not None else 'n/a'} | "
                    f"scaled return={show(scaled_return)} | "
                    f"policy loss={show(policy_loss)} | "
                    f"value loss={show(value_loss)} | "
                    f"entropy={show(entropy)} | "
                    f"KL={show(kl_divergence)} | "
                    f"KL coeff={show(current_kl_coeff)}"
                )

                if saved_checkpoint:
                    print(f"  Checkpoint: {saved_checkpoint}")

                if sampled_steps is None:
                    print(
                        "  WARNING: RLlib environment-step metric "
                        "was not found in this result."
                    )

                if kl_divergence is None:
                    print(
                        "  INFO: KL metric was not exposed under "
                        "the inspected RLlib result keys."
                    )

        print("\n========== PILOT TRAINING FINISHED ==========")
        print(f"Training CSV: {csv_path}")
        print(f"Run configuration: {metadata_path}")
        print(f"Checkpoints: {checkpoint_dir}")
        print(
            "\nA completed short pilot and finite losses do not "
            "by themselves establish that the policy has learned."
        )

        return 0

    except KeyboardInterrupt:
        print(
            "\nPilot interrupted. Previously saved checkpoints "
            "and CSV rows have been retained."
        )
        return 130

    finally:
        if algo is not None:
            try:
                algo.stop()
            except Exception as exc:
                print(f"WARNING: Algorithm cleanup failed: {exc}")

        try:
            ray.shutdown()
        except Exception as exc:
            print(f"WARNING: Ray shutdown failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())