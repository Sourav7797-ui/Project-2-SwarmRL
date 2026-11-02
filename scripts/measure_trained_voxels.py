
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import torch

from backend.training.ray_runtime import (
    configure_environment,
    init_ray,
)
from backend.training.ppo_config import build_ppo_config
from backend.training.rllib_env import (
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
)

# Reuse the exact deterministic action-selection implementation.
from scripts.evaluate_checkpoint import select_action


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "day13_dynamic_64step_100iter"
    / "checkpoint_20261030_010304_802779"
    / "iteration_0080"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure voxel discovery by a trained PPO swarm."
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--checkpoint-stage", default="dynamic_obstacles")
    parser.add_argument("--stage", default="dynamic_obstacles")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints" / "day15_trained_voxel_measurement.json",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = args.checkpoint.resolve()
    output_path = args.output.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    if not (checkpoint / "algorithm_state.pkl").exists():
        raise FileNotFoundError(
            "algorithm_state.pkl not found inside checkpoint."
        )

    algo = None
    env = None
    ray_started = False

    episode_results = []
    step_discoveries = []
    

    try:
        print("Configuring environment...")
        configure_environment()
        init_ray()
        ray_started = True

        print("\nBuilding PPO configuration...")

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

        print("\nRestoring iteration-80 checkpoint...")
        algo.restore_from_path(str(checkpoint))

        module = algo.get_module(
            module_id=SHARED_POLICY_ID
        )

        if module is None:
            raise RuntimeError(
                f"RLModule {SHARED_POLICY_ID!r} not found."
            )

        if hasattr(module, "eval"):
            module.eval()

        print("Checkpoint restored successfully.")

        env = SwarmRLlibEnv(
            {
                "num_agents": args.agents,
                "max_episode_steps": args.steps,
                "stage": args.stage,
                "reward_scale": args.reward_scale,
            }
        )

        print("\nStarting trained-policy voxel measurement...")
        print(f"Episodes: {args.episodes}")
        print(f"Agents: {args.agents}")
        print(f"Steps per episode: {args.steps}")
        print(f"Stage: {args.stage}")

        for episode in range(args.episodes):

            obs, infos = env.reset(
                seed=args.seed + episode
            )

            # Counts include discoveries made during reset.
            voxel_counts = np.zeros(
                args.agents,
                dtype=np.int64,
            )

            initial_counts = env.env._last_new_voxel_counts.copy()
            voxel_counts += initial_counts

            total_collisions = 0
            completed_steps = 0

            for step in range(args.steps):

                actions = {}

                for agent_id, observation in obs.items():
                    actions[agent_id] = select_action(
                        module,
                        observation,
                    )

                obs, rewards, terminateds, truncateds, infos = (
                    env.step(actions)
                )

                # Capture newly discovered voxels for each drone.
                step_counts = (
                    env.env._last_new_voxel_counts.copy()
                )

                voxel_counts += step_counts
                step_discoveries.extend(int(value) for value in step_counts)

                # Count collision events from the environment infos.
                for info in infos.values():
                    if isinstance(info, dict):
                        total_collisions += int(
                            info.get("collision", False)
                        )
                        total_collisions += int(
                            info.get("obstacle_collision", False)
                        )

                completed_steps += 1

                if (
                    terminateds.get("__all__", False)
                    or truncateds.get("__all__", False)
                ):
                    break

            coverage_pct = float(
                env.env.coverage_grid.mean() * 100.0
            )

            result = {
                "episode": episode + 1,
                "seed": args.seed + episode,
                "steps": completed_steps,
                "new_voxels_total": int(voxel_counts.sum()),
                "new_voxels_per_drone": [
                    int(value) for value in voxel_counts
                ],
                "coverage_pct": coverage_pct,
                "collisions": total_collisions,
            }

            episode_results.append(result)

            print(
                f"Episode {episode + 1:02d}/{args.episodes}: "
                f"new_voxels={result['new_voxels_total']}, "
                f"coverage={coverage_pct:.4f}%, "
                f"collisions={total_collisions}"
            )

        totals = [
            item["new_voxels_total"]
            for item in episode_results
        ]

        summary = {
            "experiment": "day15_trained_policy_voxel_measurement",
            "timestamp": datetime.now().isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_stage": args.checkpoint_stage,
            "evaluation_stage": args.stage,
            "agents": args.agents,
            "steps_per_episode": args.steps,
            "episodes": args.episodes,
            "reward_scale": args.reward_scale,
            "seed_start": args.seed,
            "mean_new_voxels": statistics.mean(totals),
            "median_new_voxels": statistics.median(totals),
            "min_new_voxels": min(totals),
            "max_new_voxels": max(totals),
            "total_new_voxels": sum(totals),
            "step_discovery_mean": float(np.mean(step_discoveries)),
            "step_discovery_median": float(np.median(step_discoveries)),
            "step_discovery_p90": float(np.percentile(step_discoveries, 90)),
            "step_discovery_p95": float(np.percentile(step_discoveries, 95)),
            "step_discovery_max": int(max(step_discoveries)),
            "zero_discovery_pct": float(
                np.mean(np.array(step_discoveries) == 0) * 100.0
            ),
            "mean_coverage_pct": statistics.mean(
                item["coverage_pct"]
                for item in episode_results
            ),
            "mean_collisions": statistics.mean(
                item["collisions"]
                for item in episode_results
            ),
            "mean_voxels_per_drone": [
                statistics.mean(
                    item["new_voxels_per_drone"][drone]
                    for item in episode_results
                )
                for drone in range(args.agents)
            ],
            "episodes_detail": episode_results,
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
            )

        print("\n========== DAY 15 MEASUREMENT ==========")
        print(f"Mean new voxels: {summary['mean_new_voxels']:.2f}")
        print(f"Median new voxels: {summary['median_new_voxels']:.2f}")
        print(f"Minimum: {summary['min_new_voxels']}")
        print(f"Maximum: {summary['max_new_voxels']}")
        print(f"Mean coverage: {summary['mean_coverage_pct']:.4f}%")
        print(f"Mean collisions: {summary['mean_collisions']:.2f}")

        print("\nMean new voxels per drone:")
        for drone, count in enumerate(
            summary["mean_voxels_per_drone"]
        ):
            print(f"Drone {drone}: {count:.2f}")

        print(f"\nReport saved to: {output_path}")

    finally:
        if env is not None:
            env.close()

        if algo is not None:
            algo.stop()

        if ray_started:
            ray.shutdown()


if __name__ == "__main__":
    main()