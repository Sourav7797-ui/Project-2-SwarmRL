
"""Evaluate a saved SwarmRL RLlib checkpoint without training."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm

from backend.training.ray_runtime import configure_environment
from backend.training.rllib_env import (
    SHARED_POLICY_ID,
    SwarmRLlibEnv,
    register_swarm_env,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved SwarmRL PPO checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--object-store-mb", type=int, default=100)
    parser.add_argument("--stage", default="static_world")
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--output",
        default="checkpoints/day10/evaluation.json",
    )
    return parser.parse_args()


def stats_to_dict(stats):
    """Convert episode statistics to JSON-friendly data."""
    if stats is None:
        return {}

    if is_dataclass(stats):
        result = asdict(stats)
    elif hasattr(stats, "__dict__"):
        result = vars(stats).copy()
    else:
        return {"value": str(stats)}

    cleaned = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[key] = value
        else:
            cleaned[key] = str(value)

    return cleaned


def get_deterministic_action(module, action_dist_class, observation):
    """Run one observation through the new RLlib module API."""
    obs_tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32)
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = module.forward_inference({"obs": obs_tensor})

        if "action_dist_inputs" not in outputs:
            raise KeyError(
                "RLModule inference output has no 'action_dist_inputs'. "
                f"Available keys: {list(outputs.keys())}"
            )

        distribution = action_dist_class.from_logits(
            outputs["action_dist_inputs"]
        )
        deterministic_distribution = distribution.to_deterministic()
        action = deterministic_distribution.sample()[0]

    return (
        action.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def main():
    args = parse_args()

    if args.episodes < 1 or args.agents < 1 or args.steps < 1:
        raise ValueError("episodes, agents, and steps must be positive.")
    if args.cpus < 1 or args.object_store_mb < 1:
        raise ValueError("cpus and object-store-mb must be positive.")
    if args.reward_scale <= 0:
        raise ValueError("reward-scale must be positive.")

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint}"
        )

    configure_environment()
    register_swarm_env()

    if not ray.is_initialized():
        ray.init(
            num_cpus=args.cpus,
            object_store_memory=args.object_store_mb * 1024 * 1024,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )

    algo = None
    env = None

    try:
        print(f"Loading checkpoint: {checkpoint}")
        algo = Algorithm.from_checkpoint(str(checkpoint))
        print("Checkpoint loaded successfully.")

        module = algo.get_module(SHARED_POLICY_ID)
        if module is None:
            raise RuntimeError(
                f"Could not find RLModule {SHARED_POLICY_ID!r} "
                "in the restored checkpoint."
            )

        action_dist_class = module.get_inference_action_dist_cls()
        module.eval()
        print("Shared policy module ready for inference.")

        env = SwarmRLlibEnv(
            {
                "num_agents": args.agents,
                "max_episode_steps": args.steps,
                "stage": args.stage,
                "reward_scale": args.reward_scale,
            }
        )

        results = []

        for episode in range(args.episodes):
            observations, _ = env.reset(seed=args.seed + episode)
            terminated = False
            truncated = False
            episode_steps = 0

            while not terminated and not truncated:
                actions = {}

                for agent_id, observation in observations.items():
                    action = get_deterministic_action(
                        module,
                        action_dist_class,
                        observation,
                    )
                    actions[agent_id] = np.clip(
                        action, -1.0, 1.0
                    ).astype(np.float32)

                (
                    observations,
                    _rewards,
                    terminations,
                    truncations,
                    _infos,
                ) = env.step(actions)

                terminated = bool(terminations.get("__all__", False))
                truncated = bool(truncations.get("__all__", False))
                episode_steps += 1

            record = {
                "episode": episode + 1,
                "steps": episode_steps,
                "raw_return": env.raw_episode_return,
                "environment_stats": stats_to_dict(
                    env.last_episode_stats
                ),
            }
            results.append(record)

            print(
                f"Episode {episode + 1}/{args.episodes} | "
                f"steps={episode_steps} | "
                f"raw return={record['raw_return']}"
            )

            if record["environment_stats"]:
                print(
                    "  Environment stats: "
                    + json.dumps(
                        record["environment_stats"], default=str
                    )
                )

        returns = [
            row["raw_return"]
            for row in results
            if isinstance(row["raw_return"], (int, float))
            and not isinstance(row["raw_return"], bool)
        ]

        summary = {
            "checkpoint": str(checkpoint),
            "episodes": len(results),
            "agents": args.agents,
            "max_episode_steps": args.steps,
            "stage": args.stage,
            "reward_scale": args.reward_scale,
            "exploration_enabled": False,
            "episode_results": results,
            "mean_raw_return": (
                statistics.mean(returns) if returns else None
            ),
            "min_raw_return": min(returns) if returns else None,
            "max_raw_return": max(returns) if returns else None,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n=== Evaluation summary ===")
        print(f"Episodes evaluated: {len(results)}")
        print(f"Mean raw return: {summary['mean_raw_return']}")
        print(f"Minimum raw return: {summary['min_raw_return']}")
        print(f"Maximum raw return: {summary['max_raw_return']}")
        print(f"Results saved to: {output_path.resolve()}")

    finally:
        if env is not None:
            env.close()
        if algo is not None:
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()