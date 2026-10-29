
"""Compare a trained SwarmRL PPO policy against random actions."""

from __future__ import annotations

import argparse
import json
import statistics
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
        description="Compare PPO and random-action swarm baselines."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--object-store-mb", type=int, default=100)
    parser.add_argument("--stage", default="static_world")
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--output",
        default="checkpoints/day10/baseline_comparison.json",
    )
    return parser.parse_args()


def deterministic_action(module, distribution_class, observation):
    """Infer a deterministic action using RLlib's new module API."""
    obs_tensor = torch.as_tensor(
        np.asarray(observation, dtype=np.float32)
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = module.forward_inference({"obs": obs_tensor})

        if "action_dist_inputs" not in outputs:
            raise KeyError(
                "Missing action_dist_inputs in RLModule output. "
                f"Available keys: {list(outputs.keys())}"
            )

        distribution = distribution_class.from_logits(
            outputs["action_dist_inputs"]
        ).to_deterministic()

        action = distribution.sample()[0]

    return action.detach().cpu().numpy().astype(np.float32)


def run_episode(env, policy, seed, rng, module=None, distribution_class=None):
    """Run one episode. policy is either 'ppo' or 'random'."""
    observations, _ = env.reset(seed=seed)
    done = False
    steps = 0

    while not done:
        actions = {}

        for agent_id, observation in observations.items():
            if policy == "ppo":
                action = deterministic_action(
                    module, distribution_class, observation
                )
            else:
                # The adapter expects normalized actions in [-1, 1].
                action = rng.uniform(
                    low=-1.0, high=1.0, size=3
                ).astype(np.float32)

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

        done = bool(
            terminations.get("__all__", False)
            or truncations.get("__all__", False)
        )
        steps += 1

    stats = env.last_episode_stats
    return {
        "steps": steps,
        "raw_return": env.raw_episode_return,
        "coverage_pct": stats.coverage_pct if stats else None,
        "collision_events": stats.collision_events if stats else None,
        "agent_steps": stats.agent_steps if stats else None,
        "collisions_per_1k": stats.collisions_per_1k if stats else None,
    }


def summarize(records):
    """Calculate mean and range for each numeric metric."""
    metrics = (
        "raw_return",
        "coverage_pct",
        "collision_events",
        "collisions_per_1k",
        "steps",
    )
    summary = {"episodes": len(records)}

    for metric in metrics:
        values = [
            record[metric]
            for record in records
            if isinstance(record.get(metric), (int, float))
            and not isinstance(record.get(metric), bool)
        ]

        summary[metric] = {
            "mean": statistics.mean(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return summary


def main():
    args = parse_args()

    if args.episodes < 1 or args.agents < 1 or args.steps < 1:
        raise ValueError("episodes, agents, and steps must be positive.")
    if args.reward_scale <= 0:
        raise ValueError("reward-scale must be positive.")
    if args.cpus < 1 or args.object_store_mb < 1:
        raise ValueError("cpus and object-store-mb must be positive.")

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
    ppo_env = None
    random_env = None

    try:
        print(f"Loading PPO checkpoint: {checkpoint}")
        algo = Algorithm.from_checkpoint(str(checkpoint))
        module = algo.get_module(SHARED_POLICY_ID)

        if module is None:
            raise RuntimeError(
                f"Shared RLModule {SHARED_POLICY_ID!r} was not found."
            )

        distribution_class = module.get_inference_action_dist_cls()
        module.eval()
        print("PPO checkpoint loaded.")

        env_config = {
            "num_agents": args.agents,
            "max_episode_steps": args.steps,
            "stage": args.stage,
            "reward_scale": args.reward_scale,
        }

        ppo_env = SwarmRLlibEnv(env_config)
        random_env = SwarmRLlibEnv(env_config)

        ppo_results = []
        random_results = []

        for episode in range(args.episodes):
            episode_seed = args.seed + episode

            # Separate, reproducible RNG for the random-action baseline.
            random_rng = np.random.default_rng(episode_seed + 1_000_000)

            ppo_record = run_episode(
                ppo_env,
                "ppo",
                episode_seed,
                random_rng,
                module,
                distribution_class,
            )
            random_record = run_episode(
                random_env,
                "random",
                episode_seed,
                random_rng,
            )

            ppo_record["episode"] = episode + 1
            ppo_record["seed"] = episode_seed
            random_record["episode"] = episode + 1
            random_record["seed"] = episode_seed

            ppo_results.append(ppo_record)
            random_results.append(random_record)

            print(
                f"\nEpisode {episode + 1}/{args.episodes} "
                f"(seed={episode_seed})"
            )
            print(
                f"  PPO    | return={ppo_record['raw_return']:.3f} "
                f"| coverage={ppo_record['coverage_pct']:.4f}% "
                f"| collisions={ppo_record['collision_events']}"
            )
            print(
                f"  Random | return={random_record['raw_return']:.3f} "
                f"| coverage={random_record['coverage_pct']:.4f}% "
                f"| collisions={random_record['collision_events']}"
            )

        ppo_summary = summarize(ppo_results)
        random_summary = summarize(random_results)

        comparison = {
            "checkpoint": str(checkpoint),
            "agents": args.agents,
            "max_episode_steps": args.steps,
            "stage": args.stage,
            "reward_scale": args.reward_scale,
            "episodes": args.episodes,
            "seed_start": args.seed,
            "ppo": {
                "policy": "trained PPO, deterministic actions",
                "episodes": ppo_results,
                "summary": ppo_summary,
            },
            "random": {
                "policy": "uniform random normalized actions",
                "episodes": random_results,
                "summary": random_summary,
            },
            "mean_return_difference_ppo_minus_random": (
                ppo_summary["raw_return"]["mean"]
                - random_summary["raw_return"]["mean"]
            ),
            "mean_coverage_difference_percentage_points": (
                ppo_summary["coverage_pct"]["mean"]
                - random_summary["coverage_pct"]["mean"]
            ),
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(comparison, indent=2, default=str),
            encoding="utf-8",
        )

        print("\n=== Baseline comparison summary ===")
        print(
            "PPO mean return:   "
            f"{ppo_summary['raw_return']['mean']:.4f}"
        )
        print(
            "Random mean return:"
            f" {random_summary['raw_return']['mean']:.4f}"
        )
        print(
            "PPO mean coverage: "
            f"{ppo_summary['coverage_pct']['mean']:.4f}%"
        )
        print(
            "Random mean coverage: "
            f"{random_summary['coverage_pct']['mean']:.4f}%"
        )
        print(
            "PPO mean collisions: "
            f"{ppo_summary['collision_events']['mean']:.2f}"
        )
        print(
            "Random mean collisions: "
            f"{random_summary['collision_events']['mean']:.2f}"
        )
        print(f"\nResults saved to: {output.resolve()}")

    finally:
        if ppo_env is not None:
            ppo_env.close()
        if random_env is not None:
            random_env.close()
        if algo is not None:
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()