"""Day-9 smoke test: does the whole Ray + RLlib + torch path run on THIS machine?

Builds the Phase 1 PPO algorithm (shared policy), trains a few small iterations,
and prints the key numbers. It is not meant to learn anything yet; it answers
"does it run, and are the numbers sane?" before real training (Day 10).

Run from the project root:
    python -m scripts.rllib_smoke
    python -m scripts.rllib_smoke --agents 50 --steps 200 --iterations 3
    python -m scripts.rllib_smoke --workers 2          # parallel env runners
    python -m scripts.rllib_smoke --stage wind
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _flatten(d: dict, prefix: str = ""):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}/")
        else:
            yield f"{prefix}{key}", value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100, help="episode length (also the batch size in env steps)")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0, help="Ray env runners (0 = sample in this process)")
    parser.add_argument("--cpus", type=int, default=None, help="CPUs Ray may use (default: all detected)")
    parser.add_argument("--stage", default=None, choices=[None, "static_world", "dynamic_obstacles", "wind"])
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Ray worker processes are started fresh (always on Windows) and must be able to
    # `import backend`: make the project root visible to them before Ray starts.
    os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(ROOT))

    import ray

    from backend.training.ppo_config import (
        approx_gradient_updates_per_iteration,
        build_ppo_config,
        samples_per_iteration,
    )

    batch = args.steps                      # one episode per iteration
    minibatch = min(batch, 500)
    epochs = 2

    print(f"ray {ray.__version__} | agents={args.agents} episode={args.steps} steps | "
          f"env runners={args.workers} | stage={args.stage}")
    print(f"per iteration: {samples_per_iteration(args.agents, batch):,} samples, "
          f"~{approx_gradient_updates_per_iteration(args.agents, batch, minibatch, epochs)} gradient updates "
          f"(minibatch {minibatch}, {epochs} epochs)")

    ray.init(num_cpus=args.cpus, include_dashboard=False, logging_level=logging.ERROR, ignore_reinit_error=True)
    algo = None
    try:
        config = build_ppo_config(
            num_agents=args.agents, max_episode_steps=args.steps, stage=args.stage,
            reward_scale=args.reward_scale, num_env_runners=args.workers,
            train_batch_size=batch, minibatch_size=minibatch, num_epochs=epochs,
            hidden=(64, 64), seed=args.seed,
        )
        t0 = time.time()
        algo = config.build_algo()
        print(f"algorithm built in {time.time() - t0:.1f}s\n")

        print(f"{'iter':>4} {'wall s':>7} {'env steps':>9} {'swarm return':>13} {'policy loss':>12} "
              f"{'value loss':>11} {'entropy':>8} {'vf expl.var':>11}")
        ok = True
        for i in range(1, args.iterations + 1):
            t = time.time()
            rows = dict(_flatten(algo.train()))
            wall = time.time() - t

            scaled_return = rows.get("env_runners/episode_return_mean")
            # RLlib reports the SUM of the agents' scaled rewards; undo the scale for readability.
            swarm_return = scaled_return / args.reward_scale if scaled_return is not None else float("nan")
            prefix = f"learners/shared_policy/"
            policy_loss = rows.get(prefix + "policy_loss", float("nan"))
            vf_loss = rows.get(prefix + "vf_loss", float("nan"))
            entropy = rows.get(prefix + "entropy", float("nan"))
            expl = rows.get(prefix + "vf_explained_var", float("nan"))
            steps = rows.get("env_runners/num_env_steps_sampled", float("nan"))

            return_text = f"{swarm_return:>13.1f}" if math.isfinite(swarm_return) else f"{'n/a':>13}"   # no episode finished yet
            print(f"{i:>4} {wall:>7.1f} {steps:>9.0f} {return_text} {policy_loss:>12.4f} "
                  f"{vf_loss:>11.4f} {entropy:>8.3f} {expl:>11.3f}")
            ok &= all(math.isfinite(x) for x in (policy_loss, vf_loss, entropy))

        print()
        print("SMOKE TEST OK" if ok else "SMOKE TEST FAILED: non-finite loss values")
        return 0 if ok else 1
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())