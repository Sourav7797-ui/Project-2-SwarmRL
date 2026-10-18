"""Day-6: reward distribution analysis (run BEFORE any RL training).

Runs full episodes under simple baseline policies with a
RewardComponentLogger attached, then reports where the reward comes from.
Two questions this answers:

  1. Reward balance: is one component drowning out the others? (e.g. if
     collision penalties are 95% of the total magnitude, PPO will mostly
     learn collision-avoidance and barely explore.)
  2. Degenerate optima: does a policy that does nothing score as well as
     one that explores? If so, PPO's easiest "improvement" from a random
     start is to stop moving.

Percentages are shares of GROSS magnitude (sum of absolute values), not of
the net total. Net can sit near zero when positives and negatives cancel,
which makes "% of net" meaningless.

Run from the project root:
    python -m scripts.analyze_reward_distribution
    python -m scripts.analyze_reward_distribution --episodes 10 --agents 50
"""
from __future__ import annotations

import argparse

import numpy as np

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv
from backend.reward.reward_logger import RewardComponentLogger
from backend.reward.reward_shaping import ALL_COMPONENTS


# ----------------------------- baseline policies -----------------------------
def random_policy(env, agents, rng):
    return {
        a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
        for a in agents
    }


def stationary_policy(env, agents, rng):
    """Command zero velocity and zero turn: the 'do nothing' baseline."""
    return {a: np.zeros(3, dtype=np.float32) for a in agents}


def cruise_policy(env, agents, rng):
    """Full speed, no turning, no pitch: drones fly straight until a wall."""
    top_speed = env.config.max_velocity
    return {a: np.array([top_speed, 0.0, 0.0], dtype=np.float32) for a in agents}


def _make_wander_policy(speed: float):
    """A careful mover: fixed speed, small random turns."""
    def policy(env, agents, rng):
        c = env.config
        out = {}
        for a in agents:
            turn = rng.uniform(-0.3, 0.3, size=2) * np.array([c.max_pitch_rate, c.max_yaw_rate])
            out[a] = np.array([speed, turn[0], turn[1]], dtype=np.float32)
        return out
    return policy


gentle_policy = _make_wander_policy(3.0)   # moderate speed
slow_policy = _make_wander_policy(1.0)     # barely moving: the policy PPO would
                                           # drift toward from standing still


POLICIES = {
    "random": random_policy,
    "stationary": stationary_policy,
    "cruise": cruise_policy,
    "gentle": gentle_policy,
    "slow": slow_policy,
}


# ----------------------------- one episode -----------------------------
def run_episode(config: SwarmEnvConfig, policy_fn, seed: int) -> dict:
    env = SwarmEnv(config=config)
    logger = RewardComponentLogger()
    env.reward_logger = logger
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 10_000)

    # env.reset() starts every drone facing +x (heading 0,0). A policy that
    # barely turns would then send the whole swarm into the same wall and
    # measure a pile-up instead of the reward. Spread the headings so each
    # baseline is judged on its own behavior.
    env.headings[:, 1] = rng.uniform(0.0, 2.0 * np.pi, size=env.config.num_agents)

    drone_collision_ticks = 0
    obstacle_collision_ticks = 0
    agent_steps = 0

    while env.agents:
        actions = policy_fn(env, env.agents, rng)
        _, _, _, _, infos = env.step(actions)
        for info in infos.values():
            drone_collision_ticks += int(info["collision"])
            obstacle_collision_ticks += int(info["obstacle_collision"])
        agent_steps += len(infos)

    return {
        "totals": logger.episode_totals(),
        "agent_steps": agent_steps,
        "coverage_pct": float(env.coverage_grid.mean() * 100),
        "drone_collision_ticks": drone_collision_ticks,
        "obstacle_collision_ticks": obstacle_collision_ticks,
    }


# ----------------------------- aggregation + report -----------------------------
def aggregate(results: list[dict]) -> dict:
    mean_totals = {c: float(np.mean([r["totals"].get(c, 0.0) for r in results])) for c in ALL_COMPONENTS}
    return {
        "mean_totals": mean_totals,
        "net": sum(mean_totals.values()),
        "gross": sum(abs(v) for v in mean_totals.values()),
        "agent_steps": float(np.mean([r["agent_steps"] for r in results])),
        "coverage_pct": float(np.mean([r["coverage_pct"] for r in results])),
        "drone_collision_ticks": float(np.mean([r["drone_collision_ticks"] for r in results])),
        "obstacle_collision_ticks": float(np.mean([r["obstacle_collision_ticks"] for r in results])),
    }


def print_policy_report(name: str, agg: dict) -> None:
    print(f"\n=== policy: {name} ===")
    print(f"{'component':20s} {'mean total':>12s} {'per agent-step':>15s} {'% of gross':>11s}")
    for c in ALL_COMPONENTS:
        total = agg["mean_totals"][c]
        per_step = total / max(agg["agent_steps"], 1)
        share = abs(total) / agg["gross"] * 100 if agg["gross"] > 0 else 0.0
        print(f"{c:20s} {total:12.2f} {per_step:15.5f} {share:10.1f}%")
    print(f"{'NET':20s} {agg['net']:12.2f} {agg['net'] / max(agg['agent_steps'], 1):15.5f}")
    print(f"final coverage: {agg['coverage_pct']:.1f}%   "
          f"drone-collision ticks: {agg['drone_collision_ticks']:.0f}   "
          f"obstacle-collision ticks: {agg['obstacle_collision_ticks']:.0f}")


def print_diagnosis(aggs: dict[str, dict]) -> None:
    print("\n=== diagnosis ===")
    nets = {name: a["net"] for name, a in aggs.items()}
    best_name = max(nets, key=nets.get)
    print("net reward by policy: " + ", ".join(f"{n}={v:.1f}" for n, v in nets.items()))

    # The decisive check: is there ANY sensible mover that out-earns standing
    # still? If so, PPO has a path from "do nothing" toward exploring. Pure
    # random/cruise flight are allowed to lose: they are crash-prone by design.
    movers = {n: v for n, v in nets.items() if n in ("gentle", "slow")}
    best_mover = max(movers, key=movers.get)
    if nets["stationary"] >= movers[best_mover]:
        print("WARNING: no careful mover beats 'stationary'. Standing still is a local optimum")
        print("         PPO will fall into. Raise exploration reward or lower penalties.")
    else:
        print(f"OK: '{best_mover}' beats standing still "
              f"({movers[best_mover]:.1f} vs {nets['stationary']:.1f}), so a gradient toward exploring exists.")
    if nets["stationary"] >= nets["random"]:
        print("note: 'stationary' also beats pure random flight (expected for a crash-prone policy).")

    for name, a in aggs.items():
        if a["gross"] == 0:
            continue
        pen = abs(a["mean_totals"]["drone_collision"]) + abs(a["mean_totals"]["obstacle_collision"])
        pen_share = pen / a["gross"] * 100
        if pen_share > 70:
            print(f"WARNING: under '{name}', collision penalties are {pen_share:.0f}% of gross reward "
                  f"magnitude and will dominate the learning signal.")

    print(f"highest net reward: '{best_name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = SwarmEnvConfig(num_agents=args.agents, max_episode_steps=args.steps)
    print(f"reward weights: exploration={config.exploration_reward}, "
          f"collision={config.collision_penalty}, obstacle={config.obstacle_collision_penalty}, "
          f"boundary_scale={config.boundary_penalty_scale}, "
          f"team_bonus_scale={config.team_coverage_bonus_scale}")
    print(f"{args.episodes} episodes x {args.steps} steps x {args.agents} agents per policy")

    aggs = {}
    for name, policy_fn in POLICIES.items():
        results = [run_episode(config, policy_fn, seed=args.seed + i) for i in range(args.episodes)]
        aggs[name] = aggregate(results)
        print_policy_report(name, aggs[name])

    print_diagnosis(aggs)


if __name__ == "__main__":
    main()