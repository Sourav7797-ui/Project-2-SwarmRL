"""Day-6: reward distribution analysis (run BEFORE any RL training).

Runs full episodes under three simple baseline policies with a
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
from backend.reward.curriculum import DEFAULT_STAGES
from backend.reward.reward_logger import RewardComponentLogger
from backend.reward.reward_shaping import ALL_COMPONENTS


# ----------------------------- baseline policies -----------------------------
def random_policy(env, agents, rng, obs):
    return {
        a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
        for a in agents
    }


def stationary_policy(env, agents, rng, obs):
    """Command zero velocity and zero turn: the 'do nothing' baseline."""
    return {a: np.zeros(3, dtype=np.float32) for a in agents}


def cruise_policy(env, agents, rng, obs):
    """Full speed, no turning, no pitch: drones fly straight until a wall."""
    top_speed = env.config.max_velocity
    return {a: np.array([top_speed, 0.0, 0.0], dtype=np.float32) for a in agents}


def _make_wander_policy(speed: float):
    """A careful mover: fixed speed, small random turns."""
    def policy(env, agents, rng, obs):
        c = env.config
        out = {}
        for a in agents:
            turn = rng.uniform(-0.3, 0.3, size=2) * np.array([c.max_pitch_rate, c.max_yaw_rate])
            out[a] = np.array([speed, turn[0], turn[1]], dtype=np.float32)
        return out
    return policy


def reactive_policy(env, agents, rng, obs):
    """Uses ONLY the observation vector (no privileged state): steer away from
    a nearby wall, obstacle or neighbour, level out toward mid-height, and turn away when
    the area ahead is already explored. Same 3 m/s as 'gentle', so any
    difference between the two is down to what the observation lets it do."""
    c = env.config
    k = c.k_nearest
    i_cov, i_wd, i_wb, i_alt, i_pitch = 3 * k + 1, 3 * k + 3, 3 * k + 4, 3 * k + 5, 3 * k + 6
    i_od, i_ob = 3 * k, 3 * k + 7                 # nearest-obstacle distance and body-frame bearing
    out = {}
    for a in agents:
        o = obs[a]
        yaw_rate = 0.0
        wall_dist, wall_bearing = o[i_wd], o[i_wb]
        nb_dist, nb_bearing = o[0], o[1]          # nearest neighbour
        # Priority matters: an imminent obstacle outranks the wall rule. With the wall
        # rule first, drones skimming a wall flew straight into obstacles beside it.
        if o[i_od] < 8.0 and abs(o[i_ob]) < np.pi / 2:
            yaw_rate = -(np.sign(o[i_ob]) or 1.0) * c.max_yaw_rate          # obstacle ahead: turn away
        elif wall_dist < 12.0 and abs(wall_bearing) < np.pi / 2:
            yaw_rate = -(np.sign(wall_bearing) or 1.0) * c.max_yaw_rate     # wall ahead: turn away
        elif nb_dist < 8.0 and abs(nb_bearing) < np.pi / 2:
            yaw_rate = -(np.sign(nb_bearing) or 1.0) * c.max_yaw_rate       # neighbour ahead: turn away
        elif o[i_cov] > 0.6:
            yaw_rate = rng.uniform(-1.0, 1.0) * 0.5 * c.max_yaw_rate        # explored ahead: look elsewhere
        target_pitch = -0.6 * o[i_alt]                                       # drift back toward mid-height
        pitch_rate = float(np.clip((target_pitch - o[i_pitch]) * 2.0, -c.max_pitch_rate, c.max_pitch_rate))
        out[a] = np.array([3.0, pitch_rate, yaw_rate], dtype=np.float32)
    return out


gentle_policy = _make_wander_policy(3.0)   # moderate speed
slow_policy = _make_wander_policy(1.0)     # barely moving: the policy PPO would
                                           # drift toward from standing still


POLICIES = {
    "random": random_policy,
    "stationary": stationary_policy,
    "cruise": cruise_policy,
    "gentle": gentle_policy,
    "slow": slow_policy,
    "reactive": reactive_policy,
}


# ----------------------------- one episode -----------------------------
def run_episode(config: SwarmEnvConfig, policy_fn, seed: int) -> dict:
    env = SwarmEnv(config=config)
    logger = RewardComponentLogger()
    env.reward_logger = logger
    obs, _ = env.reset(seed=seed)       # reset() now spawns random headings itself
    rng = np.random.default_rng(seed + 10_000)

    drone_collision_ticks = 0
    obstacle_collision_ticks = 0
    agent_steps = 0

    while env.agents:
        actions = policy_fn(env, env.agents, rng, obs)
        obs, _, _, _, infos = env.step(actions)
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
    print("net reward by policy: " + ", ".join(f"{n}={v:.1f}" for n, v in nets.items()))

    # Decisive check: does a policy that uses ONLY what a drone can observe beat
    # standing still? That tests both that the reward leaves a path toward
    # exploring AND that the observation carries enough information to act on it.
    # (Blind movers are not the right test: they cannot avoid walls or neighbours,
    # and their margin over 'stationary' is within seed-to-seed noise.)
    if "reactive" not in nets or "stationary" not in nets:
        print("(skipped: run both 'reactive' and 'stationary' to get the pass/fail check)")
    elif nets["reactive"] > nets["stationary"]:
        print(f"OK: an observation-driven policy beats standing still "
              f"({nets['reactive']:.1f} vs {nets['stationary']:.1f}).")
    else:
        print("WARNING: even an observation-driven policy does not beat 'stationary'. Either the")
        print("         reward leaves no path toward exploring, or the observation is missing")
        print("         information the policy needs.")

    # Is the boundary penalty learnable? Compare against the blind policy that
    # flies at the same speed: the gap is what the wall features buy.
    if "gentle" in aggs and "reactive" in aggs:
        blind_b = abs(aggs["gentle"]["mean_totals"]["boundary"])
        obs_b = abs(aggs["reactive"]["mean_totals"]["boundary"])
        if blind_b > 0:
            print(f"boundary penalty at equal speed: blind {blind_b:.1f} -> observation-driven {obs_b:.1f} "
                  f"({(1 - obs_b / blind_b) * 100:.0f}% lower)")

    blind = {n: v for n, v in nets.items() if n in ("gentle", "slow")}
    losers = [n for n, v in blind.items() if v <= nets["stationary"]]
    if losers:
        print(f"note: blind movers {losers} do not beat 'stationary'. Expected: they cannot avoid anything.")
    if "stationary" in nets and "random" in nets and nets["stationary"] >= nets["random"]:
        print("note: 'stationary' also beats pure random flight (expected for a crash-prone policy).")

    for name, a in aggs.items():
        if a["gross"] == 0:
            continue
        pen = abs(a["mean_totals"]["drone_collision"]) + abs(a["mean_totals"]["obstacle_collision"])
        pen_share = pen / a["gross"] * 100
        if pen_share > 70:
            print(f"WARNING: under '{name}', collision penalties are {pen_share:.0f}% of gross reward "
                  f"magnitude and will dominate the learning signal.")

    print(f"highest net reward: '{max(nets, key=nets.get)}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", default="none", choices=["none"] + [st.name for st in DEFAULT_STAGES],
                        help="apply a curriculum stage (obstacle count, moving obstacles, wind)")
    parser.add_argument("--policies", default=",".join(POLICIES),
                        help="comma-separated subset of: " + ", ".join(POLICIES))
    args = parser.parse_args()

    config = SwarmEnvConfig(num_agents=args.agents, max_episode_steps=args.steps)
    if args.stage != "none":
        stage = next(st for st in DEFAULT_STAGES if st.name == args.stage)
        config = stage.apply_to(config)
    print(f"stage: {args.stage}   obstacles={config.num_obstacles}  moving={config.obstacles_move}  "
          f"obstacle_speed={config.obstacle_speed}  wind_speed={config.wind_speed}")
    print(f"reward weights: exploration={config.exploration_reward}, "
          f"collision={config.collision_penalty}, obstacle={config.obstacle_collision_penalty}, "
          f"boundary_scale={config.boundary_penalty_scale}, "
          f"team_bonus_scale={config.team_coverage_bonus_scale}")
    print(f"{args.episodes} episodes x {args.steps} steps x {args.agents} agents per policy")

    aggs = {}
    chosen = [n.strip() for n in args.policies.split(",") if n.strip()]
    unknown = [n for n in chosen if n not in POLICIES]
    if unknown:
        parser.error(f"unknown policies {unknown}; choose from {list(POLICIES)}")
    for name in chosen:
        policy_fn = POLICIES[name]
        results = [run_episode(config, policy_fn, seed=args.seed + i) for i in range(args.episodes)]
        aggs[name] = aggregate(results)
        print_policy_report(name, aggs[name])

    print_diagnosis(aggs)


if __name__ == "__main__":
    main()