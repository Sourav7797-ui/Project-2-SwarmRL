"""Day-3 performance profile: step-time and effective tick rate.

Run from the project root:
    python -m scripts.profile_env_speed
"""
import time
import numpy as np

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def profile_episode(num_agents: int, num_obstacles: int, num_steps: int) -> dict:
    config = SwarmEnvConfig(num_agents=num_agents, num_obstacles=num_obstacles, max_episode_steps=num_steps)
    env = SwarmEnv(config=config)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    step_times = []

    while env.agents:
        actions = {
            a: rng.uniform(env.action_space(a).low, env.action_space(a).high).astype(np.float32)
            for a in env.agents
        }
        t0 = time.perf_counter()
        env.step(actions)
        step_times.append(time.perf_counter() - t0)

    step_times = np.array(step_times)
    return {
        "num_agents": num_agents,
        "total_steps": len(step_times),
        "mean_step_ms": step_times.mean() * 1000,
        "p95_step_ms": np.percentile(step_times, 95) * 1000,
        "total_wall_s": step_times.sum(),
        "effective_hz": 1.0 / step_times.mean(),
    }


def main():
    print(f"{'agents':>8} | {'steps':>6} | {'mean ms':>8} | {'p95 ms':>8} | {'total s':>8} | {'eff. Hz':>8}")
    print("-" * 66)
    for num_agents in (10, 50, 100, 200):
        result = profile_episode(num_agents=num_agents, num_obstacles=8, num_steps=1000)
        print(
            f"{result['num_agents']:>8} | {result['total_steps']:>6} | "
            f"{result['mean_step_ms']:>8.3f} | {result['p95_step_ms']:>8.3f} | "
            f"{result['total_wall_s']:>8.3f} | {result['effective_hz']:>8.1f}"
        )


if __name__ == "__main__":
    main()