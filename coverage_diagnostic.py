import numpy as np
from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv

config = SwarmEnvConfig(num_agents=8, max_episode_steps=32)
env = SwarmEnv(config)

observations, infos = env.reset(seed=5000)

def report(step):
    covered = int(env.coverage_grid.sum())
    total = int(env.coverage_grid.size)
    print(
        f"step={step:2d} | "
        f"covered={covered:5d}/{total} | "
        f"coverage={100.0 * covered / total:.4f}% | "
        f"position_min={np.round(env.positions.min(axis=0), 2)} | "
        f"position_max={np.round(env.positions.max(axis=0), 2)}"
    )

report(0)

rng = np.random.default_rng(5000)
for step in range(1, 33):
    actions = {}
    for agent in env.agents:
        space = env.action_space(agent)
        actions[agent] = rng.uniform(
            low=space.low,
            high=space.high,
        ).astype(np.float32)

    observations, rewards, terminations, truncations, infos = env.step(actions)

    if step in (1, 4, 8, 16, 24, 32):
        report(step)

print("Sensor range:", config.mapping_sensor_range)
print("Grid shape:", env.coverage_grid.shape)
print("Total voxels:", env.coverage_grid.size)

env.close()
