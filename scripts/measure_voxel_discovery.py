
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def main():
    config = SwarmEnvConfig(
        num_agents=8,
        num_obstacles=8,
        max_episode_steps=64,
        obstacles_move=True,
        obstacle_speed=1.5,
        wind_speed=0.0,
    )

    env = SwarmEnv(config=config)

    all_counts = []
    episode_totals = []

    try:
        for episode in range(10):
            seed = 7000 + episode
            env.reset(seed=seed)

            episode_counts = []

            for step in range(64):
                # Neutral action: zero commanded speed and
                # zero pitch/yaw rates.
                actions = {
                    agent: np.zeros(3, dtype=np.float32)
                    for agent in env.agents
                }

                _, _, _, truncations, _ = env.step(actions)

                counts = env._last_new_voxel_counts.copy()
                episode_counts.extend(counts.tolist())
                all_counts.extend(counts.tolist())

                if all(truncations.values()):
                    break

            episode_total = int(sum(episode_counts))
            episode_totals.append(episode_total)

            print(
                f"Episode {episode + 1:02d} | "
                f"Seed {seed} | "
                f"New voxels: {episode_total}"
            )

        values = np.asarray(all_counts, dtype=np.int32)

        print("\n========== VOXEL DISCOVERY REPORT ==========")
        print(f"Total drone-step observations: {len(values)}")
        print(f"Total newly discovered voxels: {int(values.sum())}")
        print(f"Mean per drone-step: {values.mean():.3f}")
        print(f"Median per drone-step: {np.median(values):.3f}")
        print(f"Maximum per drone-step: {values.max()}")
        print(f"90th percentile: {np.percentile(values, 90):.2f}")
        print(f"95th percentile: {np.percentile(values, 95):.2f}")
        print(
            f"Zero-discovery frequency: "
            f"{np.mean(values == 0) * 100:.2f}%"
        )
        print(
            f"Positive-discovery frequency: "
            f"{np.mean(values > 0) * 100:.2f}%"
        )
        print(
            f"Mean newly discovered voxels per episode: "
            f"{np.mean(episode_totals):.2f}"
        )
        print("============================================")

    finally:
        env.close()


if __name__ == "__main__":
    main()
