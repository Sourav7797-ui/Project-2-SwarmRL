"""Day-3 demo: run the env under random actions with the matplotlib
debug viewer live.

Run from the project root:
    python -m scripts.run_viewer_demo
"""
import time

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def main():
    config = SwarmEnvConfig(num_agents=12, num_obstacles=5, max_episode_steps=300)
    env = SwarmEnv(config=config, render_mode="human")
    env.reset(seed=42)

    try:
        while env.agents:
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            env.step(actions)
            env.render()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()