"""FastAPI and WebSocket server for the SwarmRL 3D viewer."""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


class Simulation:
    """Owns one lightweight SwarmEnv instance for visualization."""

    def __init__(self) -> None:
        self.config = SwarmEnvConfig()
        self.env = SwarmEnv(config=self.config)
        self.running = False
        self.tick = 0
        self.total_collision_events = 0
        self.total_obstacle_collision_events = 0
        self.last_infos: dict = {}
        self.reset(seed=42)

    def reset(self, seed: int = 42) -> None:
        self.running = False
        self.tick = 0
        self.total_collision_events = 0
        self.total_obstacle_collision_events = 0
        self.last_infos = {}
        self.env.reset(seed=seed)

    def start(self) -> None:
        if not self.env.agents:
            self.reset()
        self.running = True

    def pause(self) -> None:
        self.running = False

    def step(self) -> None:
        if not self.running:
            return

        # A deterministic exploration controller for the live demo.
        # This is not a trained PPO or MAPPO policy.
        actions = {}
        for i, agent in enumerate(self.env.agents):
            phase = self.tick * 0.025 + i * 1.37
            pitch_rate = 0.12 * math.sin(phase * 0.7)
            yaw_rate = 0.22 * math.sin(phase)
            actions[agent] = np.array(
                [4.5, pitch_rate, yaw_rate],
                dtype=np.float32,
            )

        _, _, _, truncations, infos = self.env.step(actions)
        self.tick += 1
        self.last_infos = infos

        self.total_collision_events += sum(
            int(info.get("collision_event", False))
            for info in infos.values()
        )
        self.total_obstacle_collision_events += sum(
            int(info.get("obstacle_collision_event", False))
            for info in infos.values()
        )

        if truncations and all(truncations.values()):
            self.reset(seed=42)

    def snapshot(self) -> dict:
        positions = self.env.positions
        headings = self.env.headings
        obstacles = self.env.obstacles
        coverage = self.env.coverage_grid

        collisions = sum(
            int(info.get("collision", False))
            for info in self.last_infos.values()
        )
        obstacle_collisions = sum(
            int(info.get("obstacle_collision", False))
            for info in self.last_infos.values()
        )

        return {
            "type": "state",
            "running": self.running,
            "step": self.tick,
            "world": {
                "size_xy": self.config.world_size_xy,
                "height": self.config.world_height,
                "voxel_size": self.config.voxel_size,
                "obstacle_radius": self.config.obstacle_radius,
                "drone_radius": self.config.drone_radius,
            },
            "agents": [
                {
                    "id": agent,
                    "position": positions[i].tolist(),
                    "pitch": float(headings[i, 0]),
                    "yaw": float(headings[i, 1]),
                    "speed": float(self.env.speed[i]),
                    "battery": float(self.env.battery[i]),
                    "collision": bool(
                        self.last_infos.get(agent, {}).get("collision", False)
                    ),
                }
                for i, agent in enumerate(self.env.possible_agents)
            ],
            "obstacles": obstacles.tolist(),
            "coverage": {
                "shape": list(coverage.shape),
                "covered_indices": np.flatnonzero(coverage).tolist(),
                "percentage": float(coverage.mean() * 100.0),
            },
            "metrics": {
                "agent_count": len(self.env.possible_agents),
                "active_agents": len(self.env.agents),
                "collisions": collisions,
                "obstacle_collisions": obstacle_collisions,
                "total_collision_events": self.total_collision_events,
                "total_obstacle_collision_events":
                    self.total_obstacle_collision_events,
                "mean_speed": float(self.env.speed.mean()),
                "mean_battery": float(self.env.battery.mean()),
            },
        }


simulation = Simulation()


async def simulation_loop() -> None:
    while True:
        if simulation.running:
            simulation.step()
        await asyncio.sleep(0.1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_loop())
    yield
    simulation.pause()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    simulation.env.close()


app = FastAPI(
    title="SwarmRL 3D Visualization API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "SwarmRL 3D Visualization API",
        "agents": simulation.config.num_agents,
    }


@app.get("/api/state")
async def get_state():
    return simulation.snapshot()


@app.post("/api/start")
async def start_simulation():
    simulation.start()
    return {"status": "running"}


@app.post("/api/pause")
async def pause_simulation():
    simulation.pause()
    return {"status": "paused"}


@app.post("/api/reset")
async def reset_simulation():
    simulation.reset()
    return {"status": "reset", "step": simulation.tick}


@app.websocket("/ws/state")
async def websocket_state(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(simulation.snapshot())
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
