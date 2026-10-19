"""Central configuration for the SwarmRL environment."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SwarmEnvConfig:
    # --- Swarm ---
    num_agents: int = 50
    k_nearest: int = 6

    # --- World (continuous 3D disaster zone) ---
    world_size_xy: float = 200.0
    world_height: float = 40.0
    voxel_size: float = 4.0

    # --- Physics limits ---
    max_velocity: float = 8.0
    max_pitch_rate: float = 1.0
    max_yaw_rate: float = 1.5
    # Climb/dive angle limit (Day 6 fix). Pitch used to accumulate with no
    # bound, so drones drifted past +-90 degrees and flew upside-down, then
    # got pinned against the floor/ceiling. pi/4 = 45 degrees.
    max_pitch_angle: float = 0.7853981633974483
    dt: float = 0.1

    # --- Momentum / drag ---
    max_acceleration: float = 4.0
    drag_coefficient: float = 0.02

    # --- Collision geometry ---
    drone_radius: float = 0.5
    obstacle_radius: float = 3.0

    # --- Episode ---
    max_episode_steps: int = 1000
    battery_capacity: float = 1.0

    # --- Sensing ---
    lidar_range: float = 30.0
    num_obstacles: int = 8
    # Day 7: the coverage feature samples the 3x3x3 block of voxels centred
    # this far AHEAD of the drone along its heading. 8 m = 2 voxels, so the
    # block (1-3 voxels ahead) never includes the drone's own voxel, which
    # is always already marked. (The old feature read the own voxel and was
    # therefore constant 1.0.)
    coverage_lookahead: float = 8.0

    # --- Reset ---
    # Day 7: spawn each drone with a random yaw instead of all facing +x.
    randomize_initial_heading: bool = True

    # --- Safety ---
    min_separation: float = 2.0

    # --- Reward weights ---
    exploration_reward: float = 1.0
    collision_penalty: float = -100.0
    obstacle_collision_penalty: float = -75.0
    boundary_margin: float = 5.0
    # Day 6: was -0.1. Under random flight the boundary term was ~10x the
    # exploration income because it is charged every tick while exploration
    # only pays on rare new-voxel events. The architecture doc calls this
    # penalty "small", so it is now small relative to exploration.
    boundary_penalty_scale: float = -0.01
    team_coverage_bonus_scale: float = 0.05

    @property
    def voxel_grid_shape(self) -> tuple[int, int, int]:
        nx = int(self.world_size_xy // self.voxel_size)
        ny = int(self.world_size_xy // self.voxel_size)
        nz = int(self.world_height // self.voxel_size)
        return nx, ny, nz