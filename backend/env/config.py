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

    # --- Dynamics (Day 8). Defaults = the static, windless world. The
    # curriculum stages switch these on; see backend/reward/curriculum.py. ---
    # Moving obstacles: each drifts in a fixed random 3-D direction at
    # `obstacle_speed` and bounces off the world boundary.
    obstacles_move: bool = False
    obstacle_speed: float = 0.0          # m/s
    # Wind: a horizontal drift added to every drone's position each tick. It
    # does not change the drone's own commanded/actual speed, and the policy
    # does not observe it directly. Its direction random-walks by
    # `wind_direction_drift` rad (std) per step.
    wind_speed: float = 0.0              # m/s
    wind_direction_drift: float = 0.05

    # --- Safety ---
    min_separation: float = 2.0
    # Day 8: a collision is a NEW reward event only if the drone has been clear
    # of contact for at least this many ticks. A drone sliding along an obstacle
    # surface flickers in and out of contact (the push-out leaves it exactly on
    # the surface), and without a cooldown every flicker was billed again.
    # 1 reproduces the plain "rising edge" behaviour of Day 6. 10 ticks = 1 s.
    contact_cooldown_ticks: int = 10

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