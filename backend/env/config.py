from dataclasses import dataclass


@dataclass(frozen=True)
class SwarmEnvConfig:
    # --- Swarm ---
    num_agents: int = 50
    k_nearest: int = 6              # neighbors included in each agent's observation

    # --- World (continuous 3D disaster zone) ---
    world_size_xy: float = 200.0    # meters, world spans [0, world_size_xy] in x and y
    world_height: float = 40.0      # meters, world spans [0, world_height] in z
    voxel_size: float = 4.0         # meters per coverage-grid cell edge

    # --- Physics limits (used to clip the continuous action space) ---
    max_velocity: float = 8.0       # m/s
    max_pitch_rate: float = 1.0     # rad/s
    max_yaw_rate: float = 1.5       # rad/s
    dt: float = 0.1                 # seconds per simulation tick

    # --- Episode ---
    max_episode_steps: int = 1000
    battery_capacity: float = 1.0   # normalized, drains linearly with dt

    # --- Sensing ---
    lidar_range: float = 30.0       # meters, max sensed distance to neighbors/obstacles
    num_obstacles: int = 8

    # --- Safety ---
    min_separation: float = 2.0     # meters, closer than this = collision

    @property
    def voxel_grid_shape(self) -> tuple[int, int, int]:
        nx = int(self.world_size_xy // self.voxel_size)
        ny = int(self.world_size_xy // self.voxel_size)
        nz = int(self.world_height // self.voxel_size)
        return nx, ny, nz