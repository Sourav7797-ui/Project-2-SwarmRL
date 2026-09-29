from dataclasses import dataclass

@dataclass(frozen=True)
class SwarmEnvConfig:
    num_agents: int = 50
    k_nearest: int = 6

    world_size_xy: float = 200.0
    world_height: float = 40.0
    voxel_size: float = 4.0

    max_velocity: float = 8.0       
    max_pitch_rate: float = 1.0
    max_yaw_rate: float = 1.5
    dt: float = 0.1

    max_acceleration: float = 4.0  
    drag_coefficient: float = 0.02  

    drone_radius: float = 0.5       
    obstacle_radius: float = 3.0    

    max_episode_steps: int = 1000
    battery_capacity: float = 1.0
   
    lidar_range: float = 30.0
    num_obstacles: int = 8

    min_separation: float = 2.0

    @property
    def voxel_grid_shape(self) -> tuple[int, int, int]:
        nx = int(self.world_size_xy // self.voxel_size)
        ny = int(self.world_size_xy // self.voxel_size)
        nz = int(self.world_height // self.voxel_size)
        return nx, ny, nz