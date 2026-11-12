
import numpy as np

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv


def test_sensor_coverage_marks_multiple_voxels():
    config = SwarmEnvConfig(
        num_agents=1,
        num_obstacles=0,
        world_size_xy=40.0,
        world_height=20.0,
        voxel_size=4.0,
        mapping_sensor_range=16.0,
        mapping_horizontal_fov_deg=120.0,
        mapping_vertical_fov_deg=90.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=1)

    env.positions[0] = [20.0, 20.0, 10.0]
    env.headings[0] = [0.0, 0.0]
    env.coverage_grid.fill(False)

    newly_covered = env._mark_coverage()

    assert newly_covered.shape == (1,)
    assert newly_covered.dtype == bool
    assert newly_covered[0]
    assert env.coverage_grid.sum() > 1


def test_sensor_coverage_does_not_mark_voxels_behind_drone():
    config = SwarmEnvConfig(
        num_agents=1,
        num_obstacles=0,
        world_size_xy=40.0,
        world_height=20.0,
        voxel_size=4.0,
        mapping_sensor_range=8.0,
        mapping_horizontal_fov_deg=90.0,
        mapping_vertical_fov_deg=60.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=2)

    env.positions[0] = [20.0, 20.0, 10.0]
    env.headings[0] = [0.0, 0.0]
    env.coverage_grid.fill(False)

    env._mark_coverage()

    # The drone faces the +x direction. A voxel well behind it must
    # remain uncovered.
    assert not env.coverage_grid[2, 5, 2]


def test_sensor_coverage_is_monotonic_and_repeatable():
    config = SwarmEnvConfig(num_agents=3, num_obstacles=0)
    env = SwarmEnv(config=config)
    env.reset(seed=3)

    first_count = int(env.coverage_grid.sum())
    env._mark_coverage()
    second_count = int(env.coverage_grid.sum())

    assert second_count >= first_count
def test_sensor_coverage_respects_range_limit():
    config = SwarmEnvConfig(
        num_agents=1,
        num_obstacles=0,
        world_size_xy=40.0,
        world_height=20.0,
        voxel_size=4.0,
        mapping_sensor_range=4.0,
        mapping_horizontal_fov_deg=360.0,
        mapping_vertical_fov_deg=180.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=11)

    env.positions[0] = [20.0, 20.0, 10.0]
    env.headings[0] = [0.0, 0.0]
    env.coverage_grid.fill(False)

    env._mark_coverage()

    # Voxel [5, 5, 2] has center (22, 22, 10), within range.
    assert env.coverage_grid[5, 5, 2]

    # Voxel [6, 5, 2] has center (26, 22, 10), outside range.
    assert not env.coverage_grid[6, 5, 2]


def test_sensor_coverage_respects_vertical_field_of_view():
    config = SwarmEnvConfig(
        num_agents=1,
        num_obstacles=0,
        world_size_xy=40.0,
        world_height=20.0,
        voxel_size=4.0,
        mapping_sensor_range=16.0,
        mapping_horizontal_fov_deg=120.0,
        mapping_vertical_fov_deg=30.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=12)

    env.positions[0] = [20.0, 20.0, 10.0]
    env.headings[0] = [0.0, 0.0]
    env.coverage_grid.fill(False)

    env._mark_coverage()

    # A voxel directly ahead and near the horizontal plane is visible.
    assert env.coverage_grid[6, 5, 2]

    # Voxel center (22, 22, 18) is too far above the sensor's vertical FOV.
    assert not env.coverage_grid[5, 5, 4]


def test_sensor_coverage_handles_yaw_wraparound():
    config = SwarmEnvConfig(
        num_agents=1,
        num_obstacles=0,
        world_size_xy=40.0,
        world_height=20.0,
        voxel_size=4.0,
        mapping_sensor_range=16.0,
        mapping_horizontal_fov_deg=90.0,
        mapping_vertical_fov_deg=90.0,
    )
    env = SwarmEnv(config=config)
    env.reset(seed=13)

    env.positions[0] = [20.0, 20.0, 10.0]
    env.headings[0] = [0.0, 2.0 * np.pi - 0.1]
    env.coverage_grid.fill(False)

    env._mark_coverage()

    # This voxel lies slightly below the +x axis. Its angle crosses
    # the 0/2*pi boundary relative to the drone's yaw.
    assert env.coverage_grid[6, 4, 2]
