"""Day-7 tests: what the observation MEANS, checked with scenarios whose
expected values can be worked out by hand.

Layout of one agent's observation (k = k_nearest):
  [0 : 3k]  per neighbour: distance, body-frame bearing_xy, body-frame bearing_z
  [3k]      nearest obstacle distance
  [3k+1]    coverage ahead   (fraction of explored voxels in the 3x3x3 block ahead)
  [3k+2]    battery
  [3k+3]    distance to nearest xy wall
  [3k+4]    body-frame bearing to that wall
  [3k+5]    altitude, normalised to [-1, 1]
  [3k+6]    own pitch
  [3k+7]    body-frame bearing to the nearest obstacle (added Day 8)
Convention: bearing 0 = dead ahead, +pi/2 = to the drone's LEFT.
"""
import numpy as np
import pytest

from backend.env.config import SwarmEnvConfig
from backend.env.swarm_env import SwarmEnv

K = 6
NEIGHBOR_DIST, NEIGHBOR_BXY, NEIGHBOR_BZ = 0, 1, 2     # offsets inside a neighbour triple
I_OBSTACLE, I_COVERAGE, I_BATTERY = 3 * K, 3 * K + 1, 3 * K + 2
I_WALL_DIST, I_WALL_BEARING, I_ALTITUDE, I_PITCH = 3 * K + 3, 3 * K + 4, 3 * K + 5, 3 * K + 6
I_OBSTACLE_BEARING = 3 * K + 7


def _env(num_agents=1, **kw):
    cfg = SwarmEnvConfig(
        num_agents=num_agents,
        num_obstacles=0,
        k_nearest=K,
        max_episode_steps=50,
        **kw,
    )
    env = SwarmEnv(cfg)
    env.reset(seed=0)

    # Day 7 observation tests set up their own explored/unexplored
    # scenarios, so clear coverage recorded during reset.
    env.coverage_grid[:] = False

    return env


def _observe(env):
    """Take one idle step and return observations without changing the
    manually configured coverage map used by these Day 7 tests."""
    actions = {a: np.zeros(3, dtype=np.float32) for a in env.agents}

    original_mark_coverage = env._mark_coverage
    env._mark_coverage = lambda: np.zeros(len(env.positions), dtype=bool)

    try:
        obs, *_ = env.step(actions)
    finally:
        env._mark_coverage = original_mark_coverage

    return obs


def _place(env, i, x, y, z, pitch=0.0, yaw=0.0):
    env.positions[i] = [x, y, z]
    env.headings[i] = [pitch, yaw]


# ----------------------------- shape / layout -----------------------------
def test_observation_dimension_is_3k_plus_8():
    env = _env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space("drone_0").shape == (3 * K + 8,)
    assert obs["drone_0"].shape == (3 * K + 8,)


# ----------------------------- body-frame bearings -----------------------------
def _two_drones(yaw0, other_xy, other_z=20.0, pitch0=0.0):
    env = _env(num_agents=2)
    _place(env, 0, 100.0, 100.0, 20.0, pitch=pitch0, yaw=yaw0)
    _place(env, 1, other_xy[0], other_xy[1], other_z)
    return env


def test_neighbour_dead_ahead_has_zero_bearing():
    env = _two_drones(yaw0=0.0, other_xy=(110.0, 100.0))          # facing +x, neighbour at +x
    o = _observe(env)["drone_0"]
    assert o[NEIGHBOR_DIST] == pytest.approx(10.0, abs=1e-4)
    assert o[NEIGHBOR_BXY] == pytest.approx(0.0, abs=1e-4)


def test_neighbour_to_the_left_has_positive_bearing():
    env = _two_drones(yaw0=0.0, other_xy=(100.0, 110.0))          # facing +x, neighbour at +y
    o = _observe(env)["drone_0"]
    assert o[NEIGHBOR_BXY] == pytest.approx(np.pi / 2, abs=1e-4)


def test_neighbour_to_the_right_has_negative_bearing():
    env = _two_drones(yaw0=0.0, other_xy=(100.0, 90.0))           # facing +x, neighbour at -y
    o = _observe(env)["drone_0"]
    assert o[NEIGHBOR_BXY] == pytest.approx(-np.pi / 2, abs=1e-4)


def test_bearing_is_relative_to_own_heading_not_the_world():
    """The same neighbour must read 'dead ahead' once the drone turns to face it.
    In the old world-frame observation this value never changed with heading."""
    facing_away = _two_drones(yaw0=np.pi, other_xy=(110.0, 100.0))   # neighbour is behind
    facing_it = _two_drones(yaw0=0.0, other_xy=(110.0, 100.0))       # neighbour is ahead
    behind = _observe(facing_away)["drone_0"][NEIGHBOR_BXY]
    ahead = _observe(facing_it)["drone_0"][NEIGHBOR_BXY]
    assert abs(behind) == pytest.approx(np.pi, abs=1e-4)
    assert ahead == pytest.approx(0.0, abs=1e-4)


def test_bearing_wraps_into_minus_pi_to_pi():
    env = _two_drones(yaw0=3.0 * np.pi / 2, other_xy=(110.0, 100.0))  # yaw beyond pi
    b = _observe(env)["drone_0"][NEIGHBOR_BXY]
    assert -np.pi - 1e-5 <= b <= np.pi + 1e-5
    # neighbour at world angle 0; facing 3pi/2 -> relative = -3pi/2 = +pi/2 after wrapping
    assert b == pytest.approx(np.pi / 2, abs=1e-4)


def test_vertical_bearing_is_relative_to_own_pitch():
    """Neighbour 10 m ahead and 10 m up sits at +45 deg elevation. A drone
    already pitched up 45 deg should see it dead ahead (0)."""
    level = _two_drones(yaw0=0.0, other_xy=(110.0, 100.0), other_z=30.0, pitch0=0.0)
    pitched = _two_drones(yaw0=0.0, other_xy=(110.0, 100.0), other_z=30.0, pitch0=np.pi / 4)
    assert _observe(level)["drone_0"][NEIGHBOR_BZ] == pytest.approx(np.pi / 4, abs=1e-3)
    assert _observe(pitched)["drone_0"][NEIGHBOR_BZ] == pytest.approx(0.0, abs=1e-3)


# ----------------------------- coverage ahead -----------------------------
# A drone at (100,100,20) facing +x looks at x=108. Voxel size is 4, so the
# block is voxels x 26..28, y 24..26, z 4..6 (27 cells); its own voxel (25,25,5)
# is outside the block.
def _facing_east_env():
    env = _env()
    _place(env, 0, 100.0, 100.0, 20.0, yaw=0.0)
    env.coverage_grid[:] = False
    return env


def test_coverage_ahead_is_zero_when_the_area_ahead_is_unexplored():
    env = _facing_east_env()
    assert _observe(env)["drone_0"][I_COVERAGE] == pytest.approx(0.0)


def test_coverage_ahead_is_one_when_the_area_ahead_is_fully_explored():
    env = _facing_east_env()
    env.coverage_grid[26:29, 24:27, 4:7] = True
    assert _observe(env)["drone_0"][I_COVERAGE] == pytest.approx(1.0)


def test_coverage_ahead_is_the_explored_fraction_of_the_block():
    env = _facing_east_env()
    env.coverage_grid[26:29, 24:27, 4:5] = True          # one 3x3 layer = 9 of 27 cells
    assert _observe(env)["drone_0"][I_COVERAGE] == pytest.approx(9.0 / 27.0, abs=1e-6)


def test_coverage_ahead_is_not_constant():
    """The regression this feature fixes: the old 'local density' read the
    drone's own, always-marked voxel and was exactly 1.0 forever."""
    env = _facing_east_env()
    empty = _observe(env)["drone_0"][I_COVERAGE]
    env.coverage_grid[26:29, 24:27, 4:7] = True
    full = _observe(env)["drone_0"][I_COVERAGE]
    assert full > empty


def test_coverage_ahead_depends_on_where_the_drone_is_facing():
    env = _facing_east_env()
    env.coverage_grid[26:29, 24:27, 4:7] = True          # explored to the EAST only
    east = _observe(env)["drone_0"][I_COVERAGE]
    env.headings[0, 1] = np.pi                            # turn to face WEST
    west = _observe(env)["drone_0"][I_COVERAGE]
    assert east == pytest.approx(1.0) and west == pytest.approx(0.0)


def test_world_edge_counts_as_explored():
    """Facing a wall must not look like unexplored territory worth flying to."""
    env = _env()
    _place(env, 0, 199.0, 100.0, 20.0, yaw=0.0)           # next to the +x wall, facing it
    env.coverage_grid[:] = False
    near_wall = _observe(env)["drone_0"][I_COVERAGE]
    assert near_wall > 0.3                                 # out-of-world cells pushed it up
    _place(env, 0, 100.0, 100.0, 20.0, yaw=0.0)
    in_open = _observe(env)["drone_0"][I_COVERAGE]
    assert near_wall > in_open


# ----------------------------- wall features -----------------------------
def test_wall_distance_and_bearing_when_wall_is_on_the_left():
    """At (3,100) facing +y, the x=0 wall is 3 m away on the drone's left."""
    env = _env()
    _place(env, 0, 3.0, 100.0, 20.0, yaw=np.pi / 2)
    o = _observe(env)["drone_0"]
    assert o[I_WALL_DIST] == pytest.approx(3.0, abs=1e-4)
    assert o[I_WALL_BEARING] == pytest.approx(np.pi / 2, abs=1e-4)


def test_wall_dead_ahead_has_zero_bearing():
    env = _env()
    _place(env, 0, 195.0, 100.0, 20.0, yaw=0.0)           # facing the +x wall, 5 m away
    o = _observe(env)["drone_0"]
    assert o[I_WALL_DIST] == pytest.approx(5.0, abs=1e-4)
    assert o[I_WALL_BEARING] == pytest.approx(0.0, abs=1e-4)


def test_wall_on_the_right_has_negative_bearing():
    env = _env()
    _place(env, 0, 3.0, 100.0, 20.0, yaw=-np.pi / 2)      # facing -y; x=0 wall is on the right
    o = _observe(env)["drone_0"]
    assert o[I_WALL_BEARING] == pytest.approx(-np.pi / 2, abs=1e-4)


def test_wall_reports_nothing_sensed_when_out_of_range():
    env = _env()
    _place(env, 0, 100.0, 100.0, 20.0, yaw=1.0)           # 100 m from every wall
    o = _observe(env)["drone_0"]
    assert o[I_WALL_DIST] == pytest.approx(env.config.lidar_range)   # clipped
    assert o[I_WALL_BEARING] == 0.0                                  # "nothing sensed" convention


def test_nearest_of_the_four_walls_is_chosen():
    env = _env()
    _place(env, 0, 100.0, 195.0, 20.0, yaw=0.0)           # 5 m from the y=W wall, to the drone's left
    o = _observe(env)["drone_0"]
    assert o[I_WALL_DIST] == pytest.approx(5.0, abs=1e-4)
    assert o[I_WALL_BEARING] == pytest.approx(np.pi / 2, abs=1e-4)


# ----------------------------- altitude + pitch -----------------------------
@pytest.mark.parametrize("z, expected", [(20.0, 0.0), (5.0, -0.75), (35.0, 0.75)])
def test_altitude_is_normalised_around_mid_height(z, expected):
    env = _env()
    _place(env, 0, 100.0, 100.0, z)
    assert _observe(env)["drone_0"][I_ALTITUDE] == pytest.approx(expected, abs=1e-4)


def test_observation_reports_own_pitch():
    env = _env()
    _place(env, 0, 100.0, 100.0, 20.0, pitch=0.5)
    assert _observe(env)["drone_0"][I_PITCH] == pytest.approx(0.5, abs=1e-4)


# ----------------------------- initial headings -----------------------------
def test_reset_spreads_initial_yaw_and_starts_level():
    env = SwarmEnv(SwarmEnvConfig(num_agents=50, num_obstacles=0))
    env.reset(seed=0)
    yaw = env.headings[:, 1]
    assert yaw.min() >= 0.0 and yaw.max() < 2 * np.pi
    assert yaw.std() > 1.0                                 # genuinely spread, not all facing +x
    assert np.all(env.headings[:, 0] == 0.0)               # pitch starts level


def test_initial_heading_is_reproducible_for_a_seed():
    cfg = SwarmEnvConfig(num_agents=20, num_obstacles=0)
    a, b = SwarmEnv(cfg), SwarmEnv(cfg)
    a.reset(seed=5)
    b.reset(seed=5)
    np.testing.assert_array_equal(a.headings, b.headings)


def test_randomized_heading_does_not_change_positions_or_obstacles():
    """Heading is drawn after positions/obstacles, so a seed gives the same world either way."""
    on = SwarmEnv(SwarmEnvConfig(num_agents=20, num_obstacles=5, randomize_initial_heading=True))
    off = SwarmEnv(SwarmEnvConfig(num_agents=20, num_obstacles=5, randomize_initial_heading=False))
    on.reset(seed=9)
    off.reset(seed=9)
    np.testing.assert_array_equal(on.positions, off.positions)
    np.testing.assert_array_equal(on.obstacles, off.obstacles)


def test_randomize_initial_heading_can_be_switched_off():
    env = SwarmEnv(SwarmEnvConfig(num_agents=10, num_obstacles=0, randomize_initial_heading=False))
    env.reset(seed=0)
    assert np.all(env.headings == 0.0)



# ----------------------------- obstacle bearing (Day 8) -----------------------------
def _obstacle_env(*obstacles, yaw=0.0):
    """One drone at (100,100,20) with the given obstacle centres (all >= 10 m away,
    so no collision interferes)."""
    cfg = SwarmEnvConfig(num_agents=1, num_obstacles=len(obstacles), k_nearest=K, max_episode_steps=50)
    env = SwarmEnv(cfg)
    env.reset(seed=0)
    env.obstacles[:] = obstacles
    _place(env, 0, 100.0, 100.0, 20.0, yaw=yaw)
    return env


def test_obstacle_dead_ahead_has_zero_bearing():
    o = _observe(_obstacle_env((110.0, 100.0, 20.0)))["drone_0"]
    assert o[I_OBSTACLE] == pytest.approx(10.0, abs=1e-4)
    assert o[I_OBSTACLE_BEARING] == pytest.approx(0.0, abs=1e-4)


def test_obstacle_to_the_left_has_positive_bearing():
    o = _observe(_obstacle_env((100.0, 110.0, 20.0)))["drone_0"]
    assert o[I_OBSTACLE_BEARING] == pytest.approx(np.pi / 2, abs=1e-4)


def test_obstacle_to_the_right_has_negative_bearing():
    o = _observe(_obstacle_env((100.0, 90.0, 20.0)))["drone_0"]
    assert o[I_OBSTACLE_BEARING] == pytest.approx(-np.pi / 2, abs=1e-4)


def test_obstacle_behind_has_bearing_of_magnitude_pi():
    o = _observe(_obstacle_env((90.0, 100.0, 20.0)))["drone_0"]
    assert abs(o[I_OBSTACLE_BEARING]) == pytest.approx(np.pi, abs=1e-4)


def test_obstacle_bearing_is_relative_to_own_heading():
    """Same obstacle, east of the drone. Facing +x it is dead ahead; after
    turning to face +y it is on the drone's right."""
    ahead = _observe(_obstacle_env((110.0, 100.0, 20.0), yaw=0.0))["drone_0"][I_OBSTACLE_BEARING]
    turned = _observe(_obstacle_env((110.0, 100.0, 20.0), yaw=np.pi / 2))["drone_0"][I_OBSTACLE_BEARING]
    assert ahead == pytest.approx(0.0, abs=1e-4)
    assert turned == pytest.approx(-np.pi / 2, abs=1e-4)


def test_obstacle_out_of_range_reports_nothing_sensed():
    o = _observe(_obstacle_env((150.0, 100.0, 20.0)))["drone_0"]     # 50 m away, range is 30
    assert o[I_OBSTACLE] == pytest.approx(30.0)
    assert o[I_OBSTACLE_BEARING] == 0.0


def test_no_obstacles_reports_zero_bearing():
    env = _env()                                                      # zero obstacles
    _place(env, 0, 100.0, 100.0, 20.0)
    o = _observe(env)["drone_0"]
    assert o[I_OBSTACLE] == pytest.approx(30.0)
    assert o[I_OBSTACLE_BEARING] == 0.0


def test_bearing_points_at_the_nearest_of_several_obstacles():
    """10 m ahead vs 15 m to the left: the nearer one (ahead) is reported."""
    o = _observe(_obstacle_env((110.0, 100.0, 20.0), (100.0, 115.0, 20.0)))["drone_0"]
    assert o[I_OBSTACLE] == pytest.approx(10.0, abs=1e-4)
    assert o[I_OBSTACLE_BEARING] == pytest.approx(0.0, abs=1e-4)