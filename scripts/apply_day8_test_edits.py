"""One-off helper: apply the Day 8 edits to the EXISTING test files.

Run from anywhere:
    python scripts/apply_day8_test_edits.py

For each edit it reports one of:
    applied          the edit was made
    already applied  nothing to do (safe to run the script again)
    NOT FOUND        the file does not contain the text this edit expects

A file is only rewritten if ALL of its edits succeed, so a failure never leaves
a half-edited file. Line endings (LF or CRLF) are preserved. You can delete this
script once the test suite passes.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if len(sys.argv) > 2 and sys.argv[1] == "--root":      # used only to test this script
    ROOT = Path(sys.argv[2]).resolve()

# --------------------------------------------------------------------------
# Day 3: the loop oracle learns about the obstacle bearing
# --------------------------------------------------------------------------
D3_OLD = '''        # nearest obstacle
        if len(env.obstacles) > 0:
            obstacle_dists = np.linalg.norm(env.obstacles - pos_i, axis=1)
            nearest_obstacle = float(min(obstacle_dists.min(), c.lidar_range))
        else:
            nearest_obstacle = c.lidar_range
'''
D3_NEW = '''        # nearest obstacle: distance + body-frame bearing (0 if none within range)
        if len(env.obstacles) > 0:
            to_obs = env.obstacles - pos_i
            obstacle_dists = np.linalg.norm(to_obs, axis=1)
            j = min(range(len(env.obstacles)), key=lambda t: obstacle_dists[t])   # first minimum
            nearest_obstacle = min(float(obstacle_dists[j]), c.lidar_range)
            if obstacle_dists[j] < c.lidar_range:
                obstacle_bearing = _wrap(float(np.arctan2(to_obs[j, 1], to_obs[j, 0])) - yaw_i)
            else:
                obstacle_bearing = 0.0
        else:
            nearest_obstacle = c.lidar_range
            obstacle_bearing = 0.0
'''

# --------------------------------------------------------------------------
# Day 6: a re-contact is a new event only after a full cooldown of clear ticks
# --------------------------------------------------------------------------
D6_OLD = '''    env.positions[:] = [[100.0, 100.0, 20.0], [150.0, 100.0, 20.0]]   # pull them apart
    _, _, _, _, infos = _step_idle(env, 2)
    assert infos["drone_0"]["collision"] is False
    assert infos["drone_0"]["collision_event"] is False

    env.positions[:] = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]   # collide a second time
'''
D6_NEW = '''    # Pull them apart for a full cooldown (Day 8: a brief separation is the same contact).
    env.positions[:] = [[100.0, 100.0, 20.0], [150.0, 100.0, 20.0]]
    for _ in range(env.config.contact_cooldown_ticks):
        _, _, _, _, infos = _step_idle(env, 2)
        assert infos["drone_0"]["collision"] is False
        assert infos["drone_0"]["collision_event"] is False

    env.positions[:] = [[100.0, 100.0, 20.0], [100.5, 100.0, 20.0]]   # collide a second time
'''

# --------------------------------------------------------------------------
# Day 7 observation tests: new layout line, new constant, new dimension, 8 new tests
# --------------------------------------------------------------------------
D7_CONST_OLD = "I_WALL_DIST, I_WALL_BEARING, I_ALTITUDE, I_PITCH = 3 * K + 3, 3 * K + 4, 3 * K + 5, 3 * K + 6\n"
D7_CONST_NEW = D7_CONST_OLD + "I_OBSTACLE_BEARING = 3 * K + 7\n"

D7_APPEND = '''


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
'''

# Each edit: (description, kind, args). kind "replace": (old, new, done_marker[, replace_all])
#                                        kind "regex":   (pattern, replacement, done_marker)
#                                        kind "append":  (block, done_marker)
EDITS = {
    "tests/test_day3_edge_cases.py": [
        ("oracle: obstacle distance + bearing", "replace", (D3_OLD, D3_NEW, "obstacle_bearing = _wrap(")),
        ("oracle: add bearing to the observation array", "replace",
         ("wall_dist, wall_bearing, altitude_norm, pitch_i],",
          "wall_dist, wall_bearing, altitude_norm, pitch_i, obstacle_bearing],",
          "pitch_i, obstacle_bearing]")),
    ],
    "tests/test_day4_integration.py": [
        ("contract: observation dimension is 3k + 8", "regex",
         (r"k_nearest \* 3 \+ [37]\b", "k_nearest * 3 + 8", "k_nearest * 3 + 8")),
    ],
    "tests/test_day6_env_fixes.py": [
        ("recontact test waits out the cooldown", "replace",
         (D6_OLD, D6_NEW, "range(env.config.contact_cooldown_ticks)")),
    ],
    "tests/test_day7_observations.py": [
        ("docstring: layout lists the obstacle bearing", "replace",
         ("  [3k+6]    own pitch\nConvention:",
          "  [3k+6]    own pitch\n  [3k+7]    body-frame bearing to the nearest obstacle (added Day 8)\nConvention:",
          "[3k+7]    body-frame bearing to the nearest obstacle")),
        ("constant I_OBSTACLE_BEARING", "replace", (D7_CONST_OLD, D7_CONST_NEW, "I_OBSTACLE_BEARING = 3 * K + 7")),
        ("dimension test renamed to 3k_plus_8", "replace",
         ("def test_observation_dimension_is_3k_plus_7():",
          "def test_observation_dimension_is_3k_plus_8():",
          "def test_observation_dimension_is_3k_plus_8():")),
        ("dimension test expects 3K + 8", "replace",
         ("(3 * K + 7,)", "(3 * K + 8,)", "(3 * K + 8,)", True)),
        ("8 new obstacle-bearing tests", "append", (D7_APPEND, "def test_obstacle_dead_ahead_has_zero_bearing")),
    ],
}


def apply_edit(text: str, kind: str, args: tuple) -> tuple[str, str]:
    """Return (new_text, status) where status is applied / already applied / NOT FOUND."""
    if kind == "replace":
        old, new, done = args[0], args[1], args[2]
        replace_all = len(args) > 3 and args[3]
        if done in text:                       # check the marker FIRST: some edits' new text contains their old text
            return text, "already applied"
        if old not in text:
            return text, "NOT FOUND"
        return (text.replace(old, new) if replace_all else text.replace(old, new, 1)), "applied"
    if kind == "regex":
        pattern, repl, done = args
        if re.search(pattern, text) is None:
            return text, ("already applied" if done in text else "NOT FOUND")
        new_text = re.sub(pattern, repl, text)
        return new_text, ("already applied" if new_text == text else "applied")
    if kind == "append":
        block, done = args
        if done in text:
            return text, "already applied"
        return text.rstrip("\n") + "\n" + block, "applied"
    raise ValueError(kind)


def main() -> int:
    failures = 0
    for rel_path, edits in EDITS.items():
        path = ROOT / rel_path
        print(f"\n{rel_path}")
        if not path.exists():
            print("   MISSING FILE: this file is not in your tests folder")
            failures += 1
            continue

        raw = path.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")

        file_ok, any_applied = True, False
        for description, kind, args in edits:
            text, status = apply_edit(text, kind, args)
            print(f"   {status:16s} {description}")
            if status == "NOT FOUND":
                file_ok = False
            elif status == "applied":
                any_applied = True

        if not file_ok:
            print("   -> NOT written (at least one edit did not match). Send me this file's output.")
            failures += 1
        elif any_applied:
            out = text.replace("\n", "\r\n") if crlf else text
            path.write_bytes(out.encode("utf-8"))
            print("   -> written")
        else:
            print("   -> nothing to change")

    print("\n" + ("All edits are in place." if failures == 0 else f"{failures} file(s) need attention."))
    print("Next: python -m pytest tests/ -q     (expect 185 passed)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())