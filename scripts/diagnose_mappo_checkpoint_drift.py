
"""Compare MAPPO actor and critic checkpoint weights.

Read-only diagnostic:
- Reads iteration 5 and iteration 10 module_state.pkl files.
- Uses a restricted pickle unpickler.
- Does not initialize Ray or construct an RLlib Algorithm.
- Does not restore checkpoints, train, or evaluate.
"""

from __future__ import annotations

import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np


DEFAULT_RUN = Path(
    "checkpoints/mappo_validation/run_20261108_002340_785217"
)

RELATIVE_EPSILON = 1e-12


class RestrictedCheckpointUnpickler(pickle.Unpickler):
    """Allow only the globals required by the inspected module-state format."""

    ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"): OrderedDict,
        ("numpy", "dtype"): np.dtype,
        ("numpy._core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
        ("numpy.core.numeric", "_frombuffer"): np.core.numeric._frombuffer,
    }

    def find_class(self, module: str, name: str):
        key = (module, name)
        if key not in self.ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(
                f"Blocked unexpected pickle global: {module}.{name}"
            )
        return self.ALLOWED_GLOBALS[key]


def load_module_state(path: Path) -> dict[str, np.ndarray]:
    """Load only the module state using the restricted unpickler."""

    with path.open("rb") as handle:
        state = RestrictedCheckpointUnpickler(handle).load()

    if not isinstance(state, (dict, OrderedDict)):
        raise TypeError(f"Unexpected checkpoint state type: {type(state)}")

    result = {}

    for name, value in state.items():
        if not isinstance(name, str):
            raise TypeError(f"Unexpected parameter name type: {type(name)}")

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Parameter {name!r} is not a NumPy array: {type(value)}"
            )

        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"Non-numeric parameter: {name}")

        result[name] = np.asarray(value, dtype=np.float64)

    return result


def compare_parameter(
    name: str,
    before: np.ndarray,
    after: np.ndarray,
) -> dict:
    if before.shape != after.shape:
        raise ValueError(
            f"Shape mismatch for {name}: {before.shape} vs {after.shape}"
        )

    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError(f"Non-finite values in parameter {name}")

    difference = after - before

    norm_before = float(np.linalg.norm(before))
    norm_after = float(np.linalg.norm(after))
    difference_norm = float(np.linalg.norm(difference))

    denominator = max(norm_before, RELATIVE_EPSILON)
    relative_change = difference_norm / denominator

    flat_before = before.ravel()
    flat_after = after.ravel()

    cosine_denominator = (
        np.linalg.norm(flat_before) * np.linalg.norm(flat_after)
    )

    cosine_similarity = (
        float(np.dot(flat_before, flat_after) / cosine_denominator)
        if cosine_denominator > RELATIVE_EPSILON
        else float("nan")
    )

    return {
        "name": name,
        "shape": before.shape,
        "elements": before.size,
        "norm_before": norm_before,
        "norm_after": norm_after,
        "difference_norm": difference_norm,
        "relative_change": relative_change,
        "cosine_similarity": cosine_similarity,
        "max_abs_change": float(np.max(np.abs(difference))),
        "mean_abs_change": float(np.mean(np.abs(difference))),
    }


def print_section(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)

    print(
        f"{'Parameter':<20} {'Shape':<16} {'Rel.Change':>12} "
        f"{'Cosine':>12} {'Max |Δ|':>12} {'Mean |Δ|':>12}"
    )
    print("-" * 90)

    for row in rows:
        print(
            f"{row['name']:<20} "
            f"{str(row['shape']):<16} "
            f"{row['relative_change']:>12.6f} "
            f"{row['cosine_similarity']:>12.6f} "
            f"{row['max_abs_change']:>12.6f} "
            f"{row['mean_abs_change']:>12.6f}"
        )

    if rows:
        before_norm = np.sqrt(sum(r["norm_before"] ** 2 for r in rows))
        after_norm = np.sqrt(sum(r["norm_after"] ** 2 for r in rows))
        difference_norm = np.sqrt(
            sum(r["difference_norm"] ** 2 for r in rows)
        )

        relative = difference_norm / max(
            before_norm, RELATIVE_EPSILON
        )

        print("-" * 90)
        print(f"Combined relative L2 change: {relative:.8f}")
        print(f"Combined norm, iteration 5:  {before_norm:.8f}")
        print(f"Combined norm, iteration 10: {after_norm:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MAPPO checkpoint parameter drift."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN,
        help="Training run directory containing iteration_0005 and iteration_0010",
    )
    args = parser.parse_args()

    checkpoint_5 = (
        args.run_dir
        / "iteration_0005"
        / "learner_group"
        / "learner"
        / "rl_module"
        / "shared_policy"
        / "module_state.pkl"
    )

    checkpoint_10 = (
        args.run_dir
        / "iteration_0010"
        / "learner_group"
        / "learner"
        / "rl_module"
        / "shared_policy"
        / "module_state.pkl"
    )

    for path in (checkpoint_5, checkpoint_10):
        if not path.is_file():
            raise FileNotFoundError(path)

    print("Loading iteration 5 module state...")
    state_5 = load_module_state(checkpoint_5)

    print("Loading iteration 10 module state...")
    state_10 = load_module_state(checkpoint_10)

    if state_5.keys() != state_10.keys():
        raise ValueError(
            "Parameter names differ between checkpoints.\n"
            f"Iteration 5: {list(state_5)}\n"
            f"Iteration 10: {list(state_10)}"
        )

    rows = [
        compare_parameter(name, state_5[name], state_10[name])
        for name in state_5
    ]

    actor_rows = [r for r in rows if r["name"].startswith("actor.")]
    critic_rows = [r for r in rows if r["name"].startswith("critic.")]

    print("\nMAPPO CHECKPOINT DRIFT")
    print(f"Run: {args.run_dir}")
    print("Comparison: iteration 5 -> iteration 10")
    print(f"Parameters compared: {len(rows)}")

    print_section("ACTOR PARAMETER DRIFT", actor_rows)
    print_section("CRITIC PARAMETER DRIFT", critic_rows)

    # The final actor layer has six outputs:
    # rows 0-2 are action means; rows 3-5 are log standard deviations.
    final_weight = "actor.4.weight"
    final_bias = "actor.4.bias"

    print("\n" + "=" * 90)
    print("ACTOR OUTPUT-HEAD BREAKDOWN")
    print("=" * 90)

    for name in (final_weight, final_bias):
        if name not in state_5:
            continue

        before = state_5[name]
        after = state_10[name]

        if before.shape[0] != 6:
            print(f"{name}: unexpected output dimension {before.shape}")
            continue

        for label, indices in (
            ("Action means", slice(0, 3)),
            ("Log standard deviations", slice(3, 6)),
        ):
            result = compare_parameter(
                f"{name} [{label}]",
                before[indices],
                after[indices],
            )
            print(
                f"{name:<18} {label:<25} "
                f"relative change={result['relative_change']:.8f} "
                f"cosine={result['cosine_similarity']:.8f} "
                f"max |Δ|={result['max_abs_change']:.8f}"
            )

    print("\nDiagnostic completed. No checkpoint files were modified.")


if __name__ == "__main__":
    main()