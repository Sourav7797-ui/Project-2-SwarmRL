"""Starting Ray reliably on a real machine (Task 3, Day 9).

Everything that has to be true BEFORE `ray.init()` lives here, so the smoke
script, the tests and the Day 10 training script start Ray the same way.

  * Ray worker processes are fresh Python processes (always on Windows). They
    must be able to `import backend`, so the project root goes on PYTHONPATH.
  * RLlib imports TensorFlow in every process if it is installed, even though
    this project uses torch only. That costs seconds per process and a lot of
    RAM. `RLLIB_TEST_NO_TF_IMPORT` is RLlib's switch for skipping it; worker
    processes inherit it.
  * Ray sizes its object store as a fraction of the RAM that is free right now,
    and refuses to start if too little remains. Our batches are a few MB, so a
    small fixed object store is plenty and leaves more RAM for the workers.
  * That refusal is an unhelpful ValueError; we turn it into an error that says
    what to do about it.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # backend/training/ray_runtime.py -> project root

_MEMORY_ERROR_MARKER = "available for tasks and actors"


class InsufficientMemoryError(RuntimeError):
    """Ray could not start because the machine has too little free RAM right now."""


def configure_environment(*, skip_tensorflow: bool = True, project_root: Path | None = None) -> None:
    """Set the environment variables Ray worker processes inherit. Safe to call
    repeatedly. Existing values are respected, never overwritten."""
    root = str(project_root or ROOT)
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([root] + parts)
    if skip_tensorflow:
        os.environ.setdefault("RLLIB_TEST_NO_TF_IMPORT", "1")


def available_memory_gb() -> float | None:
    """Free RAM in GB, or None if psutil is not installed."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 1024 ** 3



def _choose_object_store_mb(
    requested_mb: int | None = None,
    *,
    free_gb: float | None = None,
) -> int:
    """Choose a conservative object-store size based on currently available RAM."""
    if requested_mb is not None:
        if requested_mb < 80:
            raise ValueError("object_store_mb must be at least 80 MB.")
        return requested_mb

    # If psutil is unavailable, use a conservative fallback.
    if free_gb is None:
        free_gb = available_memory_gb()

    if free_gb is None:
        return 128

    # Leave room for Ray processes, training workers and the operating system.
    # Fail early when the machine is already critically short on free RAM.
    if free_gb < 0.75:
        raise InsufficientMemoryError(
            f"Ray startup postponed: only {free_gb:.2f} GB of RAM is available. "
            "Close memory-heavy applications and retry when more RAM is free."
        )

    # Use at most 10% of currently available RAM, capped at 256 MB.
    # Keep the allocation above Ray's practical minimum.
    chosen_mb = int(free_gb * 1024 * 0.10)
    return max(80, min(chosen_mb, 256))



def init_ray(
    *,
    num_cpus: int | None = None,
    object_store_mb: int | None = None,
    **kwargs,
):
    """Start Ray with adaptive memory settings and actionable errors."""
    configure_environment()
    import ray

    free = available_memory_gb()

    try:
        selected_mb = _choose_object_store_mb(
            object_store_mb,
            free_gb=free,
        )
    except InsufficientMemoryError as exc:
        raise InsufficientMemoryError(
            f"{exc} Close other applications, check Task Manager, "
            "and retry. An explicit object-store override cannot fix "
            "insufficient total RAM."
        ) from exc

    try:
        return ray.init(
            num_cpus=num_cpus,
            object_store_memory=selected_mb * 1024 * 1024,
            include_dashboard=False,
            logging_level=logging.ERROR,
            ignore_reinit_error=True,
            **kwargs,
        )
    except ValueError as exc:
        if _MEMORY_ERROR_MARKER not in str(exc):
            raise

        free_text = (
            f"about {free:.2f} GB"
            if free is not None
            else "an unknown amount"
        )

        raise InsufficientMemoryError(
            f"Ray could not start: this machine has {free_text} of free RAM "
            f"and an object store of {selected_mb} MB was requested. "
            "Close other applications, run `ray stop --force` if necessary, "
            "and retry. You can also lower the object store with "
            f"--object-store-mb (currently {selected_mb}). "
            f"Ray's message: {exc}"
        ) from exc
