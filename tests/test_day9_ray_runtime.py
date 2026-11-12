
"""Day-9 tests: the Ray startup helper. Ray is never actually started here:
`ray.init` is replaced with a fake so we can check what the helper passes to it
and how it reacts to failures."""
import os

import pytest

ray = pytest.importorskip("ray")

from backend.training import ray_runtime  # noqa: E402
from backend.training.ray_runtime import (  # noqa: E402
    InsufficientMemoryError,
    available_memory_gb,
    configure_environment,
    init_ray,
)

# The exact message Ray raised on the user's Windows machine when free RAM ran out.
REAL_MEMORY_ERROR = (
    "After taking into account object store and redis memory usage, the amount of memory "
    "on this node available for tasks and actors (0.09 GB) is less than 0% of total. "
    "You can adjust these settings with ray.init(memory=<bytes>, object_store_memory=<bytes>)."
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("RLLIB_TEST_NO_TF_IMPORT", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)


# ------------------------------- environment -------------------------------
def test_project_root_is_the_folder_that_contains_backend():
    assert (ray_runtime.ROOT / "backend" / "training" / "ray_runtime.py").exists()


def test_tensorflow_import_is_skipped_by_default():
    configure_environment()
    assert "RLLIB_TEST_NO_TF_IMPORT" in os.environ


def test_tensorflow_skip_can_be_turned_off():
    configure_environment(skip_tensorflow=False)
    assert "RLLIB_TEST_NO_TF_IMPORT" not in os.environ


def test_an_existing_tensorflow_setting_is_respected_not_overwritten(monkeypatch):
    monkeypatch.setenv("RLLIB_TEST_NO_TF_IMPORT", "already-set")
    configure_environment()
    assert os.environ["RLLIB_TEST_NO_TF_IMPORT"] == "already-set"


def test_project_root_is_put_on_pythonpath_for_worker_processes():
    configure_environment()
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(ray_runtime.ROOT)


def test_pythonpath_is_extended_without_duplicates_and_keeps_existing_entries(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["/somewhere/else"]))
    configure_environment()
    configure_environment()  # calling twice must not duplicate
    parts = os.environ["PYTHONPATH"].split(os.pathsep)
    assert parts.count(str(ray_runtime.ROOT)) == 1
    assert "/somewhere/else" in parts


def test_a_custom_project_root_can_be_given(tmp_path):
    configure_environment(project_root=tmp_path)
    assert os.environ["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path)


# ------------------------------- init_ray -------------------------------
class FakeInit:
    def __init__(self, raises=None):
        self.calls, self.raises = [], raises

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return "ray-context"


def test_init_ray_passes_a_bounded_object_store_and_quiet_settings(monkeypatch):
    fake = FakeInit()
    monkeypatch.setattr(ray, "init", fake)
    result = init_ray(num_cpus=3, object_store_mb=256)

    assert result == "ray-context"
    (kwargs,) = fake.calls
    assert kwargs["num_cpus"] == 3
    assert kwargs["object_store_memory"] == 256 * 1024 * 1024
    assert kwargs["include_dashboard"] is False
    assert kwargs["ignore_reinit_error"] is True


def test_init_ray_default_object_store_is_adaptive(monkeypatch):
    fake = FakeInit()
    monkeypatch.setattr(ray, "init", fake)
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.0)

    init_ray()

    object_store_bytes = fake.calls[0]["object_store_memory"]
    assert 80 * 1024 * 1024 <= object_store_bytes <= 256 * 1024 * 1024
    assert fake.calls[0]["num_cpus"] is None  # None = let Ray use every core


def test_init_ray_default_object_store_uses_available_memory(monkeypatch):
    fake = FakeInit()
    monkeypatch.setattr(ray, "init", fake)
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.0)

    init_ray()

    assert fake.calls[0]["object_store_memory"] == int(2.0 * 1024 * 0.10) * 1024 * 1024


def test_init_ray_uses_fallback_when_memory_probe_is_unavailable(monkeypatch):
    fake = FakeInit()
    monkeypatch.setattr(ray, "init", fake)
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: None)

    init_ray()

    assert fake.calls[0]["object_store_memory"] == 128 * 1024 * 1024


def test_init_ray_rejects_critically_low_memory_before_starting_ray(monkeypatch):
    fake = FakeInit()
    monkeypatch.setattr(ray, "init", fake)
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 0.4)

    with pytest.raises(InsufficientMemoryError, match="0.40 GB"):
        init_ray()

    assert fake.calls == []


def test_init_ray_configures_the_environment_before_starting_ray(monkeypatch):
    seen = {}

    def fake_init(**kwargs):
        seen["tf_flag"] = os.environ.get("RLLIB_TEST_NO_TF_IMPORT")
        seen["pythonpath"] = os.environ.get("PYTHONPATH")
        return None

    monkeypatch.setattr(ray, "init", fake_init)
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.0)

    init_ray()

    assert seen["tf_flag"] is not None
    assert str(ray_runtime.ROOT) in seen["pythonpath"]


def test_low_memory_error_becomes_an_actionable_error(monkeypatch):
    monkeypatch.setattr(
        ray, "init", FakeInit(raises=ValueError(REAL_MEMORY_ERROR))
    )
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.32)

    with pytest.raises(InsufficientMemoryError) as info:
        init_ray(object_store_mb=300)

    message = str(info.value)
    assert "free RAM" in message
    assert "ray stop --force" in message
    assert "--object-store-mb" in message and "300" in message
    assert "0.09 GB" in message  # Ray's original message is preserved
    assert isinstance(info.value.__cause__, ValueError)


def test_insufficient_memory_error_is_a_runtime_error():
    assert issubclass(InsufficientMemoryError, RuntimeError)


def test_other_value_errors_pass_through_untouched(monkeypatch):
    monkeypatch.setattr(
        ray, "init", FakeInit(raises=ValueError("num_cpus must be positive"))
    )
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.0)

    with pytest.raises(ValueError, match="num_cpus must be positive") as info:
        init_ray()

    assert not isinstance(info.value, InsufficientMemoryError)


def test_non_value_errors_pass_through_untouched(monkeypatch):
    monkeypatch.setattr(
        ray, "init", FakeInit(raises=ConnectionError("no cluster"))
    )
    monkeypatch.setattr(ray_runtime, "available_memory_gb", lambda: 2.0)

    with pytest.raises(ConnectionError):
        init_ray()


# ------------------------------- memory probe -------------------------------
def test_available_memory_is_a_positive_number_or_none():
    free = available_memory_gb()
    assert free is None or free > 0.0


def test_available_memory_returns_none_without_psutil(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("simulated: psutil not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    assert available_memory_gb() is None
