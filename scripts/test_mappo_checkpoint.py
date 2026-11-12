import tempfile
import torch

from pathlib import Path
from ray.rllib.core.columns import Columns

from backend.training.mappo_config import build_mappo_config


print("=== MAPPO CHECKPOINT ROUND-TRIP TEST ===")

config = build_mappo_config(num_agents=50)

multi_module = config.get_multi_rl_module_spec().build()
module = multi_module["shared_policy"]

batch = {
    Columns.OBS: {
        "local_obs": torch.randn(4, 26),
        "global_state": torch.randn(4, 898),
    }
}

module.eval()

with torch.no_grad():
    original_actor = module._forward_train(batch)[
        Columns.ACTION_DIST_INPUTS
    ].clone()

    original_critic = module.compute_values(batch).clone()

print("Original actor shape:", tuple(original_actor.shape))
print("Original critic shape:", tuple(original_critic.shape))

with tempfile.TemporaryDirectory(
    prefix="swarmrl_mappo_checkpoint_test_"
) as tmp:

    checkpoint_path = str(Path(tmp) / "multi_module")

    saved_path = multi_module.save_to_path(checkpoint_path)

    print("Checkpoint saved:", Path(saved_path).is_dir())
    print(
        "Metadata exists:",
        (Path(saved_path) / "metadata.json").exists(),
    )

    restored = type(multi_module).from_checkpoint(saved_path)
    restored_module = restored["shared_policy"]
    restored_module.eval()

    with torch.no_grad():
        restored_actor = restored_module._forward_train(batch)[
            Columns.ACTION_DIST_INPUTS
        ]

        restored_critic = restored_module.compute_values(batch)

    actor_match = torch.allclose(
        original_actor,
        restored_actor,
        atol=1e-6,
        rtol=1e-6,
    )

    critic_match = torch.allclose(
        original_critic,
        restored_critic,
        atol=1e-6,
        rtol=1e-6,
    )

    print("Restored module:", type(restored_module).__name__)
    print("Actor outputs match:", actor_match)
    print("Critic outputs match:", critic_match)

    print(
        "Actor max absolute difference:",
        (original_actor - restored_actor).abs().max().item(),
    )

    print(
        "Critic max absolute difference:",
        (original_critic - restored_critic).abs().max().item(),
    )

    print(
        "CHECKPOINT ROUND-TRIP:",
        "PASSED" if actor_match and critic_match else "FAILED",
    )

print("Temporary checkpoint automatically removed.")