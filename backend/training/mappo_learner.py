"""MAPPO Torch learner using RLlib's PPO optimization logic.

The centralized critic is implemented by MAPPOActorCriticModule.
RLlib's PPOTorchLearner already calls module.compute_values() when
calculating value-function loss, so the standard PPO loss can be reused.
"""

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import (
    PPOTorchLearner,
)


class MAPPOTorchLearner(PPOTorchLearner):
    """PPO learner configured for the MAPPO actor-critic module.

    This class intentionally retains RLlib's standard PPO loss:
    - clipped surrogate policy objective
    - centralized value-function loss through compute_values()
    - entropy regularization
    - optional KL penalty

    No PPO learner behavior is overridden here.
    """

    pass