
"""MAPPO RLModule with a shared decentralized actor and centralized critic.

Actor:
    Receives only each agent's local observation (26 features).
    Produces Gaussian action-distribution inputs for 3 continuous actions.

Critic:
    Receives only the centralized global state.
    Its input dimension is obtained from the environment observation space.

All agents mapped to the shared policy use the same actor and critic
parameters.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

import gymnasium
import torch
from torch import nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchDiagGaussian,
)
from ray.rllib.core.rl_module.apis.value_function_api import (
    ValueFunctionAPI,
)
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule


class MAPPOActorCriticModule(TorchRLModule, ValueFunctionAPI):
    """Shared actor and centralized critic for cooperative multi-agent PPO."""

    EXPECTED_LOCAL_DIM = 26
    EXPECTED_GLOBAL_DIM = 898  # Reference dimension for 50 agents.
    EXPECTED_ACTION_DIM = 3

    MIN_LOG_STD = -20.0
    MAX_LOG_STD = 2.0

    def setup(self) -> None:
        """Construct independent actor and centralized critic networks."""

        model_config = self.model_config or {}

        # 1. Validate observation and action spaces.
        if not isinstance(self.observation_space, gymnasium.spaces.Dict):
            raise TypeError(
                "MAPPO requires a Gymnasium Dict observation space containing "
                "'local_obs' and 'global_state'."
            )

        required_keys = {"local_obs", "global_state"}
        actual_keys = set(self.observation_space.spaces.keys())

        if actual_keys != required_keys:
            raise ValueError(
                f"Expected observation keys {required_keys}, "
                f"received {actual_keys}."
            )

        local_dim = self.observation_space["local_obs"].shape[0]
        global_dim = self.observation_space["global_state"].shape[0]

        if not hasattr(self.action_space, "shape"):
            raise TypeError("MAPPO requires a continuous action space.")

        action_dim = self.action_space.shape[0]

        if local_dim != self.EXPECTED_LOCAL_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_LOCAL_DIM} local observation features, "
                f"received {local_dim}."
            )

        if not isinstance(global_dim, int) or global_dim <= 0:
            raise ValueError(
                f"Global state dimension must be positive, received {global_dim}."
            )

        if action_dim != self.EXPECTED_ACTION_DIM:
            raise ValueError(
                f"Expected {self.EXPECTED_ACTION_DIM} continuous action "
                f"dimensions, received {action_dim}."
            )

        # Keep the actual environment-provided dimension.
        self.global_dim = global_dim

        # 2. Validate network configuration.
        hidden_dims = model_config.get("fcnet_hiddens", [256, 256])

        if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims:
            raise ValueError(
                "fcnet_hiddens must be a non-empty list or tuple."
            )

        if any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in hidden_dims
        ):
            raise ValueError(
                "All hidden layer sizes must be positive integers."
            )

        activation_name = model_config.get("fcnet_activation", "tanh")

        activation_map = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
        }

        if activation_name not in activation_map:
            raise ValueError(
                f"Unsupported activation '{activation_name}'. "
                f"Choose from {list(activation_map.keys())}."
            )

        activation_class = activation_map[activation_name]

        # 3. MLP builder.
        def build_mlp(
            input_dim: int,
            output_dim: int,
        ) -> nn.Sequential:
            layers = []
            previous_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(previous_dim, hidden_dim))
                layers.append(activation_class())
                previous_dim = hidden_dim

            layers.append(nn.Linear(previous_dim, output_dim))

            return nn.Sequential(*layers)

        # 4. Decentralized actor: local observation only.
        self.actor = build_mlp(
            input_dim=local_dim,
            output_dim=2 * action_dim,
        )

        # 5. Centralized critic: global state only.
        self.critic = build_mlp(
            input_dim=self.global_dim,
            output_dim=1,
        )

        self.action_dim = action_dim

    def get_inference_action_dist_cls(self) -> Type:
        """Return RLlib's diagonal Gaussian distribution."""
        return TorchDiagGaussian

    def _get_observations(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Extract and validate the structured observation batch."""

        if Columns.OBS not in batch:
            raise KeyError(
                f"Batch is missing the required '{Columns.OBS}' column."
            )

        observations = batch[Columns.OBS]

        if not isinstance(observations, dict):
            raise TypeError(
                "MAPPO observations must be a dictionary containing "
                "'local_obs' and 'global_state'."
            )

        if "local_obs" not in observations:
            raise KeyError("Observation is missing 'local_obs'.")

        if "global_state" not in observations:
            raise KeyError("Observation is missing 'global_state'.")

        return observations

    def _compute_action_dist_inputs(
        self,
        local_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Gaussian distribution inputs using local observations."""

        if local_obs.shape[-1] != self.EXPECTED_LOCAL_DIM:
            raise ValueError(
                f"Expected local observation dimension "
                f"{self.EXPECTED_LOCAL_DIM}, received {local_obs.shape[-1]}."
            )

        if not torch.isfinite(local_obs).all():
            raise ValueError("Local observations contain NaN or infinity.")

        raw_outputs = self.actor(local_obs)

        means, log_stds = torch.split(
            raw_outputs,
            self.action_dim,
            dim=-1,
        )

        log_stds = torch.clamp(
            log_stds,
            min=self.MIN_LOG_STD,
            max=self.MAX_LOG_STD,
        )

        return torch.cat([means, log_stds], dim=-1)

    def _forward(
        self,
        batch: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate action-distribution inputs from local observations."""

        observations = self._get_observations(batch)

        action_dist_inputs = self._compute_action_dist_inputs(
            observations["local_obs"]
        )

        return {
            Columns.ACTION_DIST_INPUTS: action_dist_inputs,
        }

    def _forward_train(
        self,
        batch: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Training forward pass; actor remains decentralized."""
        return self._forward(batch, **kwargs)

    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> torch.Tensor:
        """Predict state values using only the centralized global state.

        Implements RLlib's ValueFunctionAPI so that the GAE connector
        recognizes and invokes the centralized critic.
        """

        observations = self._get_observations(batch)
        global_state = observations["global_state"]

        if global_state.shape[-1] != self.global_dim:
            raise ValueError(
                f"Expected global state dimension "
                f"{self.global_dim}, received {global_state.shape[-1]}."
            )

        if not torch.isfinite(global_state).all():
            raise ValueError(
                "Global state contains NaN or infinity."
            )

        values = self.critic(global_state)

        return values.squeeze(-1)
