# SwarmRL MAPPO — Day 1 Technical Audit Report



**Audit date:** 11 November 2026  

**Project:** SwarmRL  

**Scope:** RLlib integration, MAPPO learner architecture, action handling, reward scaling, gradient flow, optimizer configuration, and metric extraction.



---



## 1. Executive Summary



The Day 1 MAPPO technical audit found the core implementation structurally consistent with the intended Multi-Agent Proximal Policy Optimization (MAPPO) architecture.



The implementation uses:



- A shared decentralized actor receiving each agent's local observation.

- A centralized critic receiving the global state.

- RLlib's standard PPO learner and optimization objective.

- Generalized Advantage Estimation (GAE).

- A shared policy across agents.

- Normalized continuous actions mapped to the environment's physical action bounds.



The audit verified the learner connector's output structure, finite GAE and value targets, actor and critic gradient flow, PPO value-loss integration, optimizer parameter collection, gradient-clipping configuration, action handling, and reward scaling.



**No architecture or optimizer correction was justified by the evidence.**



The principal remaining technical limitation is that the actual `MAPPOTorchLearner.update()` and optimizer step have not been executed as part of this audit. The audit establishes source-level behavior and module-level gradient flow, not a completed end-to-end learner update.



## 2. Test Results



Latest complete test suite:



| Result | Count |

|---|---:|

| Passed | 342 |

| Skipped | 2 |

| Failed | 0 |



**Skipped tests:**



1\. `tests/test_day9_rllib_smoke.py::test_one_real_training_iteration_runs_and_reports_finite_losses`

2\. `tests/test_mappo_runtime_smoke.py::test_one_real_mappo_training_iteration_runs_and_reports_finite_losses`



These are opt-in real-training smoke tests. Their omission means the full suite did not perform a real PPO or MAPPO training iteration.



Additional MAPPO test results:



| Test | Result |

|---|---:|

| Complete MAPPO integration suite | 39 passed |

| Actual learner connector batch and actor/critic gradient regression test | 1 passed |



The permanent regression test verifies connector-produced batch structure and finite, nonzero actor and critic gradients using a synthetic module-level loss.



## 3. Architecture Audit



### 3.1 Decentralized actor



The actor receives only `local_obs`.



- Local observation dimension: 26.

- Continuous action dimension: 3.

- Actor output: Gaussian distribution parameters.

- Log standard deviation is clamped to the configured range of -20 to 2.

- Training and inference use the Torch diagonal Gaussian distribution.



The actor does not directly consume the global state.



**Finding:** The actor's information boundary is consistent with decentralized execution.



### 3.2 Centralized critic



The critic receives `global_state`.



The global-state dimension is calculated as:



`7 × num_agents + 548`



For four agents:



`7 × 4 + 548 = 576`



For fifty agents:



`7 × 50 + 548 = 898`



The critic predicts a scalar value for the supplied global state.



**Finding:** The centralized critic's input dimension is dynamically determined and consistent with the intended global-state design.



### 3.3 GAE connector



The installed RLlib `GeneralAdvantageEstimation` connector calls:



`module.compute_values(batch\[mid])`



for modules implementing `ValueFunctionAPI`.



The actual learner connector diagnostic produced:



| Field | Observed result |

|---|---|

| Shared policy | Present |

| Local observations | `(12, 26)` |

| Global states | `(12, 576)` |

| Advantages | Present and finite |

| Value targets | Present and finite |



The twelve rows correspond to three observations for each of four agents.



**Finding:** The connector produces the structured observations and GAE outputs required by the MAPPO module.



## 4. PPO Learner Loss Audit



The installed RLlib `PPOTorchLearner.compute_loss_for_module()` implementation was inspected.



It calculates:



- PPO clipped surrogate policy loss.

- Centralized critic value-function loss.

- Entropy contribution.

- Optional KL penalty.



The value-function prediction is obtained through:



`module.compute_values(batch, embeddings=...)`



The value loss compares these predictions against `VALUE_TARGETS`.



The resulting value loss is included in the total objective through:



`config.vf_loss_coeff * vf_loss_clipped`



**Finding:** The centralized critic is connected to the actual PPO loss implementation at source level. Its value loss is not merely a diagnostic metric.



## 5. Actor and Critic Gradient Audit



A diagnostic used the actual learner-connector-produced batch.



It performed a module-level forward pass, calculated actor and critic outputs, constructed a synthetic combined loss, and called backward propagation.



Observed results:



| Measurement | Actor | Critic |

|---|---:|---:|

| Gradient tensors | 6 | 6 |

| Gradients finite | Yes | Yes |

| Gradient norm | 0.241505 | 0.371538 |



Both networks received finite, nonzero gradients.



**Important limitation:** This diagnostic did not execute RLlib's actual PPO loss or optimizer update. It verifies gradient connectivity for both subnetworks using the real connector-produced batch, not end-to-end PPO optimization.



## 6. Optimizer Registration Audit



The installed RLlib source was inspected.



`PPOTorchLearner.configure_optimizers_for_module()`:



1\. Retrieves the shared module.

2\. Calls `self.get_parameters(module)`.

3\. Constructs `torch.optim.Adam(params)`.

4\. Registers the optimizer and its parameters.



The concrete Torch implementation of `get_parameters()` returns:



`list(module.parameters())`



PyTorch recursively includes parameters of registered child modules.



Since the MAPPO module contains the registered actor and critic networks, the parameter collection includes both.



**Finding:** RLlib's default Adam optimizer construction is configured to include actor and critic parameters.



**Limitation:** No live optimizer was instantiated or stepped during this audit.



## 7. Gradient Clipping Audit



Effective MAPPO configuration:



| Setting | Value |

|---|---|

| `grad_clip` | 0.5 |

| `grad_clip_by` | `global_norm` |

| `log_gradients` | False |



RLlib's Torch learner source confirms that gradient post-processing applies the configured clipping function.



With global-norm clipping enabled, the configured threshold is 0.5.



The inspected source also shows that the global gradient norm metric can be logged through the global-norm clipping path even when `log_gradients=False`.



**Finding:** Global-norm gradient clipping is configured and applied by the inherited learner implementation. No correction is indicated.



## 8. Action Normalization and Clipping Audit



The MAPPO environment adapter exposes a normalized action space:



`\[-1, 1]` for each of the three action dimensions.



The environment adapter converts normalized actions into physical bounds using an affine mapping:



`low + ((clipped_action + 1) / 2) × (high - low)`



The normalized action is clipped before conversion.



The MAPPO configuration explicitly uses:



- `normalize_actions=False`

- `clip_actions=True`



The RLlib module-to-environment connector creates a separate `ACTIONS_FOR_ENV` copy for clipping.



The inspected single-agent and multi-agent environment runners:



- Send the clipped copy to the environment.

- Store the original sampled action in the episode.

- Preserve the associated action log-probability.



**Finding:** The inspected training pipeline does not show a mismatch between the stored Gaussian action and its action log-probability caused by environment-side clipping.



**Remaining diagnostic question:** The frequency of Gaussian samples outside the normalized action bounds has not been measured. High saturation could reduce effective control resolution, but no saturation problem has been established.



The deterministic evaluation path also clips actions to `\[-1, 1]` before passing them to the environment adapter.



## 9. Reward Scaling and Raw Return Audit



The `SwarmRLlibEnv` adapter:



1\. Resets the raw-return accumulator at the beginning of an episode.

2\. Accumulates unscaled environment rewards.

3\. Multiplies rewards by `reward_scale` before returning them to RLlib.

4\. Stores the unscaled episode return when the episode terminates or truncates.



The validation configuration used:



`reward_scale = 0.05`



**Finding:** Reward scaling is applied once for training, while raw episode returns remain available for evaluation and reporting.



No reward-scaling correction is indicated.



## 10. RLlib Metric Extraction Audit



The metric extraction implementation and installed RLlib source were reviewed.



The inspected source supports the expected nested result structure under the environment-runner and learner namespaces.



Expected metric paths include:



| Metric | Expected path |

|---|---|

| Mean episode return | `env_runners/episode_return_mean` |

| Sampled environment steps | `env_runners/num_env_steps_sampled` |

| Mean episode length | `env_runners/episode_len_mean` |

| Policy loss | `learners/shared_policy/policy_loss` |

| Value loss | `learners/shared_policy/vf_loss` |

| Unclipped value loss | `learners/shared_policy/vf_loss_unclipped` |

| Entropy | `learners/shared_policy/entropy` |

| Explained variance | `learners/shared_policy/vf_explained_var` |

| KL divergence | `learners/shared_policy/mean_kl_loss` |

| Current KL coefficient | `learners/shared_policy/curr_kl_coeff` |



The metric-flattening helper and extraction logic have synthetic regression coverage.



**Finding:** The extraction paths are structurally supported by the inspected RLlib source.



**Limitation:** Exact live training-result dictionaries have not been captured in this audit because no training iteration was executed. These paths should be confirmed against a future explicitly authorized run.



## 11. Existing Validation Experiment Context



Three previously completed validation runs used training seeds:



- 123

- 2026

- 31415



Each used ten training iterations with four agents and was evaluated across:



- `static_world`

- `dynamic_obstacles`

- `wind`



The paired evaluation evidence indicates substantial training-seed sensitivity.



Seed 2026 was the strongest experimental candidate:



- Mean raw return improved in dynamic-obstacle and wind evaluations.

- Mean coverage improved across all three stages.

- Paired coverage confidence intervals excluded zero for dynamic-obstacle and wind evaluations.

- Collision counts remained unchanged across the ten paired episodes in each stage.



However, the return confidence intervals included zero for all three stages. The other two training seeds generally underperformed the PPO baseline.



These results are exploratory because each evaluation used ten paired episodes and the nine comparisons were not adjusted for multiple testing.



**Recommendation:** Preserve all three checkpoints and evaluation reports. Treat seed 2026 as a promising experimental candidate, not a production-ready or robustly validated MAPPO model.



No additional training or evaluation is proposed as part of this report.



## 12. Outstanding Verification Items



The following items remain explicitly unverified:



1\. An actual `MAPPOTorchLearner.update()` using the configured MAPPO learner.

2\. A real optimizer step updating actor and critic parameters.

3\. Live RLlib result-key availability and exact metric values.

4\. Actual action saturation frequency during training.

5\. Multi-seed reproducibility beyond the three completed validation runs.

6\. Statistical confirmation using a larger evaluation sample.



These items should not be inferred from source inspection or synthetic diagnostics.



## 13. Preservation and Safety



The audit did not require:



- Changing the existing PPO trainer.

- Modifying PPO checkpoints or baselines.

- Replacing trained MAPPO checkpoints.

- Running new training experiments.

- Running new evaluation episodes.

- Performing an optimizer update.



The existing PPO baseline and completed MAPPO validation artifacts should remain preserved.



## 14. Day 1 Conclusion



**Day 1 technical audit: substantially complete.**



The evidence supports the following conclusions:



- MAPPO actor and critic architecture is structurally correct.

- The actual learner connector produces the required batch structure.

- GAE and value targets are finite in the connector diagnostic.

- Actor and critic both receive finite, nonzero gradients.

- RLlib's PPO loss includes centralized critic value loss.

- The default Torch optimizer parameter collection includes both registered networks.

- Global-norm clipping is configured at 0.5.

- Action normalization, environment clipping, and stored action log-probabilities follow a consistent pipeline.

- Reward scaling and raw-return reporting are separated correctly.

- The regression suite passes.



No speculative code changes are recommended.



**Next phase:** Day 2 should begin only after explicit authorization for controlled MAPPO experiments. Until then, retain the current checkpoints, test results, and PPO baseline unchanged.



---



*Prepared from source inspection, regression-test outputs, connector diagnostics, and previously completed validation artifacts. Findings distinguish source-level verification from behavior requiring an actual learner update or new experiment.*

