# SwarmRL MAPPO — Day 2 Experiment Protocol



**Date:** 11 November 2026  

**Purpose:** Establish controlled, reproducible criteria for deciding whether further MAPPO training is justified.



## 1. Current position



Three MAPPO validation training runs have already been completed:



| Training seed | Run ID |

|---|---|

| 123 | `run_20261108_000525_151777` |

| 2026 | `run_20261108_002340_785217` |

| 31415 | `run_20261108_003040_192693` |



Each run used:

- 4 agents

- 128 maximum episode steps

- 10 training iterations

- Training batch size 128

- Minibatch size 128

- 2 epochs

- Learning rate 0.0003

- Hidden layers 64, 64

- Reward scale 0.05



All three checkpoints were evaluated against the existing PPO baseline in:

- `static_world`

- `dynamic_obstacles`

- `wind`



Each evaluation used ten paired episodes with seeds 42–51.



## 2. Existing evidence



Seed 2026 is the leading experimental candidate.



Its mean coverage improved over PPO in all three stages:

- Static: +0.4236

- Dynamic obstacles: +0.6272

- Wind: +0.7068



Its mean raw return:

- Static: -5.8275

- Dynamic obstacles: +15.5739

- Wind: +31.6403



However, all three paired 95% return confidence intervals included zero.



The other two seeds generally underperformed PPO.



**Interpretation:** MAPPO shows promising stage-specific behavior, but robust improvement across training seeds has not been demonstrated.



## 3. Experimental objective



Determine whether a further controlled MAPPO experiment can provide stronger evidence of repeatable improvement over the PPO baseline.



The experiment must distinguish:

1\. Genuine improvement in raw episode return.

2\. Improvement in coverage.

3\. Changes in collision frequency.

4\. Sensitivity to training seed.

5\. Possible degradation in any environment stage.



## 4. Proposed experimental design



### 4.1 Baseline



Use the existing, preserved PPO checkpoint and evaluation configuration.



Do not retrain or modify PPO.



### 4.2 MAPPO candidate



Use the existing seed-2026 MAPPO checkpoint as the current candidate.



A new training run should only be considered if its purpose is explicitly defined and its configuration is fixed before execution.



Do not overwrite existing checkpoints or result directories.



### 4.3 Environment stages



Evaluate all candidates in:

- `static_world`

- `dynamic_obstacles`

- `wind`



Keep environment parameters identical between baseline and candidate.



### 4.4 Paired evaluation



Use the same episode seeds for PPO and MAPPO.



The existing seeds 42–51 provide a reproducible starting point. If a larger evaluation sample is approved, add new seeds without changing the existing reports.



Record:

- Raw episode return

- Coverage percentage

- Collision events

- Collisions per 1,000 agent steps

- Episode length

- Environment steps



### 4.5 Statistical analysis



For each stage:

- Calculate paired mean return differences.

- Calculate paired mean coverage differences.

- Report 95% confidence intervals.

- Report episode-level wins, losses and ties.

- Report collision differences.

- Identify any multiple-comparison limitations.



Do not treat a positive mean difference alone as proof of improvement.



## 5. Proposed acceptance criteria



A candidate should be considered promising only when:



1\. Mean raw return is higher than PPO in the relevant stage.

2\. Paired return confidence intervals support improvement.

3\. Coverage does not materially deteriorate.

4\. Collision frequency does not materially increase.

5\. Improvement is not confined to one training seed.

6\. No environment stage experiences an unacceptable regression.



These criteria should be finalized before any additional training or evaluation.



## 6. Resource controls



The development machine has:

- 8 physical CPU cores

- 16 logical processors

- Approximately 16 GB RAM



Previous runs used:

- 2 CPUs for Ray

- 100 MB object-store allocation

- Zero environment runners

- Four agents



Before any future training:

- Check available memory.

- Retain the existing Ray memory guard.

- Avoid concurrent training and evaluation.

- Preserve existing checkpoints.

- Record configuration and random seeds.



No Ray initialization is authorized during protocol preparation.



## 7. Execution authorization



The following activities require separate explicit approval:



| Activity | Authorization |

|---|---|

| Read existing reports and CSV files | Permitted |

| Inspect existing configuration and scripts | Permitted |

| Prepare experiment configuration | Permitted |

| Modify training code | Requires approval |

| Initialize Ray | Requires approval |

| Restore MAPPO checkpoint | Requires approval |

| Start MAPPO training | Requires approval |

| Run new evaluation episodes | Requires approval |

| Modify or replace existing checkpoints | Not authorized |



## 8. Decision gate



Before starting a new experiment, review the existing results and determine whether further training is likely to answer a specific unresolved question.



If no clear experimental question or justified configuration change exists, retain the existing evidence and proceed to finalization instead of running another experiment.



## 9. Expected deliverables



- Approved experiment configuration.

- Reproducible training and evaluation records, if authorized.

- Paired comparison tables.

- Statistical confidence intervals.

- Collision and coverage analysis.

- Final recommendation on MAPPO readiness.



## 10. Current status



**Protocol preparation:** In progress.



**Additional training:** Not authorized.



**Additional evaluation:** Not authorized.



**Existing PPO baseline and MAPPO validation artifacts:** Preserve unchanged.



The goal is a technically defensible conclusion, not simply a higher reward from another training run.

