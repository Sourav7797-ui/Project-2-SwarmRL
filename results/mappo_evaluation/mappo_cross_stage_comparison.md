# SwarmRL — MAPPO Cross-Stage Evaluation Report



**Evaluation date:** 6 November 2026  

**Report status:** Pilot evaluation completed  

**Training performed during evaluation:** No



## 1. Objective



Evaluate two independently trained MAPPO pilot checkpoints against the baseline policy across three environment stages:



- `static_world` — training environment

- `dynamic_obstacles` — cross-stage evaluation

- `wind` — cross-stage evaluation



The purpose is to assess preliminary policy improvement and cross-stage transfer. These results are exploratory and are not a statistical validation of generalization.



## 2. Training checkpoints



Both checkpoints were trained on `static_world` using the same pilot configuration.



| Run | Run ID | Checkpoint |

|---|---|---|

| Run 1 | `20261106_143814_875240` | `checkpoints/mappo_pilot/run_20261106_143814_875240/iteration_0010` |

| Run 2 | `20261106_143915_287009` | `checkpoints/mappo_pilot/run_20261106_143915_287009/iteration_0010` |



Pilot configuration:



- Agents: 4

- Training iterations: 10

- Environment steps per iteration: 128

- Total environment steps per run: 1,280

- Train batch size: 128

- Minibatch size: 128

- Epochs: 2

- Learning rate: 0.0003

- Hidden layers: 64, 64

- Seed: 42

- Training stage: `static_world`



## 3. Evaluation methodology



Each checkpoint was evaluated against the baseline policy in all three stages.



| Parameter | Value |

|---|---:|

| Episodes per model per evaluation | 10 |

| Agents | 4 |

| First episode seed | 42 |

| Episode seeds | 42–51 |

| Environment steps per episode | 128 |

| Agent steps per episode | 512 |

| Training during evaluation | No |



Evaluation directories:



| Directory | Checkpoint | Stage |

|---|---|---|

| `evaluation_20261106_150639` | Run 1 | static_world |

| `evaluation_20261106_153741` | Run 2 | static_world |

| `evaluation_20261106_160121` | Run 1 | dynamic_obstacles |

| `evaluation_20261106_163152` | Run 2 | dynamic_obstacles |

| `evaluation_20261106_163418` | Run 1 | wind |

| `evaluation_20261106_163620` | Run 2 | wind |



## 4. Results



### 4.1 Episode return



| Stage | Checkpoint | Baseline mean | MAPPO mean | Improvement | Improvement % |

|---|---|---:|---:|---:|---:|

| Static | Run 1 | 353.2781 | 417.0827 | +63.8046 | +18.1% |

| Static | Run 2 | 353.2781 | 433.3080 | +80.0299 | +22.65% |

| Dynamic | Run 1 | 346.0369 | 420.5421 | +74.5051 | +21.5% |

| Dynamic | Run 2 | 346.0369 | 441.5333 | +95.4964 | +27.6% |

| Wind | Run 1 | 312.3513 | 399.4553 | +87.1041 | +27.9% |

| Wind | Run 2 | 312.3513 | 426.0723 | +113.7210 | +36.4% |



### 4.2 Coverage



| Stage | Baseline coverage | Run 1 MAPPO | Run 2 MAPPO |

|---|---:|---:|---:|

| Static | 4.3620% | 6.3404% | 6.4004% |

| Dynamic | 4.3436% | 6.3744% | 6.5184% |

| Wind | 4.2388% | 6.2040% | 6.5124% |



Coverage improvement over baseline:



| Stage | Run 1 improvement | Run 2 improvement |

|---|---:|---:|

| Static | +1.9784 percentage points | +2.0384 percentage points |

| Dynamic | +2.0308 percentage points | +2.1748 percentage points |

| Wind | +1.9652 percentage points | +2.2736 percentage points |



### 4.3 Return variability



MAPPO episode-return standard deviation:



| Stage | Run 1 | Run 2 |

|---|---:|---:|

| Static | 51.5337 | 39.3352 |

| Dynamic | 55.8577 | 38.6186 |

| Wind | 67.7267 | 32.9954 |



Run 2 had lower return variability in all three evaluation stages.



### 4.4 Collision events



Mean collision events per episode:



| Stage | Baseline | Run 1 MAPPO | Run 2 MAPPO |

|---|---:|---:|---:|

| Static | 0.0 | 0.0 | 0.0 |

| Dynamic | 0.0 | 0.0 | 0.0 |

| Wind | 0.1 | 0.1 | 0.0 |



The wind evaluation recorded one mean collision event per ten episodes for Run 1 and zero for Run 2. Static and dynamic evaluations recorded zero for both checkpoints.



## 5. Interpretation



### 5.1 Policy improvement



Both MAPPO checkpoints achieved higher mean episode returns and coverage than the baseline in all three tested stages.



This is an encouraging initial result. The improvements are present not only in the static training environment but also in the dynamic-obstacle and wind evaluation environments.



### 5.2 Run 1 versus Run 2



Run 2 achieved higher mean return and coverage than Run 1 in every evaluated stage.



| Stage | Run 1 return | Run 2 return | Run 2 advantage |

|---|---:|---:|---:|

| Static | 417.0827 | 433.3080 | +16.2253 |

| Dynamic | 420.5421 | 441.5333 | +20.9912 |

| Wind | 399.4553 | 426.0723 | +26.6170 |



Run 2 also recorded lower return variability across all three stages and zero mean collision events in the wind evaluation.



**Provisional selection:** Run 2 is the stronger pilot checkpoint based on the current evaluation metrics. Run 1 should also be retained for reproducibility and future comparisons.



### 5.3 Cross-stage transfer



Both checkpoints improved over baseline in dynamic-obstacle and wind environments despite being trained on `static_world`.



This is preliminary evidence of cross-stage transfer. It does not establish robust generalization across different seeds, environment configurations, or larger agent populations.



### 5.4 Statistical limitations



Each evaluation used only 10 episodes. Consequently:



- The results are descriptive pilot measurements.

- No statistical significance claim is made.

- More independent seeds and episodes are required.

- Performance should be checked under additional environment configurations and agent counts.

- Collision and coverage metrics should remain part of future evaluation.



## 6. Overall conclusion



The MAPPO pilot implementation has completed two independent training runs and six baseline-versus-trained evaluations across three environment stages.



The measured results are consistently positive for mean return and coverage. Run 2 is the provisional best checkpoint.



The next research phase should focus on a controlled, larger-scale experiment with more training iterations, independent seeds, and broader evaluation coverage. That phase should begin only after the experiment configuration and acceptance criteria have been reviewed.



## 7. Reproducibility and artifact preservation



The original training checkpoints and six evaluation result directories should be retained unchanged.



The evaluation metadata records:



- Checkpoint identity

- Environment stage

- Episode count

- Agent count

- Initial seed

- `training_performed = false`



This report is a consolidated interpretation of the existing evaluation artifacts; it does not replace the original CSV summaries or JSON metadata.



**Final status:** MAPPO pilot training and cross-stage evaluation completed. Larger-scale validation remains future work.

