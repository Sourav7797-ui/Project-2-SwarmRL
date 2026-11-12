# SwarmRL â€” Multi-Agent Drone Swarm Reinforcement Learning

A multi-agent drone swarm simulation and reinforcement learning project using **Ray RLlib**, **Proximal Policy Optimization (PPO)**, and **Multi-Agent PPO (MAPPO)**.

The project supports drone swarm navigation, coverage optimization, obstacle avoidance, collision monitoring, and evaluation across multiple environment stages.

---

## 1. Technology Stack

- Python 3.13
- Ray RLlib 2.59.0
- Gymnasium
- PettingZoo
- NumPy
- Pandas
- PyTorch
- Matplotlib
- Streamlit
- psutil
- pytest

## 2. Project Structure

```text
SwarmRL/
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ env/
â”‚   â””â”€â”€ training/
â”œâ”€â”€ backups/
â”œâ”€â”€ checkpoints/
â”œâ”€â”€ frontend/
â”œâ”€â”€ results/
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ train_swarm.py
â”‚   â”œâ”€â”€ evaluate_checkpoint.py
â”‚   â”œâ”€â”€ compare_baselines.py
â”‚   â”œâ”€â”€ train_mappo.py
â”‚   â”œâ”€â”€ evaluate_mappo.py
â”‚   â”œâ”€â”€ test_mappo_checkpoint.py
â”‚   â”œâ”€â”€ rllib_smoke.py
â”‚   â””â”€â”€ other utilities
â”œâ”€â”€ tests/
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ requirements-dev.txt
â”œâ”€â”€ conftest.py
â”œâ”€â”€ .gitignore
â””â”€â”€ README.md
```

## 3. Environment Setup

### Prerequisites

- Python 3.13.1
- Windows 10/11
- NVIDIA GPU recommended for accelerated deep learning workloads

### Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

### Verify installation

```powershell
python --version
python -m pip show ray gymnasium pettingzoo numpy pandas torch streamlit pytest
```

**GPU note:** PyTorch GPU acceleration depends on the installed PyTorch build and compatible NVIDIA CUDA drivers. For a new machine, select the appropriate CUDA build using the official PyTorch installation instructions.

## 4. Running the Dashboard

From the project root:

```powershell
python -m streamlit run .\frontend\dashboard.py
```

Open:

`http://localhost:8501`

## 5. Running the 3D Swarm Viewer

The 3D viewer uses a React, Vite, and Three.js frontend connected to the FastAPI simulation backend.

### Start the backend

From `frontend\3d`, run:

```powershell
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

### Start the frontend

Open a second terminal:

```powershell
Set-Location .\frontend\3d
npm install  # only if dependencies are not installed
npm run dev
```

Open: http://127.0.0.1:5173/

### Backend endpoints

- Health check: http://127.0.0.1:8000/api/health
- Simulation state: http://127.0.0.1:8000/api/state

The viewer provides live 3D drone, obstacle, trail, and coverage visualization, with Start, Pause, and Reset controls.

### Production build

From `frontend\3d`, run:

```powershell
npm run build
```

The production output is generated in `frontend\3d\dist`.

**Policy distinction:** The current 3D viewer uses a deterministic demo controller. It does not load or execute a trained PPO or MAPPO checkpoint. RL training and evaluation remain separate.

## 6. PPO Training and Evaluation

PPO is the original training pipeline and remains supported independently of MAPPO.

### Training

Display available options:

```powershell
python -m scripts.train_swarm --help
```

### Checkpoint evaluation

```powershell
python -m scripts.evaluate_checkpoint --help
```

### Baseline comparison

```powershell
python -m scripts.compare_baselines --help
```

### PPO benchmark findings

Day13 baseline is the recommended primary checkpoint based on the completed benchmark analysis.

Day16 separation-reward training is retained as an experimental alternative. Results show trade-offs between coverage and safety; they do not establish universal superiority.

Rewards from Day13 and Day16 should not be directly compared because Day16 uses additional separation reward shaping.

**PPO preservation:** MAPPO uses dedicated training and evaluation scripts. The original PPO trainer and existing PPO checkpoints are retained.

## 7. MAPPO Architecture

MAPPO is implemented as an additional multi-agent training approach.

### Architecture: Shared Decentralized Actor + Centralized Critic

- A shared actor policy is used by all agents.
- Each actor receives its own local observation.
- A centralized critic receives the global state.
- The policy is shared through `shared_policy`.
- The actor and critic are implemented in PyTorch using RLlib's TorchRLModule interface.
- The MAPPO learner reuses RLlib's PPO learner infrastructure.

### Observation dimensions

The local observation has 26 features.

The global state dimension is dynamic:

`global_state_dim = 7 Ã— num_agents + 548`

Examples:

| Number of agents | Global-state dimension |
|---:|---:|
| 4 | 576 |
| 8 | 604 |
| 16 | 660 |
| 50 | 898 |

The fixed global-state representation supports evaluation across different environment stages.

### Important implementation files

| File | Responsibility |
|---|---|
| `backend/training/mappo_module.py` | Shared actor and centralized critic |
| `backend/training/mappo_learner.py` | MAPPO learner integration |
| `backend/training/mappo_config.py` | MAPPO configuration and connectors |
| `backend/training/rllib_env.py` | RLlib multi-agent environment adapter |
| `backend/training/ray_runtime.py` | Ray runtime and resource configuration |
| `backend/env/swarm_env.py` | Swarm environment and global-state generation |
| `scripts/train_mappo.py` | Dedicated MAPPO training |
| `scripts/evaluate_mappo.py` | Baseline-versus-MAPPO evaluation |

## 8. MAPPO Training

Display all supported options:

```powershell
python -m scripts.train_mappo --help
```

### Reproducing the completed pilot configuration

The following is the configuration used for the two completed pilot runs:

```powershell
python -m scripts.train_mappo --agents 4 --steps 128 --iterations 10 --train-batch-size 128 --minibatch-size 128 --epochs 2 --learning-rate 0.0003 --hidden 64 64 --seed 42 --workers 0 --cpus 2 --object-store-mb 100 --stage static_world --checkpoint-every 5
```

**Warning:** Executing this command starts a new training run. It is provided for reproducibility and is not required to inspect existing results.

### Important training parameters

| Parameter | Pilot value |
|---|---:|
| Agents | 4 |
| Training stage | static_world |
| Iterations | 10 |
| Environment steps per iteration | 128 |
| Train batch size | 128 |
| Minibatch size | 128 |
| Epochs | 2 |
| Learning rate | 0.0003 |
| Hidden layers | 64, 64 |
| Random seed | 42 |
| Remote workers | 0 |
| CPUs allocated to Ray | 2 |
| Object store | 100 MB |
| Checkpoint frequency | Every 5 iterations |

## 9. MAPPO Checkpoint Evaluation

Display available evaluation options:

```powershell
python -m scripts.evaluate_mappo --help
```

The evaluator compares an untrained baseline policy and a trained MAPPO checkpoint.

### Evaluation example

```powershell
python -m scripts.evaluate_mappo ".\checkpoints\mappo_pilot\run_20261106_143915_287009\iteration_0010" --episodes 10 --seed 42 --cpus 2 --object-store-mb 100 --stage static_world
```

### Cross-stage evaluation examples

Use the same checkpoint with different stage overrides:

```powershell
python -m scripts.evaluate_mappo ".\checkpoints\mappo_pilot\run_20261106_143915_287009\iteration_0010" --episodes 10 --seed 42 --cpus 2 --object-store-mb 100 --stage dynamic_obstacles
```

```powershell
python -m scripts.evaluate_mappo ".\checkpoints\mappo_pilot\run_20261106_143915_287009\iteration_0010" --episodes 10 --seed 42 --cpus 2 --object-store-mb 100 --stage wind
```

**Warning:** Evaluation initializes Ray and restores a checkpoint. These commands are for future reproduction, not necessary to review the completed evaluations.

## 10. MAPPO Pilot Evaluation Results

Two independent MAPPO pilot runs were completed using the same training configuration.

Both checkpoints were evaluated against the baseline in three environment stages:

- `static_world`
- `dynamic_obstacles`
- `wind`

### Evaluation methodology

| Parameter | Value |
|---|---:|
| Episodes per model per evaluation | 10 |
| Agents | 4 |
| First seed | 42 |
| Episode seeds | 42â€“51 |
| Environment steps per episode | 128 |
| Agent steps per episode | 512 |
| Training during evaluation | No |

### Mean episode return

| Stage | Baseline | MAPPO Run 1 | MAPPO Run 2 |
|---|---:|---:|---:|
| Static | 353.2781 | 417.0827 | 433.3080 |
| Dynamic | 346.0369 | 420.5421 | 441.5333 |
| Wind | 312.3513 | 399.4553 | 426.0723 |

### Return improvement over baseline

| Stage | Run 1 | Run 2 |
|---|---:|---:|
| Static | +18.10% | +22.65% |
| Dynamic | +21.50% | +27.60% |
| Wind | +27.90% | +36.40% |

### Coverage

| Stage | Baseline | MAPPO Run 1 | MAPPO Run 2 |
|---|---:|---:|---:|
| Static | 4.3620% | 6.3404% | 6.4004% |
| Dynamic | 4.3436% | 6.3744% | 6.5184% |
| Wind | 4.2388% | 6.2040% | 6.5124% |

### Preliminary conclusion

Run 2 achieved higher mean return and coverage than Run 1 in all three evaluation stages.

Run 2 is therefore the **provisional best pilot checkpoint**. Both checkpoints should be retained for reproducibility and further comparison.

These results are encouraging preliminary evidence of cross-stage transfer. However, ten episodes per evaluation are insufficient to establish statistical significance or robust generalization.

### Evaluation artifacts

Consolidated results are stored in:

```text
results/
â””â”€â”€ mappo_evaluation/
    â”œâ”€â”€ evaluation_20261106_150639/
    â”œâ”€â”€ evaluation_20261106_153741/
    â”œâ”€â”€ evaluation_20261106_160121/
    â”œâ”€â”€ evaluation_20261106_163152/
    â”œâ”€â”€ evaluation_20261106_163418/
    â”œâ”€â”€ evaluation_20261106_163620/
    â”œâ”€â”€ mappo_cross_stage_comparison.md
    â””â”€â”€ mappo_cross_stage_comparison.csv
```

The Markdown report contains the methodology and interpretation. The CSV provides six machine-readable evaluation records for further analysis.

## 11. MAPPO Multi-Seed Validation

Three independent validation training runs were completed:

| Training seed | Run |
|---:|---|
| 123 | `20261108_000525_151777` |
| 2026 | `20261108_002340_785217` |
| 31415 | `20261108_003040_192693` |

Each run used 4 agents, 128 environment steps per episode, 10 training
iterations, batch and minibatch sizes of 128, 2 epochs, learning rate
0.0003, and a 64x64 hidden-layer configuration.

Each checkpoint was evaluated in all three stages using 10 paired episode
seeds (42-51), with deterministic actions and no training during evaluation.

### Mean episode return

| Stage | Baseline | Seed 123 | Seed 2026 | Seed 31415 |
|---|---:|---:|---:|---:|
| Static | 353.2781 | 260.6243 | 347.4506 | 309.4991 |
| Dynamic | 346.0369 | 270.1833 | 361.6109 | 314.3025 |
| Wind | 312.3513 | 264.1727 | 343.9916 | 299.1727 |

### Mean coverage

| Stage | Baseline | Seed 123 | Seed 2026 | Seed 31415 |
|---|---:|---:|---:|---:|
| Static | 4.3620% | 4.2784% | 4.7856% | 3.8444% |
| Dynamic | 4.3436% | 4.2588% | 4.9708% | 4.0068% |
| Wind | 4.2388% | 4.1680% | 4.9456% | 4.0436% |

### Safety observations and interpretation

- Seed 123 underperformed the baseline in mean return across all stages.
  It also recorded higher collision rates in static and dynamic environments.
- Seed 2026 improved mean coverage in all three stages and mean return in
  dynamic and wind environments, but slightly underperformed in static return.
  Its observed collision rates matched the baseline.
- Seed 31415 underperformed the baseline in mean return and coverage across
  all three stages. It recorded no collisions in these evaluations.

Paired confidence-interval analysis did not establish a consistent improvement
across all nine seed/stage comparisons. Three unadjusted paired 95% return
confidence intervals excluded zero, all in the negative direction: seed 123
static-world, seed 123 dynamic-obstacle, and seed 31415 static-world. The
other six return intervals included zero, including all three seed-2026
comparisons. No multiple-comparison correction was applied; these intervals
should be interpreted as exploratory rather than confirmatory.

**Validation conclusion:** The MAPPO implementation is technically integrated
and has undergone multi-seed evaluation, but robust performance improvement
has not been demonstrated. Seed 2026 is a promising experimental candidate,
not a production-ready replacement for PPO. The established PPO pipeline
remains the reference baseline.

Detailed technical findings and experiment protocols are documented in
`documentation/MAPPO_DAY1_AUDIT.md`,
`documentation/MAPPO_DAY2_EXPERIMENT_PROTOCOL.md`, and
`documentation/MAPPO_FINAL_VALIDATION_REPORT.md`.

Validation evaluation artifacts are stored under
`results/mappo_validation_evaluation/`.

## 12. Evaluation Metrics

The project tracks several performance and safety indicators:

- Mean episode reward
- Mean coverage percentage
- Drone-drone collision events
- Drone-obstacle collision events
- Minimum pair distance
- Steps below preferred separation
- Return variability

Return and coverage should be interpreted alongside safety metrics rather than in isolation.

## 13. Ray Resource Considerations

- Use one Ray process at a time.
- Use zero remote workers for conservative local execution.
- Avoid running training and evaluation simultaneously.
- Keep object-store allocation appropriate for available system memory.
- Close Ray resources after experiments.
- Do not launch multiple training runs concurrently on a memory-constrained machine.

## 14. Testing

Run the complete test suite from the project root:

```powershell
python -m pytest -q
```

Latest verified full regression result (11 November 2026):

```text
342 passed, 2 skipped
```

The skipped tests are reported separately by pytest.

## 15. Project Status

### Existing PPO work

- Day15: Exploration reward optimization â€” complete
- Day16: Multi-agent separation reward â€” complete
- Day17: Training and benchmark analysis â€” complete
- Day18: Streamlit dashboard â€” complete
- Day19: Integration and documentation â€” complete

### MAPPO work

- Shared decentralized actor and centralized critic â€” implemented
- MAPPO learner integration â€” implemented
- MAPPO configuration and RLlib environment integration â€” implemented
- Dedicated MAPPO training script â€” implemented
- Dedicated MAPPO evaluation script â€” implemented
- Two pilot training runs â€” completed
- Six pilot baseline-versus-MAPPO evaluations â€” completed
- Nine multi-seed validation stage evaluations â€” completed
- Cross-stage comparison report â€” completed
- Machine-readable CSV comparison â€” completed
- Full regression suite â€” 342 passed, 2 skipped

### Remaining research work

The validation experiments are complete. Further research remains:

- Train with additional independent random seeds.
- Increase training iterations and environment steps.
- Evaluate more episodes per checkpoint.
- Test additional agent counts.
- Assess safety and coverage under broader environment configurations.
- Compare MAPPO and PPO under a controlled, equivalent evaluation protocol.

No larger-scale training or evaluation is implied by this documentation.

---

**Project summary:** SwarmRL retains its established PPO pipeline and includes an integrated MAPPO implementation. The dashboard and regression suite are verified. MAPPO pilot results were encouraging, but subsequent multi-seed validation was mixed and did not establish robust improvement. PPO therefore remains the reference baseline, while MAPPO remains experimental.
