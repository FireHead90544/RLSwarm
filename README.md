# Policy Optimization for Scalable Swarm Robotics

Official implementation of the research paper **"Policy Optimization for Scalable Swarm Robotics in Physics-Based Environments"**.

This repository contains a custom continuous physics-based simulation environment for multi-agent swarm foraging and the implementation of Proximal Policy Optimization (PPO) to solve it.

![Demo (Downsampled)](https://github.com/user-attachments/assets/6e0987a6-cf92-433d-9357-197b4aa9e77c)

## 🚀 Features

- **Custom Swarm Environment**: A Gymnasium-compatible environment with continuous physics, inertia, and ray-cast sensing.
- **PPO Implementation**: Stable-Baselines3 based PPO training with curriculum learning.
- **Emergent Behaviors**: Agents learn collision avoidance, homing, and cooperative foraging without explicit communication.
- **Benchmarks**: Comparisons with DQN, Random Walk, and Human Operators.

## 📂 Project Structure

```
.
├── env/                # Custom Swarm Environment
│   ├── swarm_env.py    # Core Gymnasium environment
│   └── wrappers.py     # SwarmVecEnv wrapper for SB3
├── models/             # Pre-trained model checkpoints (.zip)
├── experiments/        # Evaluation & benchmarking scripts
│   ├── core.py         # Shared ExperimentRunner base class
│   ├── runners/        # Individual experiment runners
│   │   ├── scalability.py   # Scalability benchmark (A)
│   │   ├── efficiency.py    # AI vs Random benchmark (C)
│   │   └── human_vs_ai.py  # Human vs AI benchmark (D)
│   ├── analysis/       # Plotting scripts and generated figures
│   └── results/        # CSV results and final report
├── docs/               # Documentation and demo assets
│   └── TRAINING_GUIDE.md
├── train.py            # Training script (PPO, checkpoint support)
├── play.py             # Inference/Visualization script
├── manual_control.py   # Human control script
├── record_video.py     # Record a gameplay video to MP4
├── check_env.py        # Validate environment compatibility with SB3
├── test_train.py       # Smoke test: short training run
├── config.py           # All hyperparameters and constants
├── requirements.txt    # Dependencies
└── archived/           # Legacy code (DQN Baseline)
```

## 🏛️ Archived Baselines

The `archived/dqn/` directory contains the original Deep Q-Network implementation used as a baseline in the research paper. While it failed to solve the task effectively (as documented in the paper), the code is preserved for reproducibility and comparative analysis.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/FireHead90544/RLSwarm
    cd RLSwarm
    ```

2.  **Install dependencies (using uv)**:
    ```bash
    uv init
    uv pip install -r requirements.txt
    ```

## 🎮 Usage

### 1. Run Pre-trained Model
Visualize the best performing PPO policy (remaining checkpoints are in the `models/` directory):
```bash
uv run play.py --model models/best_model.zip --fps 60
```
*Controls: `TAB` to switch agent view, `D` to toggle debug mode, `R` to reset the environment, `ESC` to quit.*

### 2. Train from Scratch
Start a new training session:
```bash
# Default 1M timesteps
uv run train.py

# Custom timesteps
uv run train.py --steps 2000000
```

### 3. Continued Training from Checkpoint
If you have a checkpoint and want to continue training:
```bash
# Continue from specific checkpoint
uv run train.py \
  --checkpoint models/ppo_swarm_5000000_steps.zip \
  --steps 1000000

# Timestep counter continues from checkpoint
# Logs append to TensorBoard automatically (on Colab)
```
*Check `config.py` to adjust hyperparameters.*

### 4. Manual Control
Test the environment yourself with keyboard input:
```bash
uv run manual_control.py
```
*Controls: Arrow keys to move/rotate, `TAB` to switch agents, `D` to toggle debug mode, `R` to reset the environment, `ESC` to quit.*

### 5. Record a Video
Record a gameplay video of the trained model to an MP4 file (requires `opencv-python`):
```bash
uv run record_video.py
```
*The script uses the model at `models/ppo_swarm_9771120_steps_reshaped` by default and saves `demo_debug.mp4` in the project root. Edit `record_video.py` to change the model path, output filename, or number of frames.*

### 6. Validate the Environment
Verify that the custom environment is compatible with Stable-Baselines3's API:
```bash
uv run check_env.py
```
*Runs two checks: the raw `SwarmEnv` API and the `SwarmVecEnv` wrapper used during training.*

### 7. Smoke-Test Training
Run a short end-to-end training pass (500 steps) to confirm everything is wired correctly:
```bash
uv run test_train.py
```
*The test model is saved and then immediately deleted. Useful after changes to the environment or config.*

## 📄 Documentation

- [**Research Paper**](#): Full academic paper detailing the methodology and results. (Releasing when published)
- [**Training Guide**](docs/TRAINING_GUIDE.md): Detailed guide on the training curriculum, reward tuning, and hyperparameters.
- [**Experiment Results**](experiments/results/FINAL_REPORT.md): Final research report covering scalability, efficiency, and human vs. AI benchmarks.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

MIT License.
