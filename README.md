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
├── models/             # Pre-trained models
├── docs/               # Research Paper and Documentation
├── train.py            # Training script
├── play.py             # Inference/Visualization script
├── manual_control.py   # Human control script
├── config.py           # Configuration parameters
├── requirements.txt    # Dependencies
└── archived/           # Legacy code (DQN Baseline)
```

## 🏛️ Archived Baselines

The `archived/dqn/` directory contains the original Deep Q-Network implementation used as a baseline in the research paper. While it failed to solve the task effectively (as documented in the paper), the code is preserved for reproducibility and comparative analysis.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/FireHead90544/RLSwarm
    cd NeuralSwarm
    ```

2.  **Install dependencies (using uv)**:
    ```bash
    uv init
    uv pip install -r requirements.txt
    ```

## 🎮 Usage

### 1. Run Pre-trained Model
Visualize the best performing PPO policy (remaining would be present in the github releases section):
```bash
uv run play.py --model models/best_model.zip --fps 60
```
*Controls: `TAB` to switch agent view, `D` to toggle debug mode.*

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

### 3. Manual Control
Test the environment yourself:
```bash
uv run manual_control.py
```
*Controls: Arrow keys to move, Tab to switch agents, D to toggle debug mode, R to reset the environment,ESC to quit*

## 📄 Documentation

- [**Research Paper**](#): Full academic paper detailing the methodology and results. (Releasing when published)
- [**Training Guide**](docs/TRAINING_GUIDE.md): Detailed guide on the training curriculum and hyperparameters.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## 📜 License

MIT License.
