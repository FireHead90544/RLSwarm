# Neural Swarm PPO - Complete Training Guide

## 🎯 Project Overview

Multi-agent foraging simulation using **Proximal Policy Optimization (PPO)** with **Stable Baselines3**. Agents learn to cooperatively forage for food, transport to nest, and avoid collisions through shared policy learning.

---

## ✅ What Was Built

### Environment Features
- **Multi-Agent Physics**: 5 agents with autonomous movement, rotation, collision dynamics
- **Foraging Mechanics**: Food pickup, transport, deposit at nest
- **Optimized Reward Structure**:
  - Food pickup: **+5.0**
  - Food deposit: **+10.0** (2x pickup incentivizes completion)
  - Wall collision: **-0.5** (increased from -0.1 after initial training)
  - Agent collision: **-0.25** (allows cooperative clustering)
  - Step penalty: **-0.01** (10x stronger than initial for efficiency)

### Observation Space (15 dimensions per agent)
| Component | Dims | Description |
|-----------|------|-------------|
| **Raycasts** | 5 | Distance to walls/agents in 90° cone (0-1) |
| **Nearest Food** | 3 | Distance + direction vector (x, y) |
| **Self State** | 2 | Speed + carrying food flag |
| **Nest Location** | 3 | Distance + direction vector (x, y) |
| **Nearest Agent** | 2 | Distance + bearing (-1 to +1) |

### Action Space (5 discrete actions)
- 0: No-op
- 1: Rotate left (3°/step)
- 2: Rotate right (3°/step)
- 3: Accelerate (+0.2, max 5.0)
- 4: Decelerate (-0.2, min 0.0)

### Training Configuration
- **Algorithm**: PPO (Stable Baselines3)
- **Policy**: MlpPolicy (shared across agents)
- **Learning Rate**: 3e-4 (initial), 1e-4 (fine-tuning)
- **Batch Size**: 64
- **Rollout Steps**: 2048
- **Episodes**: Truncate after 2500 steps
- **Checkpoints**: Every 10,000 steps

---

## 🚀 Training Workflow

### 1. Fresh Training
```bash
# Default 1M timesteps
uv run train.py

# Custom timesteps
uv run train.py --steps 2000000
```

### 2. Continued Training from Checkpoint
```bash
# Continue from specific checkpoint
uv run train.py \
  --checkpoint models/ppo_swarm_5000000_steps.zip \
  --steps 1000000

# Timestep counter continues from checkpoint
# Logs append to TensorBoard automatically
```

### 3. Visualization
```bash
# View trained model (auto-enables debug HUD)
uv run play.py --model models/ppo_swarm_final.zip

# Adjust FPS for smoother playback
uv run play.py --model models/ppo_swarm_10621120_steps.zip --fps 60

# Controls:
# - D: Toggle debug HUD
# - TAB: Switch selected agent
# - R: Reset environment
# - ESC: Quit
```

### 4. Record a Gameplay Video
Capture a full episode to MP4 using OpenCV (requires `opencv-python`):
```bash
uv run record_video.py
```
*Saves `demo_debug.mp4` (2500 frames @ 60 FPS) to the project root. Edit `record_video.py` to change the model path, output filename, frame count, or resolution.*

### 5. Validate Environment & Smoke-Test Training
Before a long training run, verify the environment API and the training pipeline:
```bash
# Check raw SwarmEnv + SwarmVecEnv compatibility with SB3
uv run check_env.py

# Short end-to-end training smoke test (500 steps, auto-cleans up)
uv run test_train.py
```

### 6. Monitor Training (TensorBoard)
```bash
tensorboard --logdir logs/

# Key metrics:
# - rollout/ep_rew_mean: Episode rewards (should increase)
# - rollout/ep_len_mean: Episode length (~2500)
# - train/entropy_loss: Exploration (decreases as policy stabilizes)
# - train/approx_kl: Policy update size
# - train/explained_variance: Value function quality
```

---

## 📊 Training Results & Insights

### Convergence Timeline
| Steps | ep_rew_mean | Behavior |
|-------|-------------|----------|
| 0-1M | -20 to 0 | Random exploration, learning navigation |
| 1M-3M | 0 to +10 | Food pickup learned, occasional deposits |
| 3M-5M | +10 to +16 | Efficient foraging cycles, some collisions |
| 5M-7M | +13 to +20* | Fine-tuning with increased wall penalty |
| 7M-10M | +20 to +30* | Minimal collisions, optimal paths |

*After wall collision penalty increased to -0.5

### Successful Training Metrics (at 10.6M steps)
```
rollout/ep_rew_mean:     ~25-30
rollout/ep_len_mean:     2500 (max)
train/approx_kl:         0.011 (stable)
train/entropy_loss:      -1.09 (deterministic policy)
train/explained_variance: 0.70-0.75 (good value estimation)
```

### Observed Behaviors ✅
- ✅ **Foraging**: Actively seek and pickup food
- ✅ **Depositing**: Navigate to nest and deposit
- ✅ **Efficiency**: Minimize wasted steps (step penalty working)
- ✅ **Agent Avoidance**: Occasional collisions, mostly avoid
- ✅ **Wall Avoidance**: Minimal wall hits after reward tuning
- ✅ **Cooperation**: Natural clustering near food sources

---

## 🔧 Reward Tuning Journey

### Initial Configuration (Steps 0-5M)
```python
REWARD_WALL_COLLISION = -0.1
REWARD_AGENT_COLLISION = -0.5
REWARD_STEP = -0.001
```
**Issue**: Wall collisions not penalized enough (50x weaker than food rewards)

### Optimized Configuration (Steps 5M+)
```python
REWARD_WALL_COLLISION = -0.5   # 5x increase
REWARD_AGENT_COLLISION = -0.25 # Reduced 50% to allow clustering
REWARD_STEP = -0.01            # 10x increase for efficiency
```
**Result**: Significant reduction in wall collisions, maintained cooperative behavior

### Why These Changes Work
- **Wall collision (-0.5)**: Now hitting 10 walls = losing 1 food pickup
- **Agent collision (-0.25)**: Allows natural clustering without excessive avoidance
- **Step penalty (-0.01)**: Creates urgency (500 wasted steps = 1 food pickup lost)

---

## 🎓 Training Best Practices

### 1. Start with Default Rewards
- Let agents learn core task (foraging/depositing) first
- Wait for convergence (3-5M steps)

### 2. Identify Behavioral Issues
- Use TensorBoard metrics
- Visualize with `play.py`
- Look for repeated mistakes (e.g., wall collisions)

### 3. Adjust Rewards Gradually
- Increase penalties 2-5x at a time
- Continue from best checkpoint
- Monitor for sudden performance drops

### 4. Fine-Tune with Lower Learning Rate
```bash
# Manually edit train.py or use:
# learning_rate = 1e-4  (down from 3e-4)
```

### 5. Track Multiple Checkpoints
- Save every 10k steps
- Test various checkpoints with `play.py`
- Choose best based on visual behavior + metrics

---

## 🐛 Common Issues & Solutions

### Issue: Episodes Never End (No TensorBoard Metrics)
**Symptom**: `rollout/ep_rew_mean` not showing
**Solution**: Episodes now truncate after 2500 steps ✅

### Issue: Training Plateau
**Symptom**: `ep_rew_mean` flat for 1M+ steps
**Solutions**:
- Increase reward magnitude (2-5x)
- Reduce learning rate (1e-4 or 5e-5)
- Check if task is already solved
- Add curriculum learning (harder obstacles)

### Issue: Agents Collide Too Much
**Solutions**:
- Increase collision penalty (try -0.5 to -1.0)
- Check ray coverage (90° might be too narrow)
- Increase entropy coefficient (more exploration)

### Issue: Slow Training
**Solutions**:
- Use CPU for PPO+MlpPolicy (faster than GPU!)
- Reduce `n_steps` to 1024 (less memory, more frequent updates)
- Decrease `batch_size` to 32

### Issue: Lag in play.py
**Solution**: Reduce FPS with `--fps 30` ✅

---

## 🔬 Advanced Experimentation

### Hyperparameter Tuning
```python
# In train.py, try:
learning_rate = 1e-4 to 5e-4
n_steps = 1024 to 4096
batch_size = 32 to 128
ent_coef = 0.001 to 0.05
```

### Environment Variations
```python
# In config.py:
NUM_AGENTS = 10        # Scale up swarm
MAX_FOOD = 50          # More food sources
REWARD_STEP = -0.02    # Even more efficiency pressure
```

### Multi-Stage Training
1. Train 2M steps with easy rewards
2. Increase penalties progressively
3. Fine-tune with low learning rate

---

## 📁 Project Structure

```
.
├── env/
│   ├── swarm_env.py       # Core Gymnasium environment
│   └── wrappers.py        # SwarmVecEnv wrapper for SB3
├── models/                # Checkpoints saved here
│   └── ppo_swarm_*.zip
├── logs/                  # TensorBoard logs
│   └── PPO_*/
├── experiments/           # Evaluation & benchmarking
│   ├── core.py            # Shared ExperimentRunner base class
│   ├── runners/
│   │   ├── scalability.py # Scalability benchmark (agent count sweep)
│   │   ├── efficiency.py  # AI vs Random baseline comparison
│   │   └── human_vs_ai.py # Human vs AI 1v1 benchmark
│   ├── analysis/          # Plotting scripts and generated figures
│   └── results/           # CSV exports and FINAL_REPORT.md
├── docs/
│   └── TRAINING_GUIDE.md  # This file
├── config.py              # All hyperparameters
├── train.py               # Training script (with checkpoint support)
├── play.py                # Visualization tool
├── manual_control.py      # Manual testing / human baseline
├── record_video.py        # Record gameplay to MP4
├── check_env.py           # Environment validation (SB3 API check)
├── test_train.py          # Smoke test: short training run
└── README.md              # Quick reference
```

---

## 🎯 Google Colab Training

### Setup
```python
# Clone the repository
!git clone https://github.com/FireHead90544/RLSwarm
%cd RLSwarm

# Install dependencies
!pip install -q stable-baselines3[extra] shimmy gymnasium pygame tensorboard

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```
### Monitor (Start TensorBoard)
```python
%load_ext tensorboard
%tensorboard --logdir logs/
```

### Training
```python
# Fresh start
!python train.py --steps 2000000

# Continue from checkpoint
!python train.py \
  --checkpoint models/ppo_swarm_5000000_steps.zip \
  --steps 2000000
```

### Download Results
```python
!zip -r results.zip models/ logs/
!cp results.zip /content/drive/MyDrive/
```

---

## 🏆 Success Criteria

### Training Complete When:
- ✅ `ep_rew_mean` > 20 for 500k+ steps
- ✅ Visual inspection shows efficient foraging
- ✅ Wall collisions < 2 per episode
- ✅ Food deposit rate > 3 per episode
- ✅ `explained_variance` > 0.65

### Your Results (10.6M steps):
- ✅ `ep_rew_mean`: ~25-30
- ✅ Minimal collisions
- ✅ Efficient foraging cycles
- ✅ Cooperative clustering
- ✅ **Training successful!**

---

## 📖 Key Learnings

1. **Start Simple**: Basic rewards first, tune later
2. **Visualize Often**: `play.py` reveals issues metrics miss
3. **Iterate on Rewards**: Don't be afraid to 5x a penalty
4. **Trust Convergence**: Plateaus are normal after 3-5M steps
5. **Use Checkpoints**: Test multiple saves to find best model
6. **Episode Limits Matter**: Without truncation, no TensorBoard metrics
7. **GPU Not Always Better**: PPO+MLP runs faster on CPU

---

## 🔬 Benchmarking & Experiments

The `experiments/` directory contains runners for the three benchmark experiments reported in the paper. Run them from the project root:

```bash
# Experiment A: Scalability (test trained policy on 3, 5, 10, 15 agents)
uv run -m experiments.runners.scalability --model models/best_model.zip

# Experiment C: Efficiency (AI-trained vs random baseline, 10 episodes each)
uv run -m experiments.runners.efficiency --model models/best_model.zip

# Experiment D: Human vs AI (you play 5 episodes, then AI plays 5)
uv run -m experiments.runners.human_vs_ai --model models/best_model.zip
```

Results (CSV) are written to `experiments/results/`. Plots can be regenerated from the analysis scripts in `experiments/analysis/`.

---

## 🚀 Next Steps

1. **Experiment with metrics/graphs** (you're doing this tomorrow!)
2. **Try different agent counts** (10-20 agents)
3. **Add obstacles** to environment
4. **Implement evaluation callback** for best model selection
5. **Test transfer learning** (pretrain on simple, transfer to complex)
6. **Analyze emergent behaviors** in long episodes

---

## 🎉 Congratulations!

You've successfully:
- ✅ Built a custom Gymnasium environment
- ✅ Trained PPO agents for 10M+ steps
- ✅ Diagnosed and fixed reward scaling issues
- ✅ Achieved efficient cooperative foraging
- ✅ Implemented checkpoint-based training
- ✅ Created reproducible training pipeline

Your swarm is ready for research! 🐝

---

## 📚 References

- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium API](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)

---

**Last Updated**: 2025-11-27  
**Training Status**: ✅ Converged and Optimized (Could be further improved with more tuning) 
**Best Model**: `ppo_swarm_9771120_steps_reshaped.zip` to `ppo_swarm_10621120_steps_reshaped.zip`
