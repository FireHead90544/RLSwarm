import torch

# Toggle for GPU/Colab usage
USE_COLAB = False  # set to True on Colab
DEVICE = torch.device("cuda" if USE_COLAB and torch.cuda.is_available() else "cpu")

# Training Hyperparameters
NUM_AGENTS = 10
BATCH_SIZE = 128
GAMMA = 0.99
LR = 5e-4
REPLAY_CAPACITY = 100000
MIN_REPLAY_SIZE = 2000
TARGET_UPDATE_FREQ = 2500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.9995

# Training Loop
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 2000
TRAIN_FREQ = 4

# Logging
LOG_INTERVAL = 100
SAVE_INTERVAL = 500

# Model Paths
MODEL_SAVE_PATH = "checkpoints/"
TENSORBOARD_LOGDIR = "runs/"
