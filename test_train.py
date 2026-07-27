import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from env.swarm_env import SwarmEnv
from env.wrappers import SwarmVecEnv
from config import *

def make_env():
    return SwarmEnv(num_agents=5)

def main():
    print("Starting Training Smoke Test...")
    
    # Create Environment
    env = SwarmVecEnv(make_env)
    
    # Initialize PPO with small parameters for quick test
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=128, # Short rollout
        batch_size=64,
        n_epochs=2,
    )

    print("Training for 500 steps...")
    model.learn(total_timesteps=500)
    
    print("Saving test model...")
    model.save("test_model")
    
    print("Smoke Test Passed!")
    env.close()
    if os.path.exists("test_model.zip"):
        os.remove("test_model.zip")

if __name__ == "__main__":
    main()
