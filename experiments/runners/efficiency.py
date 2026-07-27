import argparse
from stable_baselines3 import PPO
from env.swarm_env import SwarmEnv
from experiments.core import ExperimentRunner
from config import *

def run_efficiency(model_path, episodes=10):
    """Experiment C: Efficiency Benchmark (AI vs Random)"""
    runner = ExperimentRunner("efficiency_benchmark")
    print(f"Starting Experiment C: Efficiency Benchmark...")
    
    # 1. Trained Policy
    print("  Testing Trained Policy...")
    model = PPO.load(model_path)
    env = SwarmEnv(num_agents=5, render_mode=None, max_episode_steps=2000)
    
    for ep in range(episodes):
        runner.run_episode(
            env, model, 
            episode_idx=ep, 
            extra_tags={"policy_type": "trained"}
        )
    env.close()
    
    # 2. Random Policy
    print("  Testing Random Policy...")
    env = SwarmEnv(num_agents=5, render_mode=None, max_episode_steps=2000)
    
    for ep in range(episodes):
        runner.run_episode(
            env, model=None, # None triggers random actions in core.py
            episode_idx=ep, 
            extra_tags={"policy_type": "random"}
        )
    env.close()
    print("Experiment C Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    run_efficiency(args.model)
