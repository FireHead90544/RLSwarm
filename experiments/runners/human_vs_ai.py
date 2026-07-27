import argparse
import pygame
import numpy as np
from stable_baselines3 import PPO
from env.swarm_env import SwarmEnv
from experiments.core import ExperimentRunner
from config import *

def run_human_vs_ai(model_path, episodes=5):
    """Experiment D: Human vs AI Benchmark (1v1)"""
    runner = ExperimentRunner("human_vs_ai")
    print(f"Starting Experiment D: Human vs AI (1v1)...")
    print("Protocol: You play 5 episodes, then AI plays 5 episodes.")
    print("Controls: Arrow Keys to move/rotate.")
    
    # 1. Human
    print("\n--- PHASE 1: HUMAN ---")
    env = SwarmEnv(num_agents=1, render_mode="human", max_episode_steps=2000)
    
    for ep in range(episodes):
        print(f"Human Episode {ep+1}/{episodes}")
        obs, _ = env.reset()
        env.render() # Initialize video system
        
        # Custom loop for manual control
        # We can't use runner.run_episode directly because we need keyboard input
        # So we implement a manual loop but use runner's logging logic manually?
        # Or we pass a "human_policy" function to runner?
        # Runner expects model.predict or random.
        # Let's write a custom loop here and log manually using runner.log_episode
        
        net_reward = 0.0
        steps = 0
        
        # Counters
        total_wall_collisions = 0
        total_agent_collisions = 0 # Should be 0 with 1 agent
        food_touched = 0
        food_deposited = 0
        
        running = True
        while running:
            # Handle Input
            action = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]: action = 1
            elif keys[pygame.K_RIGHT]: action = 2
            elif keys[pygame.K_UP]: action = 3
            elif keys[pygame.K_DOWN]: action = 4
            
            # Quit check
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    env.close()
                    return
            
            obs, rewards, terminated, truncated, infos = env.step([action])
            r = rewards[0]
            net_reward += r
            
            # Event Detection (Same logic as core.py)
            r_clean = r - REWARD_STEP
            if np.isclose(r_clean, REWARD_WALL_COLLISION, atol=0.01):
                total_wall_collisions += 1
            elif np.isclose(r_clean, REWARD_FOOD_PICKUP, atol=0.01):
                food_touched += 1
            elif np.isclose(r_clean, REWARD_FOOD_DEPOSIT, atol=0.01):
                food_deposited += 1
                
            steps += 1
            # env.render() is called inside step() when render_mode="human"
            
            if terminated or truncated:
                running = False
                
        # Log
        metrics = {
            "episode": ep,
            "policy_type": "human",
            "steps": steps,
            "num_agents": 1,
            "net_total_reward": net_reward,
            "mean_agent_reward": net_reward, # 1 agent
            "total_wall_collisions": total_wall_collisions,
            "total_agent_collisions": 0,
            "mean_wall_collisions_per_agent": total_wall_collisions,
            "mean_agent_collisions_per_agent": 0,
            "total_food_touched": food_touched,
            "total_food_deposited": food_deposited,
            "mean_food_touched_per_agent": food_touched,
            "mean_food_deposited_per_agent": food_deposited,
            "efficiency_score": net_reward / steps
        }
        runner.log_episode(metrics)
        print(f"  Score: {net_reward:.2f}")
        
    env.close()
    
    # 2. AI
    print("\n--- PHASE 2: AI ---")
    model = PPO.load(model_path)
    env = SwarmEnv(num_agents=1, render_mode="human", max_episode_steps=2000) # Render so user can see
    
    for ep in range(episodes):
        print(f"AI Episode {ep+1}/{episodes}")
        runner.run_episode(
            env, model, 
            episode_idx=ep, 
            extra_tags={"policy_type": "ai"}
        )
        # Note: run_episode doesn't render by default unless we modify it or env.render() is called in step
        # SwarmEnv.step() calls render() if render_mode is "human".
        # So it should work.
        
    env.close()
    print("Experiment D Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    run_human_vs_ai(args.model)
