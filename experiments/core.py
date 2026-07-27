import os
import csv
import time
import numpy as np
import pandas as pd
from datetime import datetime
from env.swarm_env import SwarmEnv
from config import *

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

class ExperimentRunner:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(RESULTS_DIR, f"{experiment_name}_{self.timestamp}.csv")
        self.data = []
        
    def log_episode(self, metrics):
        """Log metrics for a single episode."""
        self.data.append(metrics)
        
        # Write to CSV immediately (append mode)
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
            
    def run_episode(self, env, model=None, max_steps=2500, episode_idx=0, extra_tags=None):
        """Run a single episode and return comprehensive metrics."""
        obs, _ = env.reset()
        
        # Cumulative Metrics
        net_reward = 0.0
        agent_rewards = np.zeros(env.num_agents)
        
        # Collision Counters
        total_wall_collisions = 0
        total_agent_collisions = 0
        agent_wall_collisions = np.zeros(env.num_agents)
        agent_agent_collisions = np.zeros(env.num_agents)
        
        # Food Metrics
        food_touched = 0
        food_deposited = 0
        agent_food_touched = np.zeros(env.num_agents)
        agent_food_deposited = np.zeros(env.num_agents)
        
        steps = 0
        
        for _ in range(max_steps):
            if model:
                actions = []
                for i in range(env.num_agents):
                    action, _ = model.predict(obs[i], deterministic=True)
                    actions.append(action)
            else:
                actions = [env.action_space.sample() for _ in range(env.num_agents)]
                
            obs, rewards, terminated, truncated, infos = env.step(actions)
            
            # Update Rewards
            net_reward += sum(rewards)
            agent_rewards += rewards
            
            # Analyze Rewards to infer events (Hack since env doesn't return info dicts yet)
            # We check if reward matches specific constants (approximate float matching)
            
            for i, r in enumerate(rewards):
                # Remove step penalty from consideration
                r_clean = r - REWARD_STEP
                
                # Check Wall Collision
                # Note: Agent could hit wall AND pickup food in same step, but unlikely with current physics
                # We'll assume dominant event or check ranges
                
                # Precise checking requires modifying env to return infos, but for now:
                if np.isclose(r_clean, REWARD_WALL_COLLISION, atol=0.01):
                    total_wall_collisions += 1
                    agent_wall_collisions[i] += 1
                elif np.isclose(r_clean, REWARD_AGENT_COLLISION, atol=0.01):
                    total_agent_collisions += 1 # Note: this counts for BOTH agents involved usually
                    agent_agent_collisions[i] += 1
                elif np.isclose(r_clean, REWARD_FOOD_PICKUP, atol=0.01):
                    food_touched += 1
                    agent_food_touched[i] += 1
                elif np.isclose(r_clean, REWARD_FOOD_DEPOSIT, atol=0.01):
                    food_deposited += 1
                    agent_food_deposited[i] += 1
            
            steps += 1
            if terminated or truncated:
                break
                
        # Compile Metrics
        metrics = {
            "episode": episode_idx,
            "steps": steps,
            "num_agents": env.num_agents,
            
            # Rewards
            "net_total_reward": net_reward,
            "mean_agent_reward": np.mean(agent_rewards),
            
            # Collisions
            "total_wall_collisions": total_wall_collisions,
            "total_agent_collisions": total_agent_collisions / 2, # Divide by 2 since recorded for both
            "mean_wall_collisions_per_agent": np.mean(agent_wall_collisions),
            "mean_agent_collisions_per_agent": np.mean(agent_agent_collisions),
            
            # Food
            "total_food_touched": food_touched,
            "total_food_deposited": food_deposited,
            "mean_food_touched_per_agent": np.mean(agent_food_touched),
            "mean_food_deposited_per_agent": np.mean(agent_food_deposited),
            
            # Efficiency
            "efficiency_score": net_reward / (steps * env.num_agents)
        }
        
        if extra_tags:
            metrics.update(extra_tags)
            
        self.log_episode(metrics)
        return metrics
