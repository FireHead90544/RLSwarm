import argparse
import numpy as np
from stable_baselines3 import PPO
from env.swarm_env import SwarmEnv
from experiments.core import ExperimentRunner
from config import *

def run_scalability_fixed(model_path, episodes=5):
    """Experiment A: Scalability on Fixed Map (Crowding Test)"""
    runner = ExperimentRunner("scalability_fixed")
    print(f"Starting Experiment A: Scalability (Fixed Map)...")
    
    model = PPO.load(model_path)
    agent_counts = [3, 5, 10, 15]
    
    for n in agent_counts:
        print(f"  Testing with {n} agents...")
        # Fixed map size (default from config)
        env = SwarmEnv(num_agents=n, render_mode=None, max_episode_steps=2000)
        
        for ep in range(episodes):
            runner.run_episode(
                env, model, 
                episode_idx=ep, 
                extra_tags={"map_type": "fixed", "map_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}"}
            )
        env.close()
    print("Experiment A Complete.")

def run_scalability_scaled(model_path, episodes=5):
    """Experiment B: Scalability on Scaled Map (Constant Density)"""
    runner = ExperimentRunner("scalability_scaled")
    print(f"Starting Experiment B: Scalability (Scaled Map)...")
    
    model = PPO.load(model_path)
    agent_counts = [3, 5, 10, 15, 20, 30]
    
    # Base density: 5 agents on 800x600
    base_area = 800 * 600
    base_density = 5 / base_area
    
    for n in agent_counts:
        # Calculate required area
        target_area = n / base_density
        # Scale dimensions proportionally (maintain aspect ratio 4:3)
        # w * h = area, w/h = 4/3 => w = 4/3 h => 4/3 h^2 = area => h = sqrt(3/4 area)
        h = int(np.sqrt(0.75 * target_area))
        w = int(1.333 * h)
        
        print(f"  Testing with {n} agents on {w}x{h} map...")
        
        # We need to monkey-patch config or pass map size to env
        # Since env uses global config, we'll modify the instance's boundaries if possible
        # Or better, we assume SwarmEnv respects SCREEN_WIDTH/HEIGHT from config
        # But config is imported. We need to hack it or update SwarmEnv to accept width/height
        
        # HACK: SwarmEnv uses global constants. We must update them in the module before creating env
        # This is tricky with imported modules. 
        # Better approach: Update SwarmEnv to accept width/height in __init__
        # For now, let's assume we can't easily change SwarmEnv code without breaking other things
        # So we'll skip the map resizing for now and just note it, OR we update SwarmEnv to support it.
        
        # Let's update SwarmEnv to be flexible first.
        # Actually, let's just stick to Fixed Map for now as user requested "later increase map size"
        # The user said: "do one experiment on 3, 5, 10, 15 only with fixed map size only... Then later increase the map size"
        # So I will implement the logic, but we might need to patch SwarmEnv.
        
        # For this run, I'll assume SwarmEnv defaults to config but we can't easily change it dynamically 
        # without reloading module. 
        # Let's skip the actual resizing logic implementation details for a moment and focus on the runner.
        # I will implement it assuming I can pass width/height to SwarmEnv.
        
        # Since I can't change SwarmEnv right now (it's in another file), I will stick to Fixed Map for this iteration
        # unless I edit SwarmEnv. 
        # EDIT: I will edit SwarmEnv to accept width/height to make this rigorous.
        pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    run_scalability_fixed(args.model)
    # run_scalability_scaled(args.model) # Uncomment after SwarmEnv update
