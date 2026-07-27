# experimentation/run_dqn_eval.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import pandas as pd
from core.environment import GameEnvironment
from rl.network import SharedDQN
from rl.config import DEVICE

def evaluate_dqn(checkpoint_path, num_episodes=10, num_agents=5, max_steps=2000):
    """
    Evaluate DQN model and collect metrics.
    """
    env = GameEnvironment(num_agents=num_agents, manual_control=False, debug=False)
    obs_size = len(env.agents[0].compute_sensors(env.foods, env.agents))
    action_size = 5

    model = SharedDQN(obs_size, action_size).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()

    print(f"Loaded DQN model from {checkpoint_path}")
    print(f"Running {num_episodes} episodes with {num_agents} agents...")

    results = []

    for ep in range(num_episodes):
        env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Store rewards before step
            rewards_before = [a.reward for a in env.agents]
            
            # Get actions from model
            for a in env.agents:
                obs = a.compute_sensors(env.foods, env.agents)
                with torch.no_grad():
                    q_values = model(torch.as_tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0))
                    action = q_values.argmax(dim=1).item()
                a.take_action(action)
            
            # Step environment
            env.step()
            
            # Accumulate rewards (agents have their rewards updated during step)
            step_reward = sum(a.reward - rb for a, rb in zip(env.agents, rewards_before))
            episode_reward += step_reward
        
        # Collect metrics
        total_reward = episode_reward
        avg_reward_per_agent = total_reward / num_agents
        
        results.append({
            'episode': ep,
            'num_agents': num_agents,
            'total_reward': total_reward,
            'avg_reward_per_agent': avg_reward_per_agent,
            'steps': max_steps
        })
        
        print(f"  Episode {ep+1}/{num_episodes}: Total Reward = {total_reward:.2f}, Avg/Agent = {avg_reward_per_agent:.2f}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_EP_4000.pt"
    df = evaluate_dqn(checkpoint_path, num_episodes=10, num_agents=5, max_steps=2500)
    
    # Save results
    output_path = "experimentation/results/dqn_evaluation.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Mean Total Reward: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
    print(f"Mean Avg Reward/Agent: {df['avg_reward_per_agent'].mean():.2f} ± {df['avg_reward_per_agent'].std():.2f}")
