import torch
from core.environment import GameEnvironment
from rl.network import SharedDQN
from rl.config import DEVICE

def play():
    env = GameEnvironment(num_agents=5, manual_control=False, debug=True)
    obs_size = len(env.agents[0].compute_sensors(env.foods, env.agents))
    action_size = 5

    model = SharedDQN(obs_size, action_size).to(DEVICE)
    checkpoint_path = "checkpoints/checkpoint_EP_4000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()

    print(f"Loaded DQN model from {checkpoint_path}")
    print(f"Running ONE episode with {len(env.agents)} agents for 2500 steps")
    print("Press Ctrl+C to stop early\n")

    env.reset()
    episode_reward = 0.0
    
    for step in range(2500):
        # Track rewards before step
        rewards_before = [a.reward for a in env.agents]
        
        for a in env.agents:
            obs = a.compute_sensors(env.foods, env.agents)
            q_values = model(torch.as_tensor(obs, dtype=torch.float32).to(DEVICE).unsqueeze(0))
            action = q_values.argmax(dim=1).item()
            a.take_action(action)
        
        env.step()
        env.render()
        
        # Calculate step reward
        step_reward = sum(a.reward - rb for a, rb in zip(env.agents, rewards_before))
        episode_reward += step_reward
        
        # Print progress every 500 steps
        if (step + 1) % 500 == 0:
            print(f"Step {step + 1}/2500 - Total Reward: {episode_reward:.2f} ({episode_reward / len(env.agents):.2f}/agent)")
    
    print(f"\n=== FINAL RESULT ===")
    print(f"Total Steps: 2500")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Avg Reward/Agent: {episode_reward / len(env.agents):.2f}")

if __name__ == "__main__":
    play()
