import argparse
import pygame
from stable_baselines3 import PPO

from env.swarm_env import SwarmEnv
from config import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--fps", type=int, default=30, help="Rendering FPS")
    args = parser.parse_args()
    
    # Load Model
    model = PPO.load(args.model)
    print("Model loaded.")
    
    env = SwarmEnv(num_agents=5, render_mode="human")
    env.debug = True  # Auto-enable debug mode
    env.selected_agent = 0
    
    obs, _ = env.reset()
    
    env.render()
    
    running = True
    clock = pygame.time.Clock()
    
    print("Controls: TAB (switch agent), R (reset), D (toggle debug), ESC (quit)")
    
    while running:
        # Handle only quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    # Switch selected agent
                    env.selected_agent = (env.selected_agent + 1) % env.num_agents
                    print(f"Selected Agent: {env.selected_agent}")
                elif event.key == pygame.K_r:
                    # Reset
                    obs, _ = env.reset()
                    print("Environment reset.")
                elif event.key == pygame.K_d:
                    # Toggle debug mode
                    env.debug = not env.debug
                    print(f"Debug mode: {'ON' if env.debug else 'OFF'}")
        
        # Get actions from model
        actions = []
        for i in range(env.num_agents):
            action, _states = model.predict(obs[i], deterministic=True)
            actions.append(action)
        
        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Auto-reset if episode ends
        if terminated or truncated:
            obs, _ = env.reset()
            print("Episode ended. Auto-reset.")
        
        # Render is called in step(), but we control FPS here
        clock.tick(args.fps)
    
    env.close()
    print("Visualization closed.")

if __name__ == "__main__":
    main()
