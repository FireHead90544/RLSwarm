import pygame
import numpy as np
from env.swarm_env import SwarmEnv
from config import *

def main():
    env = SwarmEnv(num_agents=5, render_mode="human")
    obs, _ = env.reset()
    env.render() # Initialize video system
    
    running = True
    selected_agent_idx = 0
    debug_mode = True
    
    print("Manual Control Mode")
    print("Controls:")
    print("  Arrow Keys: Move/Rotate Selected Agent")
    print("  TAB: Switch Selected Agent")
    print("  D: Toggle Debug Mode")
    print("  R: Reset")
    print("  ESC: Quit")
    
    while running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    selected_agent_idx = (selected_agent_idx + 1) % env.num_agents
                    print(f"Selected Agent: {selected_agent_idx}")
                elif event.key == pygame.K_d:
                    debug_mode = not debug_mode
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    print("Reset.")

        keys = pygame.key.get_pressed()
        # Action mapping: 0: No-op, 1: Left, 2: Right, 3: Accelerate, 4: Decelerate
        if keys[pygame.K_LEFT]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_UP]:
            action = 3
        elif keys[pygame.K_DOWN]:
            action = 4
            
        actions = [0] * env.num_agents
        actions[selected_agent_idx] = action
        
        # Call step to update physics
        obs, rewards, terminated, truncated, infos = env.step(actions)
        
        env.debug = debug_mode
        env.selected_agent = selected_agent_idx
        
        # env.step() calls render() automatically if render_mode="human"

    env.close()

if __name__ == "__main__":
    main()
