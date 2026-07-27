import sys
import pygame
import cv2
import numpy as np
from stable_baselines3 import PPO
from env.swarm_env import SwarmEnv
from config import *

def main():
    # Load Model (Correct path relative to root)
    model_path = "neural_swarm_ppo/models/ppo_swarm_9771120_steps_reshaped"
    try:
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create Environment
    env = SwarmEnv(num_agents=5, render_mode="human")
    env.debug = True
    env.metadata["render_fps"] = 60  # Force env to render at 60 FPS if it uses clock.tick
    
    obs, _ = env.reset()
    
    # Video Setup
    fps = 60
    total_frames = 2500
    width, height = 800, 600
    
    # Use 'mp4v' codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = 'demo_debug.mp4'
    video = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    print(f"Recording video to {video_filename} for {total_frames} frames ({total_frames/fps:.2f}s)...")
    print("Window may appear slow due to recording overhead, this is normal.")
    
    # Initialize Pygame video system by rendering once
    env.render()
    
    frame_count = 0
    
    try:
        while frame_count < total_frames:
            # Handle Pygame events to prevent freezing
            pygame.event.pump()

            # Get actions
            actions = []
            for i in range(env.num_agents):
                action, _states = model.predict(obs[i], deterministic=True)
                actions.append(action)
            
            # Step
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Explicitly render (updates env.screen)
            env.render()
            
            # Capture frame
            if env.screen:
                try:
                    rgb_array = pygame.surfarray.array3d(env.screen)
                    rgb_array = np.transpose(rgb_array, (1, 0, 2))
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    video.write(bgr_array)
                except Exception as e:
                    print(f"\nError capturing frame: {e}")
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Recorded {frame_count}/{total_frames} frames", end='\r')

            if terminated or truncated:
                obs, _ = env.reset()
                
    except Exception as e:
        print(f"\nCRITICAL ERROR in loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        if 'video' in locals():
            video.release()
        if 'env' in locals():
            env.close()
        pygame.quit()
        print(f"\nVideo generation finished (maybe partial): {video_filename}")

if __name__ == "__main__":
    main()
