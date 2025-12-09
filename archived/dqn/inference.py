import torch
import cv2
import numpy as np
import pygame
import argparse
import os

os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from core.environment import GameEnvironment
from rl.network import SharedDQN
from rl.dqn_agent import SharedPolicyAgent
from core.config import SCREEN_WIDTH, SCREEN_HEIGHT
from rl.config import DEVICE

def generate_video(checkpoint_path, output_path, num_agents=1, max_steps=2000, fps=60, debug=False):
    """
    Generates a video of the simulation using a model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.
        output_path (str): Path to save the output MP4 video.
        max_steps (int): Maximum number of steps/frames for the video.
        fps (int): Frames per second for the output video.
        debug (bool): Whether to render debug overlays (sensor rays, HUD).
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    print(f"Initializing environment for video generation...")
    # We use headless=False to ensure the screen surface is created for rendering,
    # but we won't be creating a visible window.
    env = GameEnvironment(num_agents=num_agents, manual_control=False, debug=debug, headless=False)

    # Determine observation and action sizes from the environment
    obs_example = env.agents[0].compute_sensors(env.foods, env.agents)
    obs_size = len(obs_example)
    action_size = 5

    # Load the trained model
    print(f"Loading model from {checkpoint_path}...")
    policy_net = SharedDQN(obs_size, action_size).to(DEVICE)

    # Use torch.load with weights_only=False as the checkpoint contains other data
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    policy_net.eval()  # Set the network to evaluation mode

    # Create an agent to select actions
    agent = SharedPolicyAgent(policy_net, action_size)
    agent.epsilon = 0.05  # Use a low epsilon for near-greedy evaluation

    # Setup video writer using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (SCREEN_WIDTH, SCREEN_HEIGHT))

    print(f"Starting simulation for {max_steps} steps...")
    env.reset()
    for step in range(max_steps):
        if step % 200 == 0:
            print(f"  ... processing frame {step}/{max_steps}")

        # All agents select and take an action based on the shared policy
        observations = [a.compute_sensors(env.foods, env.agents) for a in env.agents]
        actions = [agent.select_action(obs) for obs in observations]
        for a, action in zip(env.agents, actions):
            a.take_action(action)

        # Step the environment logic
        env.step()

        # Render the current state to the environment's screen surface
        env.render()

        # Convert the pygame surface to a numpy array for OpenCV
        frame = pygame.surfarray.array3d(env.screen)
        frame = np.swapaxes(frame, 0, 1)  # Pygame is (width, height), OpenCV is (height, width)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV

        video_writer.write(frame)

    video_writer.release()
    pygame.quit()
    print(f"✅ Video successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a DeepQSwarm checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint file (e.g., checkpoints/checkpoint_EP_500.pt).")
    parser.add_argument("--output", type=str, default="simulation.mp4", help="Path to save the output video file.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps (frames) to record in the video.")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for the output video.")
    parser.add_argument("--agents", type=int, default=1, help="Number of agents to spawn in the environment.")
    parser.add_argument("--debug", action="store_true", help="Enable debug overlays (sensor rays, HUD) in the video.")

    args = parser.parse_args()

    generate_video(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_agents=args.agents,
        max_steps=args.steps,
        fps=args.fps,
        debug=args.debug
    )

