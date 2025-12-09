import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecMonitor

from env.swarm_env import SwarmEnv
from env.wrappers import SwarmVecEnv
from config import *

def make_env():
    return SwarmEnv(num_agents=5)

def main():
    parser = argparse.ArgumentParser(description="Train PPO on Swarm Environment")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to checkpoint to continue training from (e.g., models/ppo_swarm_10000_steps.zip)")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                       help="Total timesteps to train (default from config)")
    args = parser.parse_args()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Create Environment
    # We use our custom SwarmVecEnv which wraps a single SwarmEnv
    # but presents it as `num_agents` environments.
    env = SwarmVecEnv(make_env)
    
    # Wrap in VecMonitor to log episode rewards/lengths
    env = VecMonitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="ppo_swarm"
    )

    # Initialize or Load PPO
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model = PPO.load(
            args.checkpoint,
            env=env,
            tensorboard_log=LOG_DIR
        )
        print("Checkpoint loaded. Continuing training...")
    else:
        print("Starting fresh training...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01, # Encourage exploration
        )

    print(f"Training for {args.steps} timesteps...")
    print(f"Device: {model.device}")
    print(f"Episode Length: {make_env().max_episode_steps} steps")

    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False if args.checkpoint else True  # Continue timestep count if resuming
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        model.save(os.path.join(MODEL_DIR, "ppo_swarm_interrupted"))
        env.close()
        print("Model saved and environment closed.")
        return

    # Save final model
    model.save(os.path.join(MODEL_DIR, "ppo_swarm_final"))
    env.close()
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
