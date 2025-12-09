import argparse
import torch
import os
from tqdm import tqdm
from rl.network import SharedDQN
from rl.replay_buffer import ReplayBuffer
from rl.trainer import Trainer
from rl.dqn_agent import SharedPolicyAgent
from rl.config import *
from core.config import FPS
from core.environment import GameEnvironment

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Ensure directory exists

if USE_COLAB:
    from torch.utils.tensorboard import SummaryWriter

def save_checkpoint(episode, policy_net, trainer, replay_buffer, agent):
    """Saves the training state to a checkpoint file."""
    state = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'replay_buffer': replay_buffer,
        'epsilon': agent.epsilon,
    }
    filepath = f"{MODEL_SAVE_PATH}checkpoint_EP_{episode}.pt"
    torch.save(state, filepath)
    print(f"✅ Saved checkpoint: {filepath}")

def train(visual=False, checkpoint_path=None):
    env = GameEnvironment(num_agents=NUM_AGENTS, manual_control=False, debug=False, headless=not visual)

    # Initialize obs_example by computing sensors once
    _ = env.agents[0].compute_sensors(env.foods, env.agents)
    obs_example = env.agents[0].last_sensors
    obs_size = len(obs_example)
    action_size = 5

    policy_net = SharedDQN(obs_size, action_size).to(DEVICE)
    target_net = SharedDQN(obs_size, action_size).to(DEVICE)
    
    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
    trainer = Trainer(policy_net, target_net, replay_buffer)
    agent = SharedPolicyAgent(policy_net, action_size)

    start_episode = 0
    if checkpoint_path:
        print(f"Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        replay_buffer = checkpoint['replay_buffer']
        trainer.replay_buffer = replay_buffer  # Update trainer's buffer reference
        agent.epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1

    target_net.load_state_dict(policy_net.state_dict())

    writer = None
    if USE_COLAB:
        writer = SummaryWriter(log_dir=TENSORBOARD_LOGDIR, purge_step=start_episode if checkpoint_path else None)

    print(f"Training on {DEVICE} | Obs size={obs_size} | Actions={action_size}")

    try:
        for episode in range(start_episode, NUM_EPISODES):
            env.reset()
            total_reward = 0
            losses = []

            # Initialize obs for each agent
            observations = [a.compute_sensors(env.foods, env.agents) for a in env.agents]

            for step in range(MAX_STEPS_PER_EPISODE):
                for a in env.agents:
                    a.reward = 0.0

                actions = []
                for a, obs in zip(env.agents, observations):
                    action = agent.select_action(obs)
                    actions.append(action)
                    a.take_action(action)

                # Step environment
                env.step()
                if visual:
                    env.render()
                    env.clock.tick(FPS)

                # Compute next observations and rewards
                next_observations = [a.compute_sensors(env.foods, env.agents) for a in env.agents]
                rewards = [a.reward for a in env.agents]

                # Push to shared replay buffer
                done = step == MAX_STEPS_PER_EPISODE - 1
                for obs, action, reward, next_obs in zip(observations, actions, rewards, next_observations):
                    replay_buffer.push(obs, action, reward, next_obs, done)

                # Train shared policy
                if step % TRAIN_FREQ == 0:
                    loss = trainer.train_step()
                    if loss is not None:
                        losses.append(loss)

                total_reward += sum(rewards)
                observations = next_observations

            # Logging
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            avg_agent_reward = total_reward / NUM_AGENTS
            print(f"[Ep {episode:04d}] Steps={step+1} | Total={total_reward:.2f} | Avg/Agent={avg_agent_reward:.2f} | Eps={agent.epsilon:.3f} | Loss={avg_loss:.5f}")

            agent.update_epsilon()

            if USE_COLAB:
                writer.add_scalar("Loss/avg", avg_loss, global_step=episode)
                writer.add_scalar("Reward/total", total_reward, global_step=episode)
                writer.add_scalar("Reward/avg_per_agent", avg_agent_reward, global_step=episode)
                writer.add_scalar("Epsilon", agent.epsilon, global_step=episode)

            if episode % SAVE_INTERVAL == 0 and episode > 0:
                save_checkpoint(episode, policy_net, trainer, replay_buffer, agent)

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        save_checkpoint(episode, policy_net, trainer, replay_buffer, agent)
    finally:
        if USE_COLAB:
            writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="Enable visual (Pygame) training view")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    train(visual=args.visual, checkpoint_path=args.checkpoint)