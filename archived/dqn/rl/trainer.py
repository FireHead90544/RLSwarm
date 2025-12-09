import torch
import torch.nn.functional as F
import torch.optim as optim
from .config import *
from .utils import to_tensor

class Trainer:
    def __init__(self, policy_net, target_net, replay_buffer):
        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LR)
        self.step_count = 0

    def train_step(self):
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        states = to_tensor(states, DEVICE)
        actions = to_tensor(actions, DEVICE).long().unsqueeze(1)
        rewards = to_tensor(rewards, DEVICE).unsqueeze(1)
        next_states = to_tensor(next_states, DEVICE)
        dones = to_tensor(dones, DEVICE).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # Q'(s’, a’)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + GAMMA * max_next_q * (1 - dones)

        # Loss and optimize
        loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target net
        self.step_count += 1
        if self.step_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
