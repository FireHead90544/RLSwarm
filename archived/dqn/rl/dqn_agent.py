import torch
import numpy as np
from .utils import to_tensor, epsilon_greedy
from .config import DEVICE, EPS_START, EPS_END, EPS_DECAY

class SharedPolicyAgent:
    def __init__(self, policy_net, action_size):
        self.policy_net = policy_net
        self.action_size = action_size
        self.epsilon = EPS_START

    def select_action(self, state):
        state_t = to_tensor(state, DEVICE).unsqueeze(0)
        q_values = self.policy_net(state_t)
        action = epsilon_greedy(q_values, self.epsilon)
        return action

    def update_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
