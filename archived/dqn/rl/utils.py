import torch
import numpy as np

def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(0, q_values.shape[1])
    return q_values.argmax(dim=1).item()
