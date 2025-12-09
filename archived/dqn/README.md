# DeepQSwarm: A Multi-Agent Reinforcement Learning Simulation [Legacy DQN Implementation (Archived)]

This directory contains the initial Deep Q-Network (DQN) implementation used as a baseline for the research paper. 

**Note**: This code is archived and maintained for reproducibility purposes only. The main project uses the PPO implementation found in the root directory.

## 📉 Performance Context

As detailed in the research paper, this DQN implementation struggled to converge in the continuous multi-agent environment, achieving a mean reward of **-86.80** compared to PPO's **+83.20**.

## 📂 Structure

- `core/`: Legacy environment implementation (older version of `SwarmEnv`).
- `rl/`: DQN agent logic and training loop.
- `experimentation/`: Scripts used for the comparative analysis and plotting.
- `inference.py`: Script to run the trained DQN model.

## ⚠️ Usage

This code may require different dependencies than the main project. It is recommended to inspect the imports if you intend to run it.

### 1. Manual Control
Run the simulation in manual mode to control an agent with arrow keys and test the physics/environment.
```bash
python neuralswarm.py --manual
```

### 2. Train DQN Agent
Start training the Deep Q-Network from scratch. Checkpoints will be saved to `checkpoints/`.
```bash
python rl/train.py --checkpoint checkpoints/best_model.pth
```
*Note: You can resume training by modifying the script to load a checkpoint.*

### 3. Watch Trained Agent
Visualize a pre-trained model interacting with the environment.
```bash
python rl/play.py --model checkpoints/best_model.pth
```

### 4. Generate Replay Video
Run inference and save the output as a video file.
```bash
python inference.py --model checkpoints/best_model.pth --output replay.mp4
```

## Project Goal

The primary goal of this project is to simulate a swarm of autonomous agents and train them to perform a collective foraging task using multi-agent reinforcement learning (MARL). The agents learn to navigate their environment, avoid obstacles, find food, and bring it back to a central nest. This is achieved by training a single Deep Q-Network (DQN) that is shared by all agents, allowing them to learn a collaborative policy.

## Core Concepts

### Multi-Agent Reinforcement Learning (MARL)

MARL is a subfield of artificial intelligence that focuses on training multiple agents to solve problems in a shared environment. Unlike single-agent RL, MARL introduces the complexities of agent interactions, which can be cooperative, competitive, or a mix of both. In this project, the agents are cooperative, as they all work towards the same goal of maximizing food collection.

### Deep Q-Networks (DQN)

DQN is a type of reinforcement learning algorithm that uses a deep neural network to approximate the optimal action-value function (Q-function). The Q-function estimates the expected future reward for taking a specific action in a given state. By learning this function, the agent can choose the action that maximizes its expected reward. This project uses a DQN to teach the agents how to behave in the environment.

### Shared Policy

In this project, all agents share the same DQN. This means they all use the same neural network to make decisions. This approach has several advantages:
- It is computationally efficient, as only one network needs to be trained (but later proved to be wrong, as compared to PPO it was exponentially slower).
- It encourages the agents to learn similar behaviors, which can lead to effective collaboration.
- It allows for a variable number of agents in the simulation without needing to retrain the model.

### Swarm Intelligence

Swarm intelligence is the collective behavior of decentralized, self-organized systems. This project simulates basic swarm behaviors, where simple rules and interactions between agents lead to intelligent collective behavior (in this case, efficient foraging).

## Simulation Environment (`core/` directory)

The simulation is built using Pygame.

- **Agents:** Each agent is an autonomous entity with a position, orientation, and speed. They can perform a set of discrete actions: rotate left, rotate right, accelerate, and decelerate.
- **Sensors:** Agents perceive their environment using a set of raycast sensors. These sensors detect the distance to obstacles (walls and other agents) in different directions. This information, along with the location of the nearest food and the nest, forms the input to the agent's neural network.
- **Food and Nest:** The environment is populated with food items. The agents' goal is to pick up food and carry it back to the nest.
- **Rewards:** The agents are trained using a reward system:
    - **Positive Rewards:** For picking up food and for successfully depositing it at the nest.
    - **Negative Rewards:** For colliding with walls or other agents. A small negative reward is also given for each time step to encourage efficiency.

## Reinforcement Learning Implementation (`rl/` directory)

The reinforcement learning part of the project is implemented using PyTorch.

- **DQN Architecture:** The `SharedDQN` in `rl/network.py` is a multi-layer perceptron (MLP) that takes the agent's sensor readings as input and outputs the Q-values for each possible action.
- **Training Process:** The training process is detailed in `rl/train.py`. It uses a replay buffer to store past experiences (state, action, reward, next_state). During training, batches of experiences are sampled from the replay buffer to update the DQN's weights. A target network is used to stabilize the training process.
- **Hyperparameters:** Key hyperparameters for training, such as the learning rate, replay buffer capacity, and epsilon decay rate for the epsilon-greedy exploration strategy, are defined in `rl/config.py`.

## How to Run the Project

- **`neuralswarm.py`:** Run this file to start the simulation. You can specify the number of agents and whether you want to control one of them manually.
- **`rl/train.py`:** Execute this script to train the DQN model. You can also resume training from a checkpoint.
- **`rl/play.py`:** Use this to watch a pre-trained model control the agents in the simulation.
- **`inference.py`:** This script allows you to generate a video of a trained model's performance.

## File Breakdown

- **`neuralswarm.py`**: Main entry point for the simulation.
- **`inference.py`**: Generates a video of a trained model.
- **`core/`**: Contains the core simulation logic.
    - **`environment.py`**: Manages the simulation, agents, and their interactions.
    - **`agent.py`**: Defines the agent's behavior, sensors, and actions.
    - **`entities.py`**: Defines the `Food` class.
    - **`config.py`**: Configuration for the simulation environment.
- **`rl/`**: Contains the reinforcement learning implementation.
    - **`train.py`**: The main script for training the DQN.
    - **`dqn_agent.py`**: Defines the agent that uses the DQN to select actions.
    - **`network.py`**: Defines the `SharedDQN` neural network architecture.
    - **`replay_buffer.py`**: Implements the replay buffer.
    - **`trainer.py`**: Handles the DQN training loop.
    - **`config.py`**: Hyperparameters for RL training.
    - **`play.py`**: Script to watch a trained model play.
