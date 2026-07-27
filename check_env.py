import numpy as np
from env.swarm_env import SwarmEnv
from env.wrappers import SwarmVecEnv
from stable_baselines3.common.env_checker import check_env

def test_raw_env():
    print("Testing Raw SwarmEnv...")
    env = SwarmEnv(num_agents=2)
    obs, _ = env.reset()
    print(f"Reset Obs Shape: {np.shape(obs)}")
    assert len(obs) == 2
    
    actions = [env.action_space.sample() for _ in range(2)]
    obs, rewards, term, trunc, info = env.step(actions)
    print(f"Step Rewards: {rewards}")
    print(f"Step Obs Shape: {np.shape(obs)}")
    env.close()
    print("Raw Env Test Passed!")

def test_vec_env():
    print("\nTesting SwarmVecEnv Wrapper...")
    # Create wrapped env
    env = SwarmVecEnv(lambda: SwarmEnv(num_agents=5))
    
    print(f"Num Envs: {env.num_envs}")
    obs = env.reset()
    print(f"VecEnv Reset Obs Shape: {obs.shape}")
    assert obs.shape == (5, env.observation_space.shape[0])
    
    actions = [env.action_space.sample() for _ in range(5)]
    obs, rewards, dones, infos = env.step(actions)
    
    print(f"VecEnv Step Rewards: {rewards}")
    print(f"VecEnv Step Dones: {dones}")
    print("VecEnv Test Passed!")
    env.close()

if __name__ == "__main__":
    test_raw_env()
    test_vec_env()
