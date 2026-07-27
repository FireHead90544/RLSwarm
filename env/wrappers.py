import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces

class SwarmVecEnv(VecEnv):
    """
    A custom VecEnv that wraps a SINGLE SwarmEnv instance but presents it
    as `num_agents` independent environments to Stable Baselines3.
    This allows PPO to control all agents with a shared policy.
    """
    def __init__(self, env_fn):
        # Create the single physics environment
        self.swarm_env = env_fn()
        self.num_agents = self.swarm_env.num_agents
        
        # Define spaces (must be same for all agents)
        observation_space = self.swarm_env.observation_space
        action_space = self.swarm_env.action_space
        
        super().__init__(self.num_agents, observation_space, action_space)
        
        self.actions = None
        self.metadata = self.swarm_env.metadata

    def reset(self):
        """
        Reset the world. Returns observations for ALL agents.
        """
        obs_list, _ = self.swarm_env.reset()
        return np.stack(obs_list)

    def step_async(self, actions):
        """
        Store actions to be executed in step_wait.
        """
        self.actions = actions

    def step_wait(self):
        """
        Execute the stored actions in the single physics world.
        """
        obs_list, rewards, terminated, truncated, infos = self.swarm_env.step(self.actions)
        
        # Convert to numpy arrays
        obs = np.stack(obs_list)
        rews = np.array(rewards)
        dones = np.array([terminated or truncated] * self.num_agents) # If env ends, all agents end
        
        # Infos: List of dicts
        # If done, SB3 expects 'terminal_observation' in info
        infos_list = [infos.copy() for _ in range(self.num_agents)]
        
        if dones[0]:
            # If episode ended, we need to reset immediately to continue training?
            # SB3 VecEnv usually auto-resets.
            # But here, if we reset, we get new obs.
            # Standard VecEnv behavior: if done, reset automatically and return NEW obs,
            # but store OLD obs in info['terminal_observation'].
            
            new_obs_list, _ = self.swarm_env.reset()
            new_obs = np.stack(new_obs_list)
            
            for i in range(self.num_agents):
                infos_list[i]['terminal_observation'] = obs[i]
            
            obs = new_obs
            
        return obs, rews, dones, infos_list

    def close(self):
        self.swarm_env.close()

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from underlying env.
        Since we have 1 env, we just return the same value for all indices.
        """
        val = getattr(self.swarm_env, attr_name)
        if indices is None:
            indices = range(self.num_agents)
        return [val for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.swarm_env, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        method = getattr(self.swarm_env, method_name)
        return [method(*method_args, **method_kwargs) for _ in range(self.num_agents)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_agents
