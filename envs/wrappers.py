import numpy as np
import gym
from gym.spaces import Box

class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)
    
    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


# Normalize observations in [min_value, max_value]
class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, min_value=-1, max_value=1):
        super().__init__(env)
        self.original_min_value = env.observation_space.low
        self.original_max_value = env.observation_space.high
        self.min_value = min_value
        self.max_value = max_value
        self.observation_space = Box(shape=env.observation_space.shape, low=self.min_value, high=self.max_value)

    def observation(self, obs):
        if type(obs) is tuple: # fix a problem with Box2D envs (is like an "if(we are in a box2d env)")
            obs = obs[0]
        return (self.max_value-self.min_value) * ((obs-self.original_min_value)/(self.original_max_value-self.original_min_value)) + self.min_value
