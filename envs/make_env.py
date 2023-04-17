import gym
from gym.wrappers import AtariPreprocessing, FrameStack, LazyFrames
from envs.wrappers import ClipReward
from constants import ATARI_SCREEN_SIZE

def make_env(env_name, env_conv, clip_reward=False, max_episode_steps=None, seed=None, render_mode=None):
    if env_conv:
        env = gym.make(env_name,
                       frameskip=1,
                       max_episode_steps=max_episode_steps,
                       full_action_space=False,
                       repeat_action_probability=0.0,
                       render_mode=render_mode)
        
        # Note:
        # scale_obs (bool):
        # if True, then observation normalized in range [0,1) is returned.
        # It also limits memory optimization benefits of FrameStack Wrapper.
        env = AtariPreprocessing(env, frame_skip = 4, screen_size=ATARI_SCREEN_SIZE, scale_obs=True)
        env = FrameStack(env, 4)
    else:
        env = gym.make(env_name, max_episode_steps=max_episode_steps, render_mode=render_mode)
    
    if clip_reward:
        env = ClipReward(env, min_reward=-1, max_reward=1)
    
    if seed is not None:
        env.reset(seed=seed)

    return env


