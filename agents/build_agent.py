from agents.dqn_agent import DQNAgent
from agents.sac_agent import SACAgent

def build_agent(name, env, env_conv, agent_params):
    if name == "DQN":
        lr = agent_params["lr"]
        gamma = agent_params["gamma"]
        grad_clipping = agent_params["grad_clipping"]

        agent = DQNAgent(
            observation_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            env_conv=env_conv,
            lr=lr,
            grad_clipping=grad_clipping,
            gamma=gamma
        )
    elif name == "SAC":
        lr_critic = agent_params["lr_critic"]
        lr_actor = agent_params["lr_actor"]
        lr_alpha = agent_params["lr_alpha"]
        gamma = agent_params["gamma"]

        agent = SACAgent(
            observation_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            env_conv=env_conv,
            lr_critic=lr_critic,
            lr_actor=lr_actor,
            lr_alpha=lr_alpha,
            gamma=gamma
        )
    else:
        raise Exception("Incorrect agent name")
    
    return agent