import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from utils.linear_schedule import LinearSchedule
from buffers.replay_buffer import ReplayBuffer
from buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


def evaluate_agent(model, env):
    step_rewards = []
    obs = env.reset()[0]
    done = False
    truncated = False
    while not (done or truncated):
        obs = np.expand_dims(np.array(obs), axis=0)
        obs_tf = tf.constant(obs)
        action = model.step(obs_tf, stochastic=False)
        obs, rew, done, truncated, _ = env.step(int(action))
        step_rewards.append(rew)
    return np.sum(step_rewards)

def train(model, env, eval_env, prioritizer, episodes_print_freq, buffer_params, training_params, prioritized_replay=True):
    # Training params
    agent_train_freq = training_params["agent_train_freq"] # train the network every agent_train_freq timesteps
    total_timesteps = training_params["total_timesteps"]
    learning_starts = training_params["learning_starts"]
    batch_size = training_params["batch_size"]
    eval_freq = training_params["eval_freq"]
    buffer_prs_plot_freq = training_params["buffer_prs_plot_freq"]
    
    # Derived training params
    prioritizer_train_freq = 3 * agent_train_freq # train the prioritizer every prioritizer_train_freq agent updates
    target_network_update_freq = 500 * agent_train_freq # update the target network every 500 gradient updates
    
    # Evaluation by averaging mean episodes rewards
    episodes_returns_x = []
    episodes_returns_y = []
    
    # Evaluation by averaging k rollouts
    eval_returns_x = []
    eval_returns_y = []
    
    # For the analysis of priorities values during training
    priorities_history_x = []
    priorities_history_y = []

    # For the analysis of priorities stored in the buffer during training
    buffer_prs = []
    buffer_tms = []
    
    # Replay Buffer params
    buffer_size = buffer_params["buffer_size"]
    prioritized_replay_alpha = buffer_params["prioritized_replay_alpha"]
    prioritized_replay_beta0 = buffer_params["prioritized_replay_beta0"]
    exploration_fraction = buffer_params["exploration_fraction"]
    exploration_final_eps = buffer_params["exploration_final_eps"]

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        beta_schedule = LinearSchedule(total_timesteps,
                                       initial_value=prioritized_replay_beta0,
                                       final_value=1.0)
        
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_value=1.0,
                                 final_value=exploration_final_eps)

    episode_rewards = [0.0]
    obs = env.reset()[0]

    for t in range(1, total_timesteps+1):
        eps = exploration.value(t)
        
        # Convert observation to numpy array and add batch dimension
        # If the observation is of type lazyFrame (atari), the conversion unpack it
        obs_np = np.expand_dims(np.array(obs), axis=0)
        # Convert to tf tensors for using @tf.function
        obs_tf = tf.constant(obs_np)
        eps_tf = tf.constant(eps)
        action = model.step(obs_tf, eps=eps_tf)
        new_obs, rew, done, truncated, _ = env.step(int(action))
        # Store transition in the replay buffer.
        replay_buffer.store((obs, action, rew, new_obs, float(done)))
        obs = new_obs

        episode_rewards[-1] += rew

        if t <= learning_starts and prioritized_replay:
            # Actually only the RND prioritizer requires an initialization of the observation normalizer
            prioritizer.pre_train(obs_np)

        if t > learning_starts and t % agent_train_freq == 0:
            # Sample a batch of experiences from the replay buffer
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                experience = replay_buffer.sample(batch_size)
                (obses_t, actions, rewards, obses_tp1, dones) = experience
                weights, batch_idxes = np.ones_like(rewards), None
            
            # Convert to tf tensors for using @tf.function
            obses_t_tf, obses_tp1_tf = tf.constant(obses_t), tf.constant(obses_tp1)
            actions_tf, rewards_tf, dones_tf = tf.constant(actions), tf.constant(rewards), tf.constant(dones)
            weights_tf = tf.constant(weights)
            
            # Train the agent and obtain the TD errors
            td_errors = model.train(obses_t_tf, actions_tf, rewards_tf, obses_tp1_tf, dones_tf, weights_tf)

            if prioritized_replay:
                if t % prioritizer_train_freq == 0:
                    # Train the prioritizer
                    prioritizer.train_components(obses_t, actions, rewards, obses_tp1, dones)
                
                # Construct the dictionary with the parameters used by the prioritizer to compute the new priorities
                params_dict = {}
                params_dict["obses_t"] =  obses_t
                params_dict["actions"] =  actions
                params_dict["rewards"] =  rewards
                params_dict["obses_tp1"] =  obses_tp1
                params_dict["dones"] =  dones
                params_dict["td_errors"] =  td_errors
                
                # Compute new priorities
                new_priorities = prioritizer.compute_priorities(params_dict)
                # Update the priorities of the experiences in the replay buffer
                replay_buffer.update_priorities(batch_idxes, new_priorities)

                # Add max priority to history
                priorities_history_x.append(t)
                priorities_history_y.append(max(new_priorities))

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            model.update_target()

        # Logging
        if ((done or truncated) and episodes_print_freq is not None and 
            len(episode_rewards) % episodes_print_freq == 0 and t > learning_starts ):
            # Compute mean episode reward over the last "episodes_print_freq" episodes
            mean_ep_reward = round(np.mean(episode_rewards[-episodes_print_freq:]), 1)
            num_episodes = len(episode_rewards)
            log = [
                ["steps", t],
                ["episodes", num_episodes],
                ["mean episode reward", mean_ep_reward],
                ["% time spent exploring", int(100 * exploration.value(t))]
            ]
            if prioritized_replay and "new_priorities" in locals():
                log.extend(
                    [["last priorities (max)", max(new_priorities)],
                    ["last priorities (min)", min(new_priorities)],
                    ["last priorities (std)", np.std(new_priorities)]]
                )
            print(tabulate(log))
            episodes_returns_x.append(num_episodes)
            episodes_returns_y.append(mean_ep_reward)
        
        if prioritized_replay and t % buffer_prs_plot_freq == 0:
            # Save priorities of transitions actually stored in the buffer
            buffer_prs.append(replay_buffer.priorities_tolist())
            buffer_tms.append(t)
        
        if done or truncated:
            obs = env.reset()[0]
            episode_rewards.append(0.0)
        
        if t > learning_starts and t % eval_freq == 0:
            # Evaluate the agent on average return of k non-stochastic rollout
            rollout_reward = evaluate_agent(model, eval_env)
            eval_returns_x.append(t)
            eval_returns_y.append(rollout_reward)
            
            if episodes_print_freq is not None: # check if printing is disabled
                print("Evaluation at step "+str(t)+": "+str(rollout_reward))

    return (
        model,
        pd.DataFrame(data={"x":eval_returns_x,"y":eval_returns_y}),
        pd.DataFrame(data={"x":episodes_returns_x,"y":episodes_returns_y}),
        pd.DataFrame(data={"x":priorities_history_x,"y":priorities_history_y}),
        pd.DataFrame(data={"tms":buffer_tms,"prs":buffer_prs})
        )
