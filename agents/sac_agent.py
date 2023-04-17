import os
import numpy as np
import tensorflow as tf
from agents.agent import Agent
from utils.build_network import build_network


class SACAgent(Agent):
    def __init__(self, observation_shape, num_actions, env_conv, lr_critic, lr_actor, lr_alpha, gamma=1.0):
        super(SACAgent, self).__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.eps = 1e-8 #To avoid computing the logarithm of zero
        self.alpha_initial = 1.

        self.error_function_critic = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic, clipvalue=0.9)
        self.critic_optimizer2 = tf.keras.optimizers.Adam(lr_critic, clipvalue=0.9)
        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor, clipvalue=0.9)
        self.alpha_optimiser = tf.keras.optimizers.Adam(lr_alpha, clipvalue=0.9)
        
        if env_conv:
            conv = [(16,8,4),(32,4,2)]
            hidden_dense = [256]
            final_layer="dueling"
        else:
            conv = []
            hidden_dense = [128, 128]
            final_layer="dueling"

        self.critic_local = build_network(input_shape=observation_shape,
                                           n_outputs=num_actions,
                                           conv=conv,
                                           hidden_dense=hidden_dense,
                                           final_layer=final_layer)

        self.critic_local2 = build_network(input_shape=observation_shape,
                                           n_outputs=num_actions,
                                           conv=conv,
                                           hidden_dense=hidden_dense,
                                           final_layer=final_layer)

        self.critic_target = build_network(input_shape=observation_shape,
                                           n_outputs=num_actions,
                                           conv=conv,
                                           hidden_dense=hidden_dense,
                                           final_layer=final_layer)

        self.critic_target2 = build_network(input_shape=observation_shape,
                                           n_outputs=num_actions,
                                           conv=conv,
                                           hidden_dense=hidden_dense,
                                           final_layer=final_layer)

        self.update_target()

        self.actor = build_network(input_shape=observation_shape, 
                                    n_outputs=num_actions,
                                    conv=conv,
                                    hidden_dense=hidden_dense,
                                    final_layer="dense_softmax")

        # Set target entropy and temperature (alpha)
        self.target_entropy = 0.98 * -np.log(1 / num_actions)
        self.log_alpha = tf.Variable(np.log(self.alpha_initial), dtype=tf.float32)
        self.alpha = self.log_alpha
    
    @tf.function
    def step(self, obs, stochastic=True, eps=-1):
        action_probs = self.actor(obs)[0]
        if stochastic:
            # Sample action using actor policy
            output_action = tf.random.categorical([tf.math.log(action_probs + self.eps)], 1)[0][0]   
        else:
            # Choose action with maximum probability in actor policy
            output_action = tf.math.argmax(action_probs)
        return output_action
    
    @tf.function
    def compute_action_probs(self, next_states):
        # Compute pi(a|S_t+1) for each possibile action
        action_probs = self.actor(next_states)
        # Compute the logarithm of actions probabilities
        # Sum eps to each action prob to avoid computing the logarithm of zero
        log_action_probs = tf.math.log(action_probs + self.eps)
        return action_probs, log_action_probs

    @tf.function
    def compute_critic_loss(self, states, actions, rewards, next_states, dones, importance_weights):
        # Compute pi(a|S_t+1) for each possibile action
        action_probs, log_action_probs = self.compute_action_probs(next_states)

        # Compute q(S_t+1, a) using the target networks for each possible action
        next_q_values_target = self.critic_target(next_states)
        next_q_values_target2 = self.critic_target2(next_states)

        # Compute targets y(r,s',d) (weighted average (expectation) over possible actions)
        min_diff = tf.reduce_sum(action_probs * (tf.minimum(next_q_values_target, next_q_values_target2) - self.alpha * log_action_probs), axis=1)
        ys = rewards + (1-dones) * self.gamma * min_diff

        # Prepare to compute TDs
        # mask which represents in a one-hot encoding which actions has been chosen in the state S_t
        mask_t = tf.one_hot(actions, self.num_actions)
        # Critic 1
        all_q_values_local = self.critic_local(states)
        # For each transition in the batch take only the q-value of the move selected in the actions tensor
        all_q_values_masked = all_q_values_local * mask_t # shape (32, n_actions)
        selected_q_values_local = tf.reduce_sum(all_q_values_masked, axis=1, keepdims=False) # shape (32, 1)
        # Critic 2
        all_q_values_local2 = self.critic_local2(states)
        # For each transition in the batch take only the q-value of the move selected in the actions tensor
        all_q_values_masked2 = all_q_values_local2 * mask_t # shape (32, n_actions)
        selected_q_values_local2 = tf.reduce_sum(all_q_values_masked2, axis=1, keepdims=False) # shape (32, 1)
        # Compute TDs
        tds_local = self.error_function_critic(selected_q_values_local, ys) * importance_weights
        tds_local2 = self.error_function_critic(selected_q_values_local2, ys) * importance_weights
        # Compute average TDs to be used by the prioritized replay buffer (you can also use the minimum between the two)
        tds = (tds_local + tds_local2) / 2
        return tds_local, tds_local2, tds

    @tf.function
    def compute_actor_loss(self, states, importance_weights):
        # Compute pi(a|S_t) for each possibile action
        action_probs, log_action_probs = self.compute_action_probs(states)

        # Compute q(S_t, a) using the local networks for each possible action
        q_values_local = self.critic_local(states)
        q_values_local2 = self.critic_local2(states)

        # Actor loss wants to maximise at the same time the q_values and the entropy in the states
        inside_term = self.alpha * log_action_probs - tf.minimum(q_values_local, q_values_local2)
        # Compute Expectation weighted by actions probabilities
        actor_loss = tf.reduce_mean(tf.reduce_sum((action_probs * inside_term), axis=1) * importance_weights)
        return actor_loss, log_action_probs

    @tf.function
    def compute_temperature_loss(self, log_action_probs):
        #alpha_loss = -tf.reduce_mean(self.log_alpha * (log_action_probs + self.target_entropy))
        alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_action_probs + self.target_entropy))
        return alpha_loss
    
    @tf.function
    def train(self, states, actions, rewards, next_states, dones, importance_weights): 
        #  UPDATE CRITIC
        with tf.GradientTape(persistent=True) as tape:
            # Compute critic loss
            tds_local, tds_local2, tds = self.compute_critic_loss(states, actions, rewards, next_states, dones, importance_weights)
        # Compute gradients
        critic_grads = tape.gradient(tds_local, self.critic_local.trainable_variables)
        critic_grads2 = tape.gradient(tds_local2, self.critic_local2.trainable_variables)
        # Apply gradients
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_local.trainable_variables))
        self.critic_optimizer2.apply_gradients(zip(critic_grads2, self.critic_local2.trainable_variables))
        del tape
        
        # UPDATE ACTOR
        with tf.GradientTape() as tape:
            # Compute actor loss
            actor_loss, log_action_probs = self.compute_actor_loss(states, importance_weights)
        # Compute gradients
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # UPDATE ALPHA
        with tf.GradientTape() as tape:
            # Compute temperature loss
            alpha_loss = self.compute_temperature_loss(log_action_probs)
        # Compute gradients
        alpha_grads = tape.gradient(alpha_loss, self.log_alpha)
        # Apply gradients
        self.alpha_optimiser.apply_gradients(zip([alpha_grads], [self.log_alpha]))
        # Update alpha from log_alpha
        self.alpha = tf.math.exp(self.log_alpha)
        
        return tds
    
    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.critic_local.trainable_variables
        target_q_vars = self.critic_target.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)

        q_vars = self.critic_local2.trainable_variables
        target_q_vars = self.critic_target2.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)


