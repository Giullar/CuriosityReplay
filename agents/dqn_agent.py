import os
import numpy as np
import tensorflow as tf
from agents.agent import Agent
from utils.build_network import build_network

class DQNAgent(Agent):
    def __init__(self, observation_shape, num_actions, env_conv, lr, grad_clipping=None, gamma=1.0, double_q=True):
        super(DQNAgent, self).__init__()
        self.num_actions = num_actions
        self.gamma = gamma
        self.double_q = double_q
        self.grad_clipping = grad_clipping
        
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)

        if self.grad_clipping is not None:
            self.optimizer = tf.keras.optimizers.Adam(lr, clipvalue=grad_clipping)
        else:
            self.optimizer = tf.keras.optimizers.Adam(lr)
        
        if env_conv:
            conv = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
            hidden_dense = [256]
            final_layer="dueling"
        else:
            conv = []
            hidden_dense = [128, 128]
            final_layer="dueling"
        
        self.q_network = build_network(input_shape=observation_shape,
                                       n_outputs=num_actions,
                                       conv=conv,
                                       hidden_dense=hidden_dense,
                                       final_layer=final_layer)

        self.target_q_network = build_network(input_shape=observation_shape,
                                       n_outputs=num_actions,
                                       conv=conv,
                                       hidden_dense=hidden_dense,
                                       final_layer=final_layer)
                
        self.update_target()

    @tf.function
    def step(self, obs, stochastic=True, eps=0):
        if stochastic and tf.random.uniform(shape=[], minval=0, maxval=1) < eps:
            # Random action
            output_action = tf.random.uniform(shape=[], minval=0, maxval=self.num_actions, dtype=tf.int64)
        else:
            # Deterministic action
            q_values = self.q_network(obs)[0]
            output_action = tf.math.argmax(q_values)
            
        return output_action
    
    @tf.function
    def train(self, states, actions, rewards, next_states, dones, importance_weights):        
        # Compute q(S_t+1, a) for each possible action in the next state S_t+1
        next_q_values = self.q_network(next_states)
        
        if self.double_q:
            # Take the actions that maximize q(S_t+1, a)
            max_next_actions = tf.math.argmax(next_q_values, axis=1)
            # mask which represents in a one-hot encoding which actions has been chosen in the state S_t+1 (by the online net)
            mask_stp1 = tf.one_hot(max_next_actions, self.num_actions)
            # Compute q_target(S_t+1, a) using the target network for each possible action
            next_q_values_tnet = self.target_q_network(next_states)
            # Take only the q_target(S_t+1, a) for the action "a" chosen by the online q_net
            max_next_q_values = tf.reduce_sum(next_q_values_tnet*mask_stp1, axis=1, keepdims=False)
        else:
            # compute max_a q(S_t+1, a, w_t)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            
        # Compute target q-values
        # R_t+1 + discount_factor * max_a q(S_t+1, a, w_t) if done == 0, else R_t+1
        weighted_max_next_q_values = (1-dones) * self.gamma * max_next_q_values
        target_q_values = (rewards + weighted_max_next_q_values)
            
        # mask which represents in a one-hot encoding which actions has been chosen in the state S_t
        mask_st = tf.one_hot(actions, self.num_actions)
        
        with tf.GradientTape() as tape:
            # Compute q values for actions in state S_t
            all_q_values = self.q_network(states) # shape (32, n_actions)
            # set to zero the q-values of the actions different from the one selected in state S_t
            all_q_values_masked = all_q_values * mask_st # shape (32, n_actions)
            # Reduce dimension of the tensor.
            # Now we have a tensor where for each state S_t we have only The q-value of the selected move.
            q_values = tf.reduce_sum(all_q_values_masked, axis=1, keepdims=False) # shape (32, 1)
            # Compute TDs (temporal differences) errors. (they will be returned by the train)
            td_errors = q_values - target_q_values
            # Compute loss function error weighted by the importance sampling weights
            weighted_loss_fn_error = importance_weights * self.loss_fn(target_q_values, q_values)
            # Compute final loss
            loss = tf.reduce_mean(weighted_loss_fn_error)
        # Compute gradient
        grads = tape.gradient(loss, self.q_network.trainable_variables)

        # Apply gradient
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        return td_errors
    
    @tf.function(autograph=False)
    def update_target(self):
        q_vars = self.q_network.trainable_variables
        target_q_vars = self.target_q_network.trainable_variables
        for var, var_target in zip(q_vars, target_q_vars):
            var_target.assign(var)


