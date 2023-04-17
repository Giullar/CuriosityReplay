import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.build_network import build_network
from prioritizers.prioritizer import Prioritizer


class ICMPrioritizer(Prioritizer):
    def __init__(self, state_shape, n_actions, conv, params_dict, priority_combinator=None):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.conv = conv
        self.embedding_size = params_dict["embedding_size"]
        self.beta = params_dict["beta"]
        self.error_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.inv_dyn_loss_fun = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.net_optimizer = keras.optimizers.Adam(learning_rate=params_dict["learning_rate"])
        # Errors (priority) normalization
        self.priority_combinator = priority_combinator
        
        if self.conv:
            hidden_dense = [256]
            self.embedding_net = build_network(input_shape=self.state_shape, n_outputs=None,
                                               conv=[(16,8,4),(32,4,2),(32,4,2),(32,4,2)],
                                               hidden_dense=[], final_layer=None)

            self.inv_dyn_net = build_network(input_shape=(self.embedding_size + self.embedding_size,),
                                         n_outputs=self.n_actions, conv=[], hidden_dense=hidden_dense, final_layer="dense_softmax")
        
            self.forward_net = build_network(input_shape=(self.embedding_size + self.n_actions,),
                                             n_outputs=self.embedding_size, conv=[], hidden_dense=hidden_dense, final_layer="dense")  
        else:
            hidden_dense = [128]
            self.embedding_net = build_network(input_shape=self.state_shape, n_outputs=self.embedding_size,
                                               conv=[],
                                               hidden_dense=[self.embedding_size], final_layer="dense")

            self.inv_dyn_net = build_network(input_shape=(self.embedding_size + self.embedding_size,),
                                         n_outputs=self.n_actions, conv=[], hidden_dense=hidden_dense, final_layer="dense_softmax")
        
            self.forward_net = build_network(input_shape=(self.embedding_size + self.n_actions,),
                                             n_outputs=self.embedding_size, conv=[], hidden_dense=hidden_dense, final_layer="dense")  
            
          
        
    def compute_priorities(self, params_dict):
        obses_t = params_dict["obses_t"]
        actions = params_dict["actions"]
        obses_tp1 = params_dict["obses_tp1"]
        # one-hot encode the actions
        actions_one_hot = tf.one_hot(actions, depth=self.n_actions)
        # Embed the states s_t and s_t+1
        obses_t_emb = self.embedding_net(obses_t)
        obses_tp1_emb_true = self.embedding_net(obses_tp1)
        # Concatenate the embedding of s_t with the actions
        states_and_actions = tf.concat([obses_t_emb, actions_one_hot], axis=1)
        # Predict the embedding of s_t+1 using the forward net
        obses_tp1_emb_pred = self.forward_net(states_and_actions)
        errors = 1/2 * self.error_function(obses_tp1_emb_true, obses_tp1_emb_pred)

        if self.priority_combinator is not None:
            # return the combinated priorities
            tds = params_dict["td_errors"]
            rewards = params_dict["rewards"]
            priorities_comb = self.priority_combinator.combine_priorities(errors.numpy(), tds, rewards)
        else:
            priorities_comb = np.abs(errors)
        return priorities_comb

    def train_components(self, obses_t, actions, rewards, obses_tp1, dones):
        # one-hot encode the actions
        actions_one_hot = tf.one_hot(actions, depth=self.n_actions)
        with tf.GradientTape() as tape:
            # Embed the states s_t and s_t+1
            obses_t_emb = self.embedding_net(obses_t)
            obses_tp1_emb = self.embedding_net(obses_tp1)
            # COMPUTE INVERSE DYNAMICS LOSS
            # Concatenate the embedding of s_t with the embedding of s_t+1
            states_emb = tf.concat([obses_t_emb, obses_tp1_emb], axis=1)
            # Compute the predicted action a_t given the embeddings of s_t and s_t+1 
            actions_pred = self.inv_dyn_net(states_emb)
            loss_inv_dyn = self.inv_dyn_loss_fun(actions_one_hot, actions_pred)
            # COMPUTE FORWARD MODEL LOSS
            # Concatenate the embedding of s_t with the actions
            states_and_actions = tf.concat([obses_t_emb, actions_one_hot], axis=1)
            # Use the forward model to predict the embedding of s_t+1 given s_t and a_t
            obses_tp1_emb_pred = self.forward_net(states_and_actions)
            loss_forward_model = tf.reduce_mean(self.error_function(obses_tp1_emb, obses_tp1_emb_pred))
            # TOTAL LOSS
            loss = (1-self.beta)*loss_inv_dyn + self.beta*loss_forward_model
        # Compute gradient
        grads = tape.gradient(target = loss,
                              sources = self.embedding_net.trainable_variables +
                              self.inv_dyn_net.trainable_variables +
                              self.forward_net.trainable_variables)
                              
        # Apply gradient
        self.net_optimizer.apply_gradients(zip(grads,
                                      self.embedding_net.trainable_variables +
                                      self.inv_dyn_net.trainable_variables +
                                      self.forward_net.trainable_variables))