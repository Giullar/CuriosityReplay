import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.build_network import build_network
from prioritizers.prioritizer import Prioritizer
from utils.normalizer import AverageStdNormalizer


class RNDPrioritizer(Prioritizer):
    def __init__(self, state_shape, n_actions, conv, params_dict, priority_combinator=None):
        self.n_actions = n_actions
        self.target_output_dimensionality = n_actions # dimensionality of the output of the target/predictor networks
        self.conv = conv
        if self.conv:
            self.state_shape = (1, state_shape[1], state_shape[2])
        else:
            self.state_shape = state_shape

        self.error_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.net_optimizer = keras.optimizers.Adam(learning_rate=params_dict["learning_rate"])
        # Errors (priority) normalization
        self.priority_combinator = priority_combinator
        
        # Define observation normalizer
        self.obs_norm = AverageStdNormalizer(shape=self.state_shape, center=0, clip_range_inf=-5, clip_range_sup=5)

        # Define networks structure
        if self.conv:
            conv = [(16,8,4),(32,4,2)]
            hidden_dense = [256]
        else:
            conv = []
            hidden_dense = [128, 128] #[256, 256]
        self.target_net = build_network(input_shape=self.state_shape, n_outputs=self.target_output_dimensionality, conv=conv, hidden_dense=hidden_dense, final_layer="dense")
        self.predictor_net = build_network(input_shape=self.state_shape, n_outputs=self.target_output_dimensionality, conv=conv, hidden_dense=hidden_dense, final_layer="dense")
        
    def pre_train(self, obs):
        # Call this function some times before the training starts to initialize the obs normalizer
        if self.conv:
            obs = obs[:,[3],:,:] #[3] instead of just 3 to preserve dimensionality

        self.obs_norm.normalize(obs)


    def compute_priorities(self, params_dict):
        obses = params_dict["obses_tp1"]
        # if framestack, extract the only the last frame from each image, or do a max_pooling operation
        if self.conv:
            obses = obses[:,[3],:,:] #[3] instead of just 3 to preserve dimensionality
        
        obses_norm = self.obs_norm.normalize(obses, update=False)

        target_output = self.target_net(obses_norm)
        predictor_output = self.predictor_net(obses_norm)
        errors = self.error_function(target_output, predictor_output)

        if self.priority_combinator is not None:
            # return the combinated priorities
            tds = params_dict["td_errors"]
            rewards = params_dict["rewards"]
            priorities_comb = self.priority_combinator.combine_priorities(errors.numpy(), tds, rewards)
        else:
            priorities_comb = np.abs(errors)
        return priorities_comb


    def train_components(self, obses_t, actions, rewards, obses_tp1, dones):
        obses = obses_tp1
        # if framestack, extract the only the last frame from each image, or do a max_pooling operation
        if self.conv:
            obses = obses[:,[3],:,:] #[3] instead of just 3 to preserve dimensionality
        
        obses_norm = self.obs_norm.normalize(obses)

        target_output = self.target_net(obses_norm)
        with tf.GradientTape() as tape:
            predictor_output = self.predictor_net(obses_norm)
            # TOTAL LOSS
            loss = tf.reduce_mean(self.error_function(target_output, predictor_output))
        # Compute gradient
        grads = tape.gradient(target = loss, sources = self.predictor_net.trainable_variables)
        # Apply gradient
        self.net_optimizer.apply_gradients(zip(grads, self.predictor_net.trainable_variables))
    

