from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow import keras
from prioritizers.prioritizer import Prioritizer

class UniformNoiseLayer(keras.layers.Layer):
    def __init__(self, minval, maxval):
        super(UniformNoiseLayer, self).__init__()
        self.minval = minval
        self.maxval = maxval
    
    def call(self, inputs):
        return inputs + tf.random.uniform(shape=tf.shape(inputs), minval=self.minval, maxval=self.maxval)
        

class AESHPrioritizer(Prioritizer): # AutoEncoder SimHash Prioritizer
    def __init__(self, state_shape, n_actions, conv, params_dict, priority_combinator=None):
        self.n_actions = n_actions
        self.conv = conv
        self.k = params_dict["k"] # Length of the binary code
        self.encoder_output_size = params_dict["encoder_output_size"]
        self.lambda_regularizer = params_dict["lambda_regularizer"]
        self.beta = params_dict["beta"]
        if self.conv:
            self.state_shape = (1, state_shape[1], state_shape[2])
            # pixels intensities are in [0,1]
            self.autoencoder_reconstr_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            self.net_optimizer = keras.optimizers.Adam(learning_rate=params_dict["learning_rate"])
            # Define the AE: encoder and decoder
            # Note: if we are working with obs with framestack, we will use only the last frame, and not the full stack
            self.encoder, self.decoder = self.__build_autoencoder()
            # Generate a SimHash matrix
            self.simhash_matrix = np.random.standard_normal(size=(self.k, self.encoder_output_size))
        else:
            self.state_shape = state_shape
            self.autoencoder_reconstr_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
            # Generate a SimHash matrix
            self.simhash_matrix = np.random.standard_normal(size=(self.k, self.state_shape[0]))
        
        # Errors (priority) normalization
        self.priority_combinator = priority_combinator
        # Create the count table
        self.count_table = defaultdict(int)
        
    def __build_autoencoder(self):
        w_initializer = "glorot_uniform"
        b_initializer = "zeros"
        # ENCODER DEFINITION
        inputs_encoder = x = keras.Input(shape=self.state_shape)
        x = keras.layers.Conv2D(filters=96, kernel_size=6, strides=2, activation='relu', padding='valid',
                                data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2D(filters=96, kernel_size=6, strides=2, activation='relu', padding='valid',
                                data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2D(filters=96, kernel_size=4, strides=2, activation='relu', padding='valid',
                                data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2D(filters=96, kernel_size=4, strides=1, activation='relu', padding='valid',
                                data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(1024, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Dense(self.encoder_output_size, activation='sigmoid', kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        outputs_encoder = x

        # DECODER DEFINITION
        inputs_decoder = x = keras.Input(shape=(self.encoder_output_size,))
        x = UniformNoiseLayer(minval=-0.3, maxval=0.3)(x)
        x = keras.layers.Dense(2400, activation='tanh', kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        #x = keras.layers.Reshape(target_shape=(5,5,96))(x) #data_format=channels_last (default)
        x = keras.layers.Reshape(target_shape=(96,5,5))(x) #data_format=channels_first
        x = keras.layers.Conv2DTranspose(filters=96, kernel_size=4, strides=1, activation='relu', padding='valid',
                                         data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2DTranspose(filters=96, kernel_size=4, strides=2, activation='relu', padding='valid',
                                         data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2DTranspose(filters=96, kernel_size=6, strides=2, activation='relu', padding='valid',
                                         data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2DTranspose(filters=96, kernel_size=6, strides=2, activation='relu', padding='valid',
                                         data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        x = keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same',
                                data_format="channels_first", kernel_initializer=w_initializer, bias_initializer=b_initializer)(x)
        outputs_decoder = x

        encoder = tf.keras.Model(inputs=inputs_encoder, outputs=outputs_encoder)
        decoder = tf.keras.Model(inputs=inputs_decoder, outputs=outputs_decoder)

        return encoder, decoder
    
    def __compute_hash(self, state_embs):
        # Theoretically we should multiply the simhash_matrix by the transpose of the state_emb batch, but
        # if we multiply the state_emb batch by the transpose of the simhash_matrix we obtain the same result
        # but transposed. This is useful because in this way we have a resulting matrix in which the row[i] is the
        # encoding of the state_emb[i].
        return np.sign(np.matmul(state_embs, self.simhash_matrix.T))

    def compute_priorities(self, params_dict):
        frames = params_dict["obses_tp1"]
        
        # if framestack, extract the only the last frame from each image, or do a max_pooling operation
        if self.conv:
            frames = frames[:,[3],:,:] #[3] instead of just 3 to preserve dimensionality
            # Apply AE (only encoder part) and SimHash to map the observations to their binary codes
            state_embs = self.encoder(frames)
            state_embs = tf.round(state_embs) # Round values to 0/1
            hash_codes = self.__compute_hash(state_embs)
        else:
            hash_codes = self.__compute_hash(frames)
        
        # Compute curiosity scores
        curiosity_scores = np.empty((frames.shape[0],), dtype=np.float32)
        for i, hash_code in enumerate(hash_codes):
            # Use the codes to retrieve the associated counts from the count table
            # Also, increment in the count table the counts associated to those observations
            c = self.count_table[str(hash_code)] = self.count_table[str(hash_code)] + 1
            # Compute the priorities using the updated counts with the formula and return them
            curiosity_scores[i] = self.beta/np.sqrt(c)
        
        if self.priority_combinator is not None:
            # return the combinated priorities
            tds = params_dict["td_errors"]
            rewards = params_dict["rewards"]
            priorities_comb = self.priority_combinator.combine_priorities(curiosity_scores, tds, rewards)
        else:
            priorities_comb = np.abs(curiosity_scores)
        return priorities_comb

    def train_components(self, obses_t, actions, rewards, obses_tp1, dones): # Only trains the AE
        if self.conv:
            batch_size = obses_tp1.shape[0]
            frames = obses_tp1
            # if framestack, extract the only the last frame from each image, or do a max_pooling operation
            frames = frames[:,[3],:,:] #[3] instead of just 3 to preserve dimensionality
            # Train the AE using also the decoder part
            with tf.GradientTape() as tape:
                frames_encoded = self.encoder(frames)
                frames_decoded = self.decoder(frames_encoded)
                reconstr_loss = self.autoencoder_reconstr_loss(frames, frames_decoded)
                min_term_1 = tf.pow(tf.subtract(1, frames_encoded), 2)
                min_term_2 = tf.pow(frames_encoded, 2)
                min_complete_reduced = tf.reduce_sum(tf.math.minimum(min_term_1, min_term_2))
                min_weighted = (1/batch_size) * (self.lambda_regularizer/self.k) * min_complete_reduced
                # TOTAL LOSS 
                loss = reconstr_loss + min_weighted
                
            # Compute gradient
            grads = tape.gradient(target = loss, sources = self.encoder.trainable_variables +
                                  self.decoder.trainable_variables)
            # Apply gradient
            self.net_optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables + 
                                                   self.decoder.trainable_variables))
        
