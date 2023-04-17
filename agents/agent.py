import os
import tensorflow as tf

class Agent(tf.keras.Model):
    def __init__(self):
        super(Agent, self).__init__()

    def step(self, obs, stochastic=True, eps=0):
        pass
    
    def train(self, states, actions, rewards, next_states, dones, importance_weights):
        pass

    def update_target(self):
        pass

    def save_model(self, dir, suffix=""):
        pass
        
    def load_model(self, dir, suffix=""):
        pass

    def save_model(self, dir, model_name):
        checkpoint = tf.train.Checkpoint(root=self)
        # Save a checkpoint to dir/model_name-{save_counter}. 
        # Every time checkpoint.save is called, the save counter is increased.
        #checkpoint.save(os.path.join(dir, model_name))
        checkpoint.write(os.path.join(dir, model_name))

    def load_model(self, dir, model_name):
        checkpoint = tf.train.Checkpoint(root=self)
        # Restore the checkpointed values to the `model` object.
        checkpoint.read(os.path.join(dir, model_name)).assert_existing_objects_matched()