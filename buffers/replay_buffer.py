import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [None] * capacity
        self.next_idx = 0
        self.n_items = 0
    
    def store(self, experience):
        self.data[self.next_idx] = experience
        self.next_idx += 1
        
        if self.next_idx >= self.capacity:
            self.next_idx = 0
        
        if self.n_items < self.capacity:
            self.n_items += 1
        
    def sample(self, batch_size):
        # Sample indices of experiences
        indices = np.random.randint(self.n_items, size=batch_size)
        
        states, actions, next_states = [], [], []
        rewards = np.empty((batch_size,), dtype=np.float32)
        dones = np.empty((batch_size,), dtype=np.float32)
        
        for i, index in enumerate(indices):
            batch = self.data[index]
            states.append(batch[0])
            actions.append(batch[1])
            rewards[i] = batch[2]
            next_states.append(batch[3])
            dones[i] = batch[4]
            
        return (np.array(states, copy=False),
                np.array(actions, copy=False),
                rewards,
                np.array(next_states, copy=False),
                dones)

