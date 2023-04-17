import numpy as np
from utils.sum_tree import SumTree


class PrioritizedReplayBuffer(object):
    P_EPS = 1e-6  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken

    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.tree = SumTree(self.capacity)
        self.max_priority = PrioritizedReplayBuffer.P_EPS #1.0
        
    # Define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # NOTE: Pay attention if priorities may take arbitrarily large values!!!!!!!
        self.tree.add(self.max_priority, experience)   # set the max priority for new priority
        
    # sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n, beta):
        #minibatch = []
        max_imp_sampl_weight = -1
        
        # We will use lists for data for which the type can change with the environment
        # Otherwise we could also dynamically extract the data types at runtime.
        obses_t, actions, obses_tp1 = [], [], []
        rewards = np.empty((n,), dtype=np.float32)
        dones = np.empty((n,), dtype=np.float32)
        
        batch_idxes = np.empty((n,), dtype=np.int32)
        weights = np.empty((n,), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            priority = np.random.uniform(a, b)
            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(priority)
            
            # Compute importance sampling weights
            w = (self.capacity * priority) ** (-beta)
            if w > max_imp_sampl_weight:
                max_imp_sampl_weight = w
            
            weights[i] = w
            batch_idxes[i] = index
            obses_t.append(data[0])
            actions.append(data[1])
            rewards[i] = data[2]
            obses_tp1.append(data[3])
            dones[i] = data[4]
        
        # Normalize importance sampling weights by dividing them by the maximum weight
        weights = np.divide(weights, max_imp_sampl_weight)
        
        #Note: if an observation is of type lazyFrame (atari), the conversion to numpy array unpack it
        return (np.array(obses_t),
                np.array(actions),
                rewards,
                np.array(obses_tp1),
                dones,
                weights,
                batch_idxes)
    
    # Update the priorities on the tree
    def update_priorities(self, tree_idx, priorities):
        priorities += PrioritizedReplayBuffer.P_EPS
        priorities = np.power(priorities, self.alpha)

        for ti, p in zip(tree_idx, priorities):
            self.tree.update(ti, p)
            self.max_priority = max(self.max_priority, p)
    
    def priorities_tolist(self):
        return self.tree.priorities_tolist()
