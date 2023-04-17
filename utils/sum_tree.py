import numpy as np


class SumTree(object):
    data_pointer = 0
    
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        # we initialize the tree with all nodes = 0, and initialize the data with all values = 0
        self.tree = np.zeros(2 * capacity - 1)        
        #self.data = np.zeros(capacity, dtype=object) # Contains the experiences (so the size of data is capacity)
        self.data = [None] * capacity # Contains the experiences (so the size of data is capacity)
    
    # Add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0
            
    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        
    # Get a leaf from our tree.
    # Returns the leaf_index, priority value of that leaf and experience associated with that leaf index.
    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else: # downward search
                if v > self.tree[left_child_index]: # go right
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
                else: # go left
                    parent_index = left_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
    
    def priorities_tolist(self):
        return self.tree[-self.capacity:].copy()
