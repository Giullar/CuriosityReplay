import numpy as np
from prioritizers.prioritizer import Prioritizer

class TDPrioritizer(Prioritizer):
    def compute_priorities(self, params_dict):
        return np.abs(np.clip(params_dict["td_errors"], -1, 1))

    def train_components(self, obses_t, actions, rewards, obses_tp1, dones):
        pass