import numpy as np
from utils.normalizer import AverageMeanNormalizer, AverageStdNormalizer

# TD-LIKE NORMALIZATION FOR ALL MODELS
class PriorityCombinator:
    def __init__(self, w_ic = 0.3, w_td = 0.6, w_rw = 0.1):
        # Priority Combination Weights
        self.w_ic = w_ic
        self.w_td = w_td
        self.w_rw = w_rw
    
    def combine_priorities(self, ic_priorities, td_errors, rewards):
        ic_priorities_norm = np.abs(np.clip(ic_priorities, -1, 1))
        td_priorities_norm = np.abs(np.clip(td_errors, -1, 1))
        rw_priorities_norm = np.clip(rewards, 0, 1)
        priorities_comb = (ic_priorities_norm * self.w_ic +
                           td_priorities_norm * self.w_td +
                           rw_priorities_norm * self.w_rw)
        return priorities_comb
