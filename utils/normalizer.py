import numpy as np


# These classes can be used with both array of scalars and array of observations (nd array).
# In the case of observations (nd array), the normalization is computed for each pixel indipendenly.

########## DIVIDE BY MEAN NORMALIZATION. USED IN PRIORITY COMBINATION:

class RunningMean:
    def __init__(self, shape=(), epsilon=1e-2, clip_range_inf=np.NINF, clip_range_sup=np.inf):
        self.n_a = 1
        self.avg_a = np.zeros(shape, "float64") # mean(values)
        self.epsilon = epsilon
        self.clip_range_sup = clip_range_sup
        self.clip_range_inf = clip_range_inf
        
    def update(self, n_b, avg_b):
        n = self.n_a + n_b
        self.avg_a = (self.avg_a * self.n_a + avg_b * n_b)/n
        self.n_a = n

    def normalize(self, v):
        v_norm = np.clip(np.divide(v, self.avg_a), self.clip_range_inf, self.clip_range_sup)
        return v_norm


class AverageMeanNormalizer:
    def __init__(self, shape=(), clip_range_sup=np.inf, clip_range_inf=np.NINF):
        self.rm = RunningMean(shape=shape, clip_range_inf=clip_range_inf, clip_range_sup=clip_range_sup)
    
    def normalize(self, obs_batch, update=True):
        # accetta come parametro un batch di obs (frames).
        # restituisce la sua versione normalizzata pixel per pixel
        if update:
            batch_size = obs_batch.shape[0]
            mean = np.mean(obs_batch, axis=0) # compute the mean over the batch dimension (has the shape of an obs)
            self.rm.update(batch_size, mean)
        return self.rm.normalize(obs_batch)

    
########## (VALUE-MEAN)/STD NORMALIZATION (STANDARDIZATION). USED IN RND PRIORITIZER APPLIED TO OBSERVATIONS:

class RunningStd:
    def __init__(self, shape=(), epsilon=1e-2, center=0, clip_range_inf=np.NINF, clip_range_sup=np.inf):
        self.n_a = 1
        self.avg_a = np.zeros(shape, "float64") # mean(values)
        self.M2_a = np.zeros(shape, "float64") # sum((values-mean)**2)
        self.std = np.zeros(shape, "float64")
        self.epsilon = epsilon
        self.center = center
        self.clip_range_sup = clip_range_sup
        self.clip_range_inf = clip_range_inf
        
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def update(self, n_b, avg_b, M2_b):
        n = self.n_a + n_b
        delta = avg_b - self.avg_a
        M2 = self.M2_a + M2_b + np.square(delta) * (self.n_a * n_b / n)
        var_ab = M2 / (n - 1)

        self.avg_a = (self.avg_a * self.n_a + avg_b * n_b)/n
        self.n_a = n
        self.M2_a = M2
        self.std = np.sqrt(np.maximum(var_ab, self.epsilon))

    def normalize(self, v):
        return np.clip(np.divide((v - self.avg_a), self.std) + self.center, self.clip_range_inf, self.clip_range_sup)

    def denormalize(self, v):
        return self.avg_a + v * self.std


class AverageStdNormalizer:
    def __init__(self, shape=(), center=0, clip_range_sup=np.inf, clip_range_inf=np.NINF):
        self.rstd = RunningStd(shape=shape, center=center, clip_range_inf=clip_range_inf, clip_range_sup=clip_range_sup)
    
    def normalize(self, obs_batch, update=True):
        # accetta come parametro un batch di obs (frames).
        # restituisce la sua versione normalizzata pixel per pixel
        if update:
            batch_size = obs_batch.shape[0]
            mean = np.mean(obs_batch, axis=0) # compute the mean over the batch dimension (has the shape of an obs)
            mse_from_mean = np.sum(np.square(obs_batch-mean), axis=0) # (has the shape of an obs)
            self.rstd.update(batch_size, mean, mse_from_mean)
        return self.rstd.normalize(obs_batch)


