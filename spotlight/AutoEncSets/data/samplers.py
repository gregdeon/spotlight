import numpy as np
from data import reindex

class Sampler(object):
    def __init__(self, batch_size, dataset, seed=None, sample_test=False):
        self.rng = np.random.RandomState(seed)
        self.sample_test = sample_test
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __call__(self, batch_size=None):
        raise NotImplemented()
        
class UniformSampler(Sampler):
    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        dataset = self.dataset
        if self.sample_test:
            n = len(self.dataset)
        else:
            indicator = self.dataset.indicator
            idx = np.arange(indicator.shape[0])[indicator != 2]
            n = idx.shape[0]
        indices = self.rng.choice(idx, batch_size, replace=False)
        return dataset[indices]

class ConditionalSampler(Sampler):
    def __init__(self, maxN, maxM, dataset, seed=None, sample_test=False, batch_size=None):
        # TODO: figure out the right way to map between maxN, maxM and batch_size
        self.rng = np.random.RandomState(seed)
        self.sample_test = sample_test
        self.dataset = dataset
        self.batch_size = batch_size
        self.maxN = maxN
        self.maxM = maxM

    def __call__(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        rng = self.rng
        if self.sample_test:
            mask_indices = dataset.index
            dataset = self.dataset
        else:
            indicator = self.dataset.indicator
            mask_indices = self.dataset.index[indicator != 2]
            dataset = self.dataset
        
        num_vals = mask_indices.shape[0]
        N, M = mask_indices.max(axis=0) + 1
        maxN = self.maxN
        maxM = self.maxM

        pN = np.bincount(mask_indices[:,0], minlength=N).astype(np.float32)
        pN /= pN.sum()
        ind_n = np.arange(N)[pN != 0] # If there are 0s in p and replace is False, we cant select N=maxN unique values. Filter out 0s.
        pN = pN[pN != 0]
        maxN = min(maxN, ind_n.shape[0])
        ind_n = rng.choice(ind_n, size=maxN, replace=False, p=pN)

        select_row = np.in1d(mask_indices[:,0], ind_n)
        rows = mask_indices[select_row]

        pM = np.bincount(rows[:,1], minlength=M).astype(np.float32)
        pM /= pM.sum()
        ind_m = np.arange(M)[pM!=0] # If there are 0s in p and replace is False, we cant select M=maxM unique values. Filter out 0s.
        pM = pM[pM!=0] 
        maxM = min(maxM, ind_m.shape[0])
        ind_m = rng.choice(ind_m, size=maxM, replace=False, p=pM)

        select_col = np.in1d(mask_indices[:,1], ind_m)
        select_row_col = np.logical_and(select_row, select_col)
        indices = np.arange(num_vals)[select_row_col]
        data = dataset[indices]
        data["index"] = reindex(data["index"])
        return data
