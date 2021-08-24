from scipy.sparse import csr_matrix
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.autograd import Variable
import torch

def reindex(input_index):
    '''
    Rebuild input_index with minimal unique indices.
    e.g. if the true indices are [567, 234, 567, 102]
    then they will be mapped to  [0, 1, 0, 2].
    '''
    idx = np.zeros_like(input_index)
    for c in [0, 1]:
        _, idx[:, c] = np.unique(input_index[:,c], return_inverse=True)
    return idx

def reindex_all(*args):
    '''
    Given a list of indices, reindex the over the union 
    of the indices and return the new split indices.
    '''
    lengths = [m.shape[0] for m in args]
    merged = np.concatenate(args, axis=0)
    ids = reindex(merged)
    indices = []
    k = 0
    for l in lengths:
        indices.append((k, k+l))
        k += l
    return [ids[i:j, :] for i, j in indices]

def collate_fn(sample):
    sample = default_collate(sample)
    # reindex so that we don't overflow
    index, = reindex_all(sample["index"].numpy())
    return {"index": torch.from_numpy(index), 
            "target": sample["target"], 
            "input": sample["input"], 
            "indicator": sample["indicator"]}



class CompletionDataset(Dataset):
    '''
    A dataset object for holding 
    '''
    def __init__(self, values, index, indicator, one_hot=True, return_test=False, unsorted=False):
        self.return_test = return_test
        self.indicator = np.array(indicator, dtype="int32")
        self.n_train = (indicator != 2).sum()
        self.n_test = (indicator == 2).sum()
        
        self.index = np.array(index, dtype="int")
        self.values = np.array(values, dtype="float32").reshape(index.shape[0], -1) # ensure 2D
        if unsorted:
            # ensure that the values are sorted by indicator so that we only return test values
            # when return_test is true
            idx = np.argsort(indicator)
            self.indicator = self.indicator[idx]
            self.index = self.index[idx,:]
            self.values = self.values[idx,:]

        self.one_hot = one_hot
        if one_hot:
            unique, inv = np.unique(values, return_inverse=True)
            n_unique = unique.shape[0]
            self.input = np.zeros((self.values.shape[0], n_unique), dtype="float32")
            self.input[np.arange(self.values.shape[0]), inv] = 1.
        else:
            self.input = self.values[:, None]

    def __len__(self):
        if self.return_test:
            return self.n_train + self.n_test
        else:
            return self.n_train

    def __getitem__(self, index):
        return {"index": self.index[index, :], 
                "input": self.input[index, :], 
                "target": self.values[index],
                "indicator": self.indicator[index],
                "sample_ids": index}

# Dense implementation functions... (mostly unused)

def prep(x, dtype="float32"):
    if dtype is not None:
        x = np.array(x, dtype=dtype)
    return Variable(torch.from_numpy(x), requires_grad=False)

def df_to_matrix(df, users, movies):
    row = df.user_id - 1
    col = df.movie_id - 1
    data = np.array(df.rating) #- 3.5
    return csr_matrix((data, (row, col)), shape=(users, movies), dtype="float32")

def get_mask(matrix):
    return np.array(1.*(matrix > 0.), dtype="float32")

def to_indicator(mat):
    out = np.zeros((mat.shape[0], mat.shape[1], 5))
    for i in range(1, 6):
        out[:, :, i-1] = (1 * (mat == i)).reshape((mat.shape[0], mat.shape[1]))
    return np.array(out,  dtype="float32")

def to_number(mat):
    out = (np.argmax(mat, axis=2).reshape((mat.shape[0], mat.shape[1], 1)))
    out[mat.sum(axis=2) > 0] += 1
    return np.array(out, dtype="float32")
