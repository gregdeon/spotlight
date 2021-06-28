from pdb import set_trace as bp
import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import combinations
import numpy as np

import numpy as np

def append_features(index, interaction=None, row_values=None, col_values=None, dtype="float32"):
    '''
    Append features to the values matrix using the index to map to the correct dimension.

    Used when we have row or column features. Assumes that the index a zero-index (i.e. counts from zero).
    '''
    if interaction is None and row_values is None and col_values is None:
        raise Exception("Must supply at least one value array.")
    values = np.zeros((index.shape[0], 0), dtype=dtype)
    if interaction is not None:
        if len(interaction.shape) == 1:
            interaction = interaction[:, None]
        values = np.concatenate([values, interaction], axis=1)
    if row_values is not None:
        if len(row_values.shape) == 1:
            row_values = row_values[:, None]
        values = np.concatenate([values, row_values[index[:, 0], ...]], axis=1)
    if col_values is not None:
        if len(col_values.shape) == 1:
            col_values = col_values[:, None]
        values = np.concatenate([values, col_values[index[:, 1], ...]], axis=1)
    return values

def subsets(n, return_empty=False):
    '''
    Get all proper subsets of [0, 1, ..., n]
    '''
    sub = [i for j in range(n) for i in combinations(range(n), j)]
    if return_empty:
        return sub
    else:
        return sub[1:]

def to_valid_index(index):
    _, valid_index = np.unique(index, axis=0, return_inverse=True)
    return valid_index

def prepare_global_index(index, axes=None):
    if axes is None:
        axes = subsets(index.shape[1])
    return np.concatenate([to_valid_index(index[:, ax])[:, None] for ax in axes], axis=1)

def append_features(index, interaction=None, row_values=None, col_values=None, dtype="float32"):
    '''
    Append features to the values matrix using the index to map to the correct dimension.

    Used when we have row or column features. Assumes that the index a zero-index (i.e. counts from zero).
    '''
    if interaction is None and row_values is None and col_values is None:
        raise Exception("Must supply at least one value array.")
    values = np.zeros((index.shape[0], 0), dtype=dtype)
    if interaction is not None:
        if len(interaction.shape) == 1:
            interaction = interaction[:, None]
        values = np.concatenate([values, interaction], axis=1)
    if row_values is not None:
        if len(row_values.shape) == 1:
            row_values = row_values[:, None]
        values = np.concatenate([values, row_values[index[:, 0], ...]], axis=1)
    if col_values is not None:
        if len(col_values.shape) == 1:
            col_values = col_values[:, None]
        values = np.concatenate([values, col_values[index[:, 1], ...]], axis=1)
    return values

class SparsePool(nn.Module):
    '''
    Sparse pooling with lazy memory management. Memory is set with the initial index, but 
    can be reallocated as needed by changing the index.

    Caching deals with the memory limitations of these models by computing the pooling layers on
    CPU memory. A typical forward pass still uses batches on the GPU but pools on the CPU 
    (see SparseExchangable below).
    '''
    def __init__(self, full_index, out_features, out_size=None, keep_dims=True, eps=1e-9, cache_size=None, axis=None, control=False, poly=1, deepset=None):
        super(SparsePool, self).__init__()
        self.eps = eps
        if axis is not None:
            full_index = full_index[:, axis]
        self.axis = axis
        self._index = None
        self.out_features = out_features
        self.keep_dims = keep_dims
        if out_size is None:
            out_size = int(full_index.max() + 1)
        self.out_size = out_size
        self.output = Variable(torch.zeros((out_size, self.out_features))).to(full_index.device) # TODO: this should be none
        self.norm = Variable(torch.zeros((out_size))).to(full_index.device)
        self.cache_size = cache_size
        self.poly = poly
        self.deepset = deepset

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        '''
        Setter for changing the index. If the index changes, we recalculate the normalization terms
        and if necessary, resize memory allocation.
        '''
        if self.axis is not None:
            index = index[:, self.axis]
        self._index = index
        out_size = int(index.max() + 1)
        if out_size != self.out_size or self.norm is None:
            del self.output, self.norm
            self.output = Variable(torch.zeros((out_size, self.out_features)))
            self.norm = Variable(torch.zeros((out_size)))
            self.output = self.output.to(index.device)
            self.norm = self.norm.to(index.device)
            self.out_size = out_size
        
        self.norm = torch.zeros_like(self.norm.to(index.device)).index_add_(0, index,
                                         torch.ones_like(index.float())) + self.eps
        
    def zero_cache(self):
        '''
        We incrementally compute the pooled representation in batches, so we need a way of clearing
        the cached representation.
        '''
        if self.cache_size is None:
            raise ValueError("Must specify a cache size if using a cache")
        self._cache = torch.zeros((self.cache_size, self.out_features))
        self._cache_norm = torch.zeros((self.cache_size)) + self.eps
        
    def update_cache(self, input, index):
        '''
        Add a batch to the pooled representation
        '''
        self._cache = self._cache.index_add_(0, index.cpu(), input.cpu())
        self._cache_norm = self._cache_norm.index_add_(0, index.cpu(),
                                            torch.ones_like(index.cpu().float())) + self.eps
    
    def get_cache(self, index, keep_dims=True):
        '''
        Return the pooled representation.
        '''
        output = self._cache / self._cache_norm[:, None].float()
        if keep_dims:
            return torch.index_select(output, 0, index.cpu())
        else:
            return output

    def mean(self, input, index, ind_max):
        output = torch.zeros((ind_max, input.shape[1])).to(input.device).index_add_(0, 
                                                          index, 
                                                          input)
        norm = torch.zeros(ind_max).to(input.device).index_add_(0, index, torch.ones_like(index).float()) + self.eps
        
        return output / norm[:, None].float()
    
    def forward(self, input, keep_dims=None, cached_activations=None, index=None):
        '''
        Regular forward pass.
        '''
        if index is None:
            index = self.index
            if index is None:
                raise Exception("Must set or pass an index before calling the model.")
        else:
            global_index = index
            unique, index = torch.unique(index.cpu(), return_inverse=True, sorted=True)
            index = index.to(input.device)

        if keep_dims is None:
            keep_dims = self.keep_dims
        ind_max = int(index.max() + 1)
        
        if self.deepset is not None:
            input = self.deepset(input)
        
        if self.poly > 1:
            input_poly = torch.zeros((input.shape[0], input.shape[1] * self.poly))
            polys = [torch.ones_like(input).to(input.device), input]
            for p in range(2, self.poly + 1):
                # Chebyshev polynomials
                polys.append(2 * input * polys[-1] - polys[-2])
            input = torch.cat(polys[1:], dim=1)

        output = self.mean(input, index, ind_max)
        if cached_activations is not None and self.training:
            with torch.no_grad():
                cached_subsample = torch.zeros_like(output).index_add_(0, 
                                                                    index, 
                                                                    cached_activations.to(output.device))
                cached_subsample = cached_subsample / norm[:, None].float()
                cached_full = self._cache[unique, :] / self._cache_norm[unique, None].float()
                cv = (cached_subsample + cached_full.to(cached_subsample.device)).detach()
            output = output - cv

        if keep_dims:
            return torch.index_select(output,
                                      0, index)
        else:
            return output
        

def mean_pool(input, index, axis=0, out_size=None, keep_dims=True, eps=1e-9):
    '''
    Sparse mean pooling. This function performs the same role as the class
    above but is approximately 15% slower. Kept in the codebase because it
    is much more readable.
    '''
    if out_size is None:
        out_size = index[:, axis].max().data[0] + 1
    # Sum across values
    out = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    out = out.index_add_(0, index[:, axis], input)
    
    # Normalization
    norm = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    norm = norm.index_add_(0, index[:, axis], torch.ones_like(input)) + eps
    if keep_dims:
        return torch.index_select(out / norm, 0, index[:, axis])
    else:
        return out / norm

class SparseExchangeable(nn.Module):
    """
    Sparse exchangable matrix layer
    """
    def __init__(self, in_features, out_features, index, bias=True, cache_size=None, use_control_variates=False, poly_pool=1, deepset=None):
        super(SparseExchangeable, self).__init__()
        if not isinstance(deepset, list):
            deepset = [deepset] * index.shape[1]
        self._index = index
        self.pooling = nn.ModuleList([SparsePool(index[:, i], in_features, poly=poly_pool, deepset=deepset[i]) for i in range(index.shape[1])])
        self.linear = nn.Linear(in_features=in_features * 2 + in_features * poly_pool * (index.shape[1]),
                                out_features=out_features,
                                bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.cache_size = cache_size
        self.cv = use_control_variates

    def zero_cache(self):
        self._cache = torch.zeros(self.cache_size, self.out_features)
    
    def update_cache(self, input, index, batch_size=10000):
        nnz = index.shape[0] # number of non-zeros
        cache_sizes = index.cpu().numpy().max(axis=0) + 1
        batch_size = min(batch_size, nnz)
        splits = max(nnz // batch_size, 1)
        for i, p in enumerate(self.pooling):
            p.cache_size = int(cache_sizes[i])
            p.zero_cache()
        
        indices = list(np.split(np.arange( (nnz // batch_size) * batch_size ), splits)) + [np.arange( (nnz // batch_size) * batch_size, nnz )]
        for i in indices:
            for j, p in enumerate(self.pooling):
                p.update_cache(input.cpu()[i, ...], index.cpu()[i, j])
        pooled = [p.get_cache(index.cpu()[:,i], keep_dims=True) for i, p in enumerate(self.pooling)]
        pooled += [torch.mean(input.cpu(), dim=0).expand_as(input)]
        stacked = torch.cat([input.cpu()] + pooled, dim=1)
        for i in indices:
            self._cache[i,...] = self.linear(stacked[i,...].to(self.linear.weight.device)).cpu()
    
    def get_cache(self, idx=None):
        if idx is None:
            return self._cache
        else:
            return self._cache[idx, ...]
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        for i, module in enumerate(self.pooling):
            module.index = index[:, i]
        self._index = index
    
    def forward(self, input, cached_activations=None, index=None):
        pooled = [pool_axis(input, cached_activations=cached_activations, index=index[:, i] if index is not None else None) 
                  for i, pool_axis in enumerate(self.pooling)]
        pooled += [torch.mean(input, dim=0).expand_as(input)]
        stacked = torch.cat([input] + pooled, dim=1)
        activation = self.linear(stacked)
        return activation

    def cached_forward(self, input, index, batch_size):
        self.cache_size = index.shape[0]
        self.zero_cache()
        self.update_cache(input, index, batch_size=batch_size)
        return self.get_cache()

    # def cached_forward(self, input, index, batch_size):
    #     self.cache_size = index.shape[0]
    #     self.zero_cache()
    #     self.update_cache(input, index, batch_size=batch_size)
    #     return self.get_cache()

class SparseFactorize(nn.Module):
    """
    Sparse factorization layer
    """

    def forward(self, input, index):
        row_mean = mean_pool(input, index, 0)
        col_mean = mean_pool(input, index, 1)
        return torch.cat([row_mean, col_mean], dim=1)#, index


class SparseSequential(nn.Module):
    def __init__(self, index, *args):
        super(SparseSequential, self).__init__()
        self._index = index
        self.layers = nn.ModuleList(list(args))
        
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        for l in self.layers:
            if hasattr(l, "index"):
                l.index = index
        self._index = index
    
    def forward(self, input, index=None, sampling_index=None):
        out = input
        if sampling_index is not None:
            cache = input
        else:
            cache = None
        for id, l in enumerate(self.layers):
            if isinstance(l, SparseExchangeable):
                out = l(out, cached_activations=cache, index=index)
                if sampling_index is not None:
                    cache = l.get_cache(sampling_index)
            else:
                out = l(out)
                if sampling_index is not None:
                    cache = l(cache)
        return out
    
    def cached_forward(self, input, index, batch_size=10000):
        with torch.no_grad():
            state = input
            for i, layer in enumerate(self.layers):
                if hasattr(layer, "cached_forward"):
                    state = layer.cached_forward(state, index, batch_size)
                else:
                    state = layer(state)
            return state

# Not used...

def mean_pool(input, index, axis=0, out_size=None, keep_dims=True, eps=1e-9):
    '''
    Sparse mean pooling. This function performs the same role as the class
    above but is approximately 15% slower. Kept in the codebase because it
    is much more readable.
    '''
    if out_size is None:
        out_size = index[:, axis].max().data[0] + 1
    # Sum across values
    out = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    out = out.index_add_(0, index[:, axis], input)
    
    # Normalization
    norm = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    norm = norm.index_add_(0, index[:, axis], torch.ones_like(input)) + eps
    if keep_dims:
        return torch.index_select(out / norm, 0, index[:, axis])
    else:
        return out / norm
