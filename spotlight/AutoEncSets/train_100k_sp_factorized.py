import time
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
from torch.autograd import Variable

from exchangable_tensor.sp_layers import SparseExchangeable, SparseSequential, SparsePool
from data import prep, collate_fn, CompletionDataset

from data.loader import IndexIterator
from data.samplers import ConditionalSampler, UniformSampler
import data.recsys

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='number of epochs to train for', default=5000)
parser.add_argument('--clip', type=float, help='Gradient clipping: max norm of the gradients', default=1.)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
parser.add_argument('--nocuda', action='store_true', help='disables cuda')
parser.add_argument('--sampler', default='uniform', help='Which sampling method to use',
                                choices=['uniform','conditional'])
args = parser.parse_args()
use_cuda = not args.nocuda
device = 'cuda' if use_cuda else 'cpu'

def prep_data(x, requires_grad=False):
    '''
    Helper function for setting up data variables
    '''
    x = Variable(x, requires_grad=requires_grad)
    if use_cuda:
        x = x.cuda()
    return x

def mask_inputs(batch):
    '''
    Mask inputs by setting some subset of the ratings to 0.
    '''
    input = batch['input']
    # indicator == 0 if training example, 1 if validation example and 2 if test example
    indicator = batch['indicator']
    
    if not isinstance(input, np.ndarray):
        input = input.numpy()
    if not isinstance(indicator, np.ndarray):
        indicator = indicator.numpy()
    
    # set validation and test ratings to 0.
    input[indicator == 2] = input[indicator == 2] * 0.
    input[indicator == 1] = input[indicator == 1] * 0.
    
    # prepare for pytorch by moving numpy arrays to torch arrays
    batch["input"] = torch.from_numpy(input)
    for key in ["target", "index"]:
        if isinstance(batch[key], np.ndarray):
            batch[key] = torch.from_numpy(batch[key])
    return batch, drop
    

data = data.recsys.ml100k(0.)
dataloader = torch.utils.data.DataLoader(data, num_workers=1, 
                                         collate_fn=collate_fn, 
                                         batch_size=80000, shuffle=True 
                                        )
index = prep(data.index, dtype="int")
if use_cuda:
    index = index.cuda()

# build model


class FactorizedAutoencoder(nn.Module):
    def __init__(self, index_train, index_eval=None, embedding_dim=32):
        super(FactorizedAutoencoder, self).__init__() 
        if index_eval is None:
            index_eval = index_train
        self.enc = SparseSequential(index_train, 
                            SparseExchangeable(5,150, index_train), 
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                           # SparseExchangeable(150,150, index_train),
                           # nn.LeakyReLU(),
                           # torch.nn.Dropout(p=0.5),
                            #SparseExchangeable(150,150, index_train),
                            #nn.LeakyReLU(),
                            #torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,150, index_train),
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,embedding_dim, index_train)
                        )

        self.dec = SparseSequential(index_eval, 
                            SparseExchangeable(2 * embedding_dim, 150, index_eval), 
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                          #  SparseExchangeable(150,150, index_eval),
                          #  nn.LeakyReLU(),
                          #  torch.nn.Dropout(p=0.5),
                            #SparseExchangeable(150,150, index_eval),
                            #nn.LeakyReLU(),
                            #torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,150, index_eval),
                            nn.LeakyReLU(),
                            torch.nn.Dropout(p=0.5),
                            SparseExchangeable(150,5, index_eval)
                        )
        
        self.pool_row = SparsePool(index_train[:, 0], embedding_dim, keep_dims=False)
        self.pool_col = SparsePool(index_train[:, 1], embedding_dim, keep_dims=False)
        self._index_train = index_train
        self._index_eval = index_eval

    def set_indices(self, index_train, index_eval=None):
        if index_eval is None:
            index_eval = index_train
        self.index_train = index_train
        self.index_eval = index_eval

    @property
    def index_train(self):
        return self._index_train
    
    @index_train.setter
    def index_train(self, index):
        self.enc.index = index
        self.pool_row.index = index[:, 0]
        self.pool_col.index = index[:, 1]
        self._index_train = index

    @property
    def index_eval(self):
        return self._index_eval
    
    @index_eval.setter
    def index_eval(self, index):
        self.dec.index = index
        self._index_eval = index

    def forward(self, input):
        encoded = self.enc(input)
        row_mean = self.pool_row(encoded)
        col_mean = self.pool_col(encoded)
        embeddings = torch.cat([torch.index_select(row_mean, 0, self.index_eval[:, 0]), 
                                torch.index_select(col_mean, 0, self.index_eval[:, 1])], dim=1)
        output = self.dec(embeddings)
        return output


model = FactorizedAutoencoder(index)

if use_cuda:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Prepare cross entropy loss
ce = torch.nn.CrossEntropyLoss(reduce=False)
def masked_loss(output, target, drop=None):
    if drop is None:
        return ce(output, target).mean()
    else:
        mask = torch.zeros_like(target)
        mask[drop] = 1
        mask = mask.float()
        ce_loss = ce(output, target)
        return (mask * ce_loss) / mask.sum()

# Prepare mean square error loss
mse = torch.nn.MSELoss()
values = prep_data(torch.arange(1,6)[None,:])
softmax = torch.nn.Softmax(dim=1)
def expected_mse(output, target):
    output = softmax(output)
    y = (output * values).sum(dim=1)
    return mse(y, target)

if args.sampler == "uniform":
    samples_per_batch = 80000
    sampler = UniformSampler(samples_per_batch, data)
    iters_per_epoch = int(data.n_train / samples_per_batch)
else:
    maxN = 400
    maxM = 400
    N, M = data.index.max(axis=0) + 1
    sampler = ConditionalSampler(maxN, maxM, data)
    iters_per_epoch = int(np.ceil(N//maxN) * np.ceil(M//maxM))
    
t = time.time()
for epoch in range(args.epochs):
    # Training steps
    model.train()
    iterator = IndexIterator(iters_per_epoch, sampler, n_workers=1, epochs=1)
    for sampled_batch in tqdm(iterator):
        train_idx = sampled_batch['indicator'] == 0
        target = torch.from_numpy((sampled_batch["target"][train_idx] - 1)).to(device).long()
        input = torch.from_numpy(sampled_batch["input"][train_idx]).to(device)
        index_train = torch.from_numpy(sampled_batch["index"][train_idx]).to(device)
        model.set_indices(index_train)
        optimizer.zero_grad()
        output = model(input)
        l = masked_loss(output, target.squeeze(1))
        l.backward()
        clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    # Evaluation
    model.eval()
    with torch.no_grad():
        full_batch = data[np.arange(100000)]
        train_idx = full_batch['indicator'] == 0
        eval_idx = full_batch['indicator'] > 0
        target = torch.from_numpy((full_batch["target"][eval_idx])).to(device).long()
        input = torch.from_numpy(full_batch["input"][train_idx]).to(device)
        index_train = torch.from_numpy(full_batch["index"][train_idx]).to(device)
        index_eval = torch.from_numpy(full_batch["index"][eval_idx]).to(device)
        model.set_indices(index_train, index_eval)
        test_loss = expected_mse(model(input), target.squeeze(1).float())
        tqdm.write("%d, %s, %s" % (epoch, l.cpu().data.numpy(), 
                            np.sqrt(test_loss.cpu().data.numpy())))
        torch.cuda.empty_cache()

torch.save(model, '100k_model.pt')
sec_per_ep = (time.time() - t) / args.epochs
print("Time per epoch: %1.3f" % (sec_per_ep))
print("Est total time: %1.3f" % (sec_per_ep * 10000 / 60 / 60))
