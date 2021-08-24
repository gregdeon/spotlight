import time
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch
from torch.autograd import Variable

from exchangable_tensor.sp_layers import SparseExchangeable, SparseSequential
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

def prep_data(x, requires_grad=False):
    '''
    Helper function for setting up data variables
    '''
    x = Variable(x, requires_grad=requires_grad)
    if use_cuda:
        x = x.cuda()
    return x

def mask_inputs(batch, percent_train = 0.15):
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
    
    # during training we set some percent of the training ratings to 0.
    if percent_train > 0.:
        # set validation and test ratings to 0.
        input[indicator == 2] = input[indicator == 2] * 0.
        input[indicator == 1] = input[indicator == 1] * 0.
        
        # sample training ratings to set to 0.
        n_train = input.shape[0]
        idx = np.arange(n_train)
        drop = np.random.permutation(idx[indicator == 0])[0:int(percent_train * n_train)]
        input[drop] = input[drop] * 0.
        
        # prepare for pytorch by moving numpy arrays to torch arrays
        batch["input"] = torch.from_numpy(input)
        for key in ["target", "index"]:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
        return batch, drop
    
    # during evaluation only the test ratings are set to 0.
    else:
        # set test ratings to zero.
        input[indicator == 2] = input[indicator == 2] * 0.
        idx = np.arange(input.shape[0])
        drop = idx[indicator == 2]
        
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

units = 100
# build model
enc = SparseSequential(index, 
                       SparseExchangeable(5,units, index), 
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(units,units, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(units,units, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       #SparseExchangeable(units,units, index),
                       #nn.LeakyReLU(),
                      # torch.nn.Dropout(p=0.5),
                       #SparseExchangeable(units,units, index),
                       #nn.LeakyReLU(),
                       #torch.nn.Dropout(p=0.5),
                       SparseExchangeable(units,units, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(units,5, index)
                   )
if use_cuda:
    enc.cuda()
optimizer = torch.optim.Adam(enc.parameters(), lr=args.lr)

# Prepare cross entropy loss
ce = torch.nn.CrossEntropyLoss(reduce=False)
def masked_loss(output, target, drop, alpha=0.1):
    mask = torch.zeros_like(target)
    mask[drop] = 1
    mask = mask.float()
    ce_loss = ce(output, target)
    return ((1-alpha) * mask * ce_loss / mask.sum() + alpha * (1-mask) * ce_loss / (1-mask).sum()).sum()

# Prepare mean square error loss
mse = torch.nn.MSELoss()
values = prep_data(torch.arange(1,6)[None,:])
softmax = torch.nn.Softmax(dim=1)
def expected_mse(output, target):
    output = softmax(output)
    y = (output * values).sum(dim=1)
    return mse(y, target)

if args.sampler == "uniform":
    samples_per_batch = 1000
    sampler = UniformSampler(samples_per_batch, data)
    iters_per_epoch = int(data.n_train / samples_per_batch)
else:
    maxN = 400
    maxM = 400
    N, M = data.index.max(axis=0) + 1
    sampler = ConditionalSampler(maxN, maxM, data)
    iters_per_epoch = int(np.ceil(N//maxN) * np.ceil(M//maxM))
    
t = time.time()
update_interval = 1
for epoch in range(args.epochs):
    # Training steps
    enc.train()
    iterator = IndexIterator(iters_per_epoch, sampler, n_workers=1, epochs=1)
    if epoch % update_interval == 0:
        enc.cached_forward(torch.from_numpy(data.input), torch.from_numpy(data.index), batch_size=80000)
    for sampled_batch in tqdm(iterator):
        sampled_batch, drop = mask_inputs(sampled_batch)
        target = prep_data((sampled_batch["target"] - 1).long())
        input = prep_data(sampled_batch["input"])
        index = prep_data(sampled_batch["index"])
        #enc.index = index
        optimizer.zero_grad()
        output = enc(input, index=index, sampling_index=sampled_batch["sample_ids"])
        l = masked_loss(output, target.squeeze(1), drop)
        l.backward()
        clip_grad_norm_(enc.parameters(), args.clip)
        optimizer.step()
    # Evaluation
    enc.eval()
    full_batch, drop = mask_inputs(data[np.arange(100000)], 0.)
    target = prep_data((full_batch["target"]).long())
    input = prep_data(full_batch["input"])
    index = prep_data(full_batch["index"])
    enc.index = index
    test_loss = expected_mse(enc(input)[drop,:], target.squeeze(1).float()[drop])
    tqdm.write("%d, %s, %s" % (epoch, l.cpu().data.numpy(), 
                           np.sqrt(test_loss.cpu().data.numpy())))
    torch.cuda.empty_cache()

torch.save(enc, '100k_model.pt')
sec_per_ep = (time.time() - t) / args.epochs
print("Time per epoch: %1.3f" % (sec_per_ep))
print("Est total time: %1.3f" % (sec_per_ep * 10000 / 60 / 60))
