from __future__ import print_function, absolute_import

import exchangable_tensor.models
from exchangable_tensor.losses import mse, ce, softmax
from data import df_to_matrix, get_mask, to_indicator, to_number
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
CUDA = True

Encoder = exchangable_tensor.models.Encoder
Decoder = exchangable_tensor.models.Decoder

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train = pd.read_csv("./data/ml-100k/u1.base", sep="\t", names=r_cols, encoding='latin-1')
validation = pd.read_csv("./data/ml-100k/u1.test", sep="\t", names=r_cols, encoding='latin-1')
train = df_to_matrix(train, 943, 1682).toarray()
train_id = to_indicator(train)
train_mask = get_mask(train)
validation = df_to_matrix(validation, 943, 1682).toarray()
valid_id = to_indicator(validation)
valid_mask = get_mask(validation)

def prep_var(x):
    v = Variable(torch.from_numpy(x))
    if CUDA:
        v = v.cuda()
    return v

enc = Encoder(5, [14, 5], functions="max", embedding_pool="mean")
dec = Decoder(5*2, [14, 5], functions="mean")
if CUDA:
    enc.cuda()
    dec.cuda()

pars = [i for i in enc.parameters()] + [i for i in dec.parameters()]
for p in pars:
    if len(p.size()) > 1:
        nn.init.normal(p,std=0.01)
    else:
       nn.init.constant(p, 0.01)

optimizer = torch.optim.Adam(pars, lr=0.01)

train_x = prep_var(train)
train_id = prep_var(train_id)
train_mask = prep_var(train_mask[:, :, None])
val_x = prep_var(validation)
val_mask = prep_var(valid_mask[:, :, None])
valid_id = prep_var(valid_id)

def expected_val(pred):
    n, m, d = pred.size()
    p = Variable(torch.arange(1,6)).view((5,1))
    if CUDA:
        p = p.cuda()
    return torch.mm(softmax(pred).view((n*m, d)), p).view((n,m, 1)) 

epochs = 1000
for ep in xrange(epochs):
    optimizer.zero_grad()
    #print(train_id.size(), train_mask.size())
    embeddings = enc(train_id, train_mask)
    y_hat = dec(embeddings, train_mask)
    train_loss = ce(y_hat, train_id, train_mask)
    reg_loss = 0
    for p in pars:
        reg_loss += torch.sum(torch.pow(p, 2))
    loss = train_loss + 0.0001 * reg_loss
    loss.backward()
    mse_train = mse(expected_val(y_hat), train_x, train_mask)
    optimizer.step()
    if ep % 1 == 0:
        val_hat = dec(enc(train_id, train_mask), val_mask)
        mse_val = mse(expected_val(val_hat), val_x, val_mask)
        val_loss = np.sqrt(mse_val.data[0])
    print('Train Epoch: {}, Loss: {:.6f}, MSE: {:.6f}, Val_loss: {:.6f}'.format(ep, loss.data[0], np.sqrt(mse_train.data[0]), val_loss))
