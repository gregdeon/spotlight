from __future__ import print_function, absolute_import

import exchangable_tensor.models
from exchangable_tensor.losses import mse, ce, softmax
from data import df_to_matrix, get_mask, to_indicator, to_number
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from scipy.sparse import csr_matrix

def sub_mask(row, col, users, movies, p=0.5, to_dense=True):
    n = row.shape[0]
    l = int(n * p)
    idx = np.random.permutation(n)
    matrix = csr_matrix((np.ones(l), (row[idx[0:l]], col[idx[0:l]])),
                        shape=(users, movies), dtype="float32")
    inverse = csr_matrix((np.ones(n-l), (row[idx[l:]], col[idx[l:]])),
                        shape=(users, movies), dtype="float32")
    if to_dense:
        return matrix.toarray(), inverse.toarray()
    else:
        return matrix, inverse

DenoisingAE = exchangable_tensor.models.DenoisingAE

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train_data = pd.read_csv("./data/ml-100k/u1.base", sep="\t", names=r_cols, encoding='latin-1')
validation_data = pd.read_csv("./data/ml-100k/u1.test", sep="\t", names=r_cols, encoding='latin-1')
train = df_to_matrix(train_data, 943, 1682).toarray()
train_id = to_indicator(train)
train_mask = get_mask(train)
validation = df_to_matrix(validation_data, 943, 1682).toarray()
valid_id = to_indicator(validation)
valid_mask = get_mask(validation)

def prep_var(x):
    return Variable(torch.from_numpy(x))

dec = DenoisingAE(5, [50, 50, 5], functions="mean")

for p in dec.parameters():
    if len(p.size()) > 1:
        nn.init.normal(p,std=0.001)
    else:
       nn.init.constant(p, 0.1)

optimizer = torch.optim.Adam(dec.parameters(), lr=0.00005)

train_x = prep_var(train)
train_mask = prep_var(train_mask[:, :, None])
train_id = prep_var(train_id)
val_x = prep_var(validation)
val_mask = prep_var(valid_mask[:, :, None])
valid_id = prep_var(valid_id)

def expected_val(pred):
    n, m, d = pred.size()
    return torch.mm(softmax(pred).view((n*m, d)), Variable(torch.arange(1,6)).view((5,1))).view((n,m, 1))

epochs = 1000
retain_p = 0.8
for ep in xrange(epochs):
    optimizer.zero_grad()
    #print(train_id.size(), train_mask.size())
    mask, inv_mask = [prep_var(i[:,:,None]) for i in sub_mask(train_data.user_id-1,
                                                              train_data.movie_id-1,
                                                              943, 1682, retain_p)]
    y_hat = dec(train_id, mask)
    #print(y_hat.data.numpy())
    #print(y_hat.data.numpy()[0,0,:])
    #exit()
    train_loss = ce(y_hat, train_id, inv_mask)
    reg_loss = 0
    for p in dec.parameters():
        reg_loss += torch.sum(torch.pow(p, 2))
    loss = train_loss + 0.0001 * reg_loss
    loss.backward()
    mse_train = mse(expected_val(y_hat), train_x, inv_mask)
    optimizer.step()
    if ep % 1 == 0:
        val_hat = dec(train_id, val_mask)
        mse_val = mse(expected_val(val_hat), val_x, val_mask)
        val_loss = np.sqrt(mse_val.data[0])
    print('Train Epoch: {}, Loss: {:.6f}, MSE: {:.6f}, Val_loss: {:.6f}'.format(ep, loss.data[0], np.sqrt(mse_train.data[0]), val_loss))
