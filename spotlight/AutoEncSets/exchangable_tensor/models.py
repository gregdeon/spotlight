import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from layers import SetPool, MatrixLayer, MatrixLinear

class Encoder(nn.Module):
    def __init__(self, input_dim, units, functions="mean", activation="relu", embedding_pool="max"):
        super(Encoder, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [input_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            axes = ["row", "column", "both"] if i < (len(units)-1) else []
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1], axes=axes))
        self.embeddings = [SetPool("row", embedding_pool, expand=True), 
                           SetPool("column", embedding_pool, expand=True)]
        self.layers = nn.ModuleList(layers)

    def forward(self, input, mask=None):
        state = input
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state, mask)
            if i == last:
                break
            if self.activation == "relu":
                state = F.relu(state)
            if mask is not None:
                state = torch.mul(state, mask)
        return [f(state, mask) for f in self.embeddings]
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, units, functions="mean", activation="relu"):
        super(Decoder, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [embedding_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, input_list, mask=None):
        state = torch.cat(input_list, dim=2)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state, mask)
            if i == last:
                break
            if self.activation == "relu":
                state = F.relu(state)
                if mask is not None:
                    state = torch.mul(state, mask)
        return state

class DenoisingAE(nn.Module):
    def __init__(self, input_dim, units, functions="mean", activation="relu"):
        super(DenoisingAE, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [input_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, input, in_mask):
        state = input
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state, pool_mask=in_mask if i == 0 else None)
            if i == last:
                break
            if self.activation == "relu":
                state = F.relu(state)
        return state
