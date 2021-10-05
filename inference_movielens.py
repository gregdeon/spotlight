import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch_spotlight
from torch_spotlight.utils import *
from torch_spotlight.models import *

# Import functions from local copy of AutoEncSets
module_path = os.path.abspath(os.path.join('spotlight/AutoEncSets'))
if module_path not in sys.path:
    sys.path.append(module_path)
import AutoEncSets.data.recsys as recsys
from AutoEncSets.data import prep, collate_fn, CompletionDataset

data_dir = os.environ['DATA_DIR'] 
ml_100k_dir = os.path.join(data_dir, 'movielens', 'ml-100k')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'movielens', '100k_model.pt')

# Load dataset
print('Loading dataset...')
dataset = recsys.ml100k(0.0, path=ml_100k_dir, return_test=True)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, 
                                         collate_fn=collate_fn, 
                                         batch_size=100000, shuffle=False 
                                        )
index = prep(dataset.index, dtype="int")
index = index.cuda()

# Load model
print('Loading model...')
model = torch.load(model_path).to('cuda')
model.to('cuda')

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook

hook_handle = model.dec.layers[6].linear.register_forward_hook(get_input('last_layer'))

# Run inference on entire dataset
print('Running inference...')
ce = torch.nn.CrossEntropyLoss(reduction='none')
softmax = nn.Softmax(dim=1)

hidden_list = []
loss_list = []
output_list = []

with torch.no_grad():
    for batch in dataloader:
        idx = batch['indicator'] == 2
        input_batch = (batch["input"][idx]).to("cuda")
        index_batch = (batch["index"][idx]).to("cuda")
        target_batch = (batch["target"][idx] - 1).to("cuda").long()
        model.set_indices(index_batch)

        output = model(input_batch)
        predictions = softmax(output)
        losses = ce(output, target_batch.squeeze())
        
        hidden_list.append(hidden_layers['last_layer'].cpu())
        output_list.append(predictions.cpu())
        loss_list.append(losses.cpu())
        
embeddings = torch.vstack(hidden_list)
outputs = torch.hstack(output_list)
losses = torch.hstack(loss_list)

results = InferenceResults(
    embeddings = torch.clone(embeddings_cls),
    outputs    = outputs,
    losses     = losses,
)
saveResults(
    os.path.join('inference_results', 'movielens_val_deepset.pkl'),
    results
)
