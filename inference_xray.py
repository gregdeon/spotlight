import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models

import os
import numpy as np
from tqdm import tqdm

import spotlight
from spotlight.datasets import *
from spotlight.utils import *
from spotlight.models import *

data_dir = os.environ['DATA_DIR'] 
xray_dir = os.path.join(data_dir, 'xray')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'xray', 'cnn.pt')

# Load training set
print('Loading dataset...')
xray = ChestXRay(
    xray_dir,
    train=True,
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ]))

dataloader = DataLoader(
    xray,
    shuffle=False,
    batch_size=624,
    pin_memory=True,
    num_workers=0)

# Load model
print('Loading model...')
model = xray_model

# Load parameters
checkpoint = torch.load(model_path, map_location='cpu') 
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    name = k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()
model.cuda()

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook
model[8].register_forward_hook(get_input('last_layer'))

# Run inference on entire dataset
print('Running inference...')
hidden_list = []
loss_list = []
output_list = []
criterion = nn.CrossEntropyLoss(reduction='none')
softmax = nn.Softmax(dim=1)
with torch.no_grad():
    for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
        inputs = batch[0].cuda()
        targets = batch[1].cuda()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        predictions = softmax(outputs)
        
        hidden_list.append(hidden_layers['last_layer'].cpu())
        loss_list.append(loss.cpu())
        output_list.append(predictions[:, 1].cpu())

embeddings = torch.vstack(hidden_list)
outputs = torch.hstack(output_list)
losses = torch.hstack(loss_list)

saveInferenceResults(
    fname      = os.path.join('inference_results', 'xray_train_cnn.pkl'),
    embeddings = embeddings,
    outputs    = outputs,
    losses     = losses,
)
