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

data_dir = os.environ['DATA_DIR'] 
fairface_dir = os.path.join(data_dir, 'fairface')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'fairface', 'best_resnet.pt')

# Load validation set
print('Loading dataset...')
fairface = FairFace(
    fairface_dir,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4844, 0.3597, 0.3050), (0.2565, 0.2224, 0.2162))
    ])
)

dataloader = DataLoader(
    fairface,
    batch_size=256,
)

# Load model
print('Loading model...')
model = models.resnet34(pretrained=False, progress=False)
model.fc = nn.Linear(512, 2) 
model.eval()

# Load parameters
checkpoint = torch.load(os.path.join(model_path), map_location='cuda:0') 
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to('cuda:0')

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook
model.fc.register_forward_hook(get_input('last_layer'))

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
        targets = batch[2].cuda()
        
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
    fname      = os.path.join('inference_results', 'fairface_val_resnet.pkl'),
    embeddings = embeddings,
    outputs    = outputs,
    losses     = losses,
)
