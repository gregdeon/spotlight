import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models, datasets

import os
import numpy as np
from tqdm import tqdm

import torch_spotlight
from torch_spotlight.utils import *

data_dir = os.environ['DATA_DIR'] 
imagenet_dir = os.path.join(data_dir, 'imagenet')
# Note: we load a pretrained ImageNet model from a file, rather than from the internet, to make inference work without an internet connection
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'imagenet', 'resnet18.pth')

# Load validation set
print('Loading dataset...')
imagenet = datasets.ImageNet(
    root=imagenet_dir, 
    split='val',
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
)

dataloader = DataLoader(
    imagenet,
    shuffle=False,
    batch_size=128,
    pin_memory=True,
    num_workers=0
)

# Load model
print('Loading model...')
model = models.resnet18(pretrained=False, progress=False)
model.eval()

# Load parameters
state_dict = torch.load(os.path.join(model_path), map_location='cuda:0') 
model.load_state_dict(state_dict)
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

results = InferenceResults(
    embeddings = torch.clone(embeddings_cls),
    outputs    = outputs,
    losses     = losses,
)
saveResults(
    os.path.join('inference_results', 'imagenet_val_resnet.pkl'),
    results
)
