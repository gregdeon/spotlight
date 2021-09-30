import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models

import os
import numpy as np
from tqdm import tqdm

import torch_spotlight
from torch_spotlight.datasets import *
from torch_spotlight.utils import *
from torch_spotlight.utils import InferenceResults, saveResults
import wilds
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

# Load full set
data_dir = os.environ['TMPDIR'] + '/data'
dataset = get_dataset(dataset='iwildcam', root_dir=data_dir)

# Get the test set
test_data = dataset.get_subset('test',
                    transform=transforms.Compose([transforms.Resize((448,448)),
                                                  transforms.ToTensor()]))

# Prepare the standard data loader
test_loader = get_eval_loader('standard', test_data, batch_size=16)

# Load model
print('Loading model ...')
model = models.resnet50(pretrained=False, progress=False)
model.fc = nn.Linear(2048, 182)
model.eval()

# Load parameters
print('Loading parameters ...')
checkpoint = torch.load(os.path.join('model/iwildcam_seed:0_epoch:best_model.pth'), map_location=torch.device('cuda'))
state_dict = checkpoint['algorithm']
new_state_dict = {}
for k, v in state_dict.items():
    name = k[6:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to('cuda')

# Add hook to capture hidden layer
print('Adding hook ...')
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
    for batch_num, batch in tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True):
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
    embeddings = embeddings,
    outputs    = outputs,
    losses     = losses,
)
saveResults(os.path.join('inference_results', 'iwild_test_resnet_ood.pkl'), results)