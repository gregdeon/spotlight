import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk

import os
import numpy as np
from tqdm import tqdm

import spotlight
from spotlight.utils import *

# Note: we load cached copies of the dataset, tokenizer, and model to make inference work without an internet connection
data_dir = os.environ['DATA_DIR'] 
amazon_dir = os.path.join(data_dir, 'amazon')
model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'amazon')

# Load tokenizer + model
print('Loading model...')
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    cache_dir=model_path, 
    do_lower_case=True, 
    do_basic_tokenize=True,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    cache_dir=model_path,
    local_files_only=True
)
model.eval()
model.to('cuda')

# Load validation set
print('Loading dataset...')
dataset = load_from_disk(amazon_dir)
dataloader = DataLoader(
    Subset(dataset['train'], range(20_000)),
    batch_size=256,
)

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook
model.classifier.register_forward_hook(get_input('last_layer'))

# Run inference on entire dataset
print('Running inference...')
hidden_list = []
loss_list = []
output_list = []
criterion = nn.CrossEntropyLoss(reduction='none')
softmax = nn.Softmax(dim=1)
with torch.no_grad():
    for batch_num, batch in tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True):
        inputs = tokenizer(batch['content'], padding=True, return_tensors='pt').to('cuda')
        targets = batch['label'].cuda()
        
        outputs = model(**inputs)['logits']
        loss = criterion(outputs, targets)
        predictions = softmax(outputs)
        
        hidden_list.append(hidden_layers['last_layer'].cpu())
        loss_list.append(loss.cpu())
        output_list.append(predictions[:, 1].cpu())

embeddings = torch.vstack(hidden_list)
outputs = torch.hstack(output_list)
losses = torch.hstack(loss_list)

saveInferenceResults(
    fname      = os.path.join('inference_results', 'amazon_train_sst.pkl'),
    embeddings = embeddings,
    outputs    = outputs,
    losses     = losses,
)
