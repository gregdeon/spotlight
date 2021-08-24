import math
import time
import numpy as np
import pandas as pd
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import VisionDataset

class ChestXRay(VisionDataset):
    def __init__(
        self,
        root,
        train = True,
        transform = None
    ) -> None: 
        super(ChestXRay, self).__init__(root, transform=transform)
        self.train = train
        if train:
            self.img_dir = 'train'
        else:
            self.img_dir = 'test'
        
        self.normal_files = [f for f in listdir(join(root, self.img_dir, 'NORMAL')) if isfile(join(root, self.img_dir, 'NORMAL', f))]
        self.pneumonia_files = [f for f in listdir(join(root, self.img_dir, 'PNEUMONIA')) if isfile(join(root, self.img_dir, 'PNEUMONIA', f))]
        self.filenames = self.normal_files + self.pneumonia_files
        self.label = np.zeros(len(self.filenames), dtype=np.int)
        self.label[len(self.normal_files):] = 1
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        if (index >= 0 and index < len(self.normal_files)) or (index < 0 and index + len(self.pneumonia_files) < 0):
            im = Image.open(join(self.root, self.img_dir, 'NORMAL', self.filenames[index]))
        else:
            im = Image.open(join(self.root, self.img_dir, 'PNEUMONIA', self.filenames[index]))
        if self.transform is not None:
            im = self.transform(im)
        return im, self.label[index]

batch_size = 512

# Datasets
data_dir = os.environ['DATA_DIR'] 
xray_dir = os.path.join(data_dir, 'xray')
# TRAIN DATA
chest_train = ChestXRay(
    xray_dir,
    train=True,
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((150,150)),
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ]))


# TEST DATA
chest_val = ChestXRay(
    xray_dir,
    train=False,
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((150,150)),
        transforms.ToTensor(),
    ]))

train_sampler = torch.utils.data.RandomSampler(chest_train)

train_loader = DataLoader(
    chest_train,
    shuffle=False,
    batch_size=batch_size,
    pin_memory=True,
    num_workers=0,
    sampler=train_sampler)
    
val_loader = DataLoader(
    chest_val,
    shuffle=False,
    batch_size=624,
    pin_memory=True,
    num_workers=0)

model = nn.Sequential(
    nn.Conv2d(1, 10, (3,3)),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.MaxPool2d((2,2)),
    nn.Flatten(),
    nn.Linear(54760, 100),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(100, 2)
)
model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

# Timing for training
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%3dm %2ds' % (m, s)

# Printing during training
def print_train(epoch, batch_idx, time, acc, loss):
    print("E%4d:B%4d %s \t acc: %.6f \t loss: %.6f"%(
          epoch, batch_idx, time, acc, loss),
          flush=True)

train_losses = []
valid_losses = []

start = time.time()
model.train()
for epoch in range(1, 21):
    for batch_idx, batch in enumerate(train_loader, 1):
        inp = batch[0].cuda()
        tgt = batch[1].cuda()
        
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
           
        if batch_idx % 1 == 0:
            _, pred = torch.max(out, dim=1)
            train_acc = (pred == tgt).sum().float() / batch[0].size(0)
            print_train(epoch, batch_idx, timeSince(start), train_acc, loss.detach())
               
        train_losses.append(loss.item())
            
    # Validation and checkpoint
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_acc = 0.
        for vbatch_idx, vbatch in enumerate(val_loader, 1):
            inp = vbatch[0].cuda()
            tgt = vbatch[1].cuda()
            
            out = model(inp)
            losses = criterion(out, tgt).detach()
            valid_loss += losses
            
            _, pred = torch.max(out, dim=1)
            valid_acc += (pred == tgt).sum().float()
            
        valid_acc /= len(chest_val)
        
    print("")
    print("VALIDATION:")
    print_train(epoch, 1, timeSince(start), valid_acc, valid_loss)
    print("")
    valid_losses.append(valid_loss.item())
    model.train()

model_dir = os.environ['MODEL_DIR'] 
model_path = os.path.join(model_dir, 'xray', 'cnn.pt')
torch.save({
    'epochs': epoch-1,
    'batch_size': batch_size,
    'lr': 3e-3,
    'weight_decay': 0.01,
    'train_losses': train_losses,
    'valid_losses': valid_losses,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, model_path)
