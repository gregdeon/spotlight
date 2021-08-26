import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models

import math
import time
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

sys.path.append('/scratch/st-kevinlb-1/gregdeon/spotlight/')
from spotlight.datasets import *
from spotlight.adversary import *

# command line interface
# TODO:
# - spotlight type: need to debug non-spherical adversary...
# - weight scheme: regularize toward uniform? moving average? raw spotlight weights? (do these still apply with the mini-epoch plan?)
parser = argparse.ArgumentParser(description='Run spotlight adversary.')
parser.add_argument('--checkpoint_path',    type=str,   required=True, help='Path to store most recent model; best model has "_best" appended')
parser.add_argument('--results_path',       type=str,   required=True, help='Path to store diagnostic results')
parser.add_argument('--adversary', required=True, choices=['spotlight', 'loss', 'random'], help='Method used to select epoch weights')
parser.add_argument('--spotlight_size',     type=float, default=0.1,   help='Minimum dataset fraction required in spotlight')
parser.add_argument('--spotlight_strength', type=float, default=1.0,   help='Distance from uniform distribution (1.0 = only spotlight; 0.0 = only uniform)')
parser.add_argument('--spotlight_first_epoch', action='store_true',    help='Reweight the dataset on the first epoch')
parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adam',  help='Optimizer (for ML model, not spotlight)')
parser.add_argument('--learning_rate',      type=float, default=3e-4,  help='Optimizer learning rate')
parser.add_argument('--momentum',           type=float, default=0,     help='Optimizer momentum')
parser.add_argument('--weight_decay',       type=float, default=0,     help='Optimizer weight decay')
parser.add_argument('--l2_penalty',         type=float, default=0,     help='Coefficient for L2 regularization')
parser.add_argument('--num_epochs',         type=int,   required=True, help='Number of training epochs')

args = parser.parse_args()
print(args)

PRINT_EVERY = 20
HYPERP = {
    "epochs": args.num_epochs,
    "batch_size": 128,
    "optimizer": args.optimizer,
    "learning_rate": args.learning_rate,
    "momentum": args.momentum,
    "weight_decay": args.weight_decay,
    "num_workers": 0,
    "spotlight_min_weight_fraction": args.spotlight_size,
    "spotlight_strength": args.spotlight_strength,
    "spotlight_learning_rate": 1e-2,
    "spotlight_num_steps": 5000,
}

tmp_dir = os.environ['TMPDIR'] 
fairface_dir = os.path.join(tmp_dir, 'fairface')

# load datasets
fairface_train = FairFace(
    fairface_dir,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4844, 0.3597, 0.3050), (0.2565, 0.2224, 0.2162))
    ])
)

fairface_val = FairFace(
    fairface_dir,
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4844, 0.3597, 0.3050), (0.2565, 0.2224, 0.2162))
    ])
)

# debugging
# fairface_train = torch.utils.data.Subset(fairface_train, range(1024))
# fairface_val = torch.utils.data.Subset(fairface_val, range(128))

# full train loader: used for inference and training passes
train_loader = DataLoader(
    fairface_train,
    batch_size=HYPERP['batch_size'],
    shuffle=True,
    pin_memory=True,
)

# full val loader: used for validation passes
val_loader = DataLoader(
    fairface_val,
    batch_size=HYPERP['batch_size'],
    shuffle=False,
    pin_memory=True,
)

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
        
# Printing during validation
def print_valid(epoch, time, acc, loss):
    print("E%4d %s \t acc: %.6f \t loss: %.6f"%(
        epoch, time, acc, loss),
        flush=True)

# Add hook to capture hidden layer
hidden_layers = {}
def get_input(name):
    def hook(model, input, output):
        if name in hidden_layers:
            del hidden_layers[name]
        hidden_layers[name] = input[0].detach()
    return hook

# Set up model + hook
model = models.resnet18(pretrained=False, progress=False)
model.fc = nn.Linear(512, 2) 
model.cuda()
model.fc.register_forward_hook(get_input('last_layer'))

if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    raise ValueError('Unrecognized optimizer: %s' % args.optimizer)
criterion = nn.CrossEntropyLoss(reduction='none')

# Set up training variables + performance trackers
current_epoch = 1
step = 1
start = time.time()
best_loss = 1e5
train_losses = []
valid_losses = []

train_len = len(fairface_train)

# performance over time
# inference results (training set)
epoch_weights_inference = torch.full((HYPERP['epochs'], len(fairface_train)), -1.0)
epoch_accs_inference    = torch.full((HYPERP['epochs'], len(fairface_train)), -1.0)
epoch_losses_inference  = torch.full((HYPERP['epochs'], len(fairface_train)), -1.0)
# validation set results
epoch_weights_val      = torch.full((HYPERP['epochs'], len(fairface_val)), -1.0)
epoch_accs_val         = torch.full((HYPERP['epochs'], len(fairface_val)), -1.0)
epoch_losses_val       = torch.full((HYPERP['epochs'], len(fairface_val)), -1.0)
# training set results
epoch_accs_train       = torch.full((HYPERP['epochs'], len(fairface_train)), -1.0)
epoch_losses_train     = torch.full((HYPERP['epochs'], len(fairface_train)), -1.0)

######### Training #########

model.train()
for epoch in range(current_epoch, HYPERP["epochs"] + 1):
    # run inference (entire training set), including embeddings
    print('Running inference...')
    embeddings = torch.zeros(train_len, 512)
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            inp = batch[0].cuda()
            tgt = batch[2].cuda()
            batch_indices = batch[4]

            out = model(inp)
            losses = criterion(out, tgt).detach()

            _, pred = torch.max(out, dim=1)
            correct = (pred == tgt)

            epoch_losses_inference[epoch - 1, batch_indices] = losses.cpu()
            epoch_accs_inference[epoch - 1, batch_indices] = correct.cpu().float()
            embeddings[batch_indices, :] = hidden_layers['last_layer'].cpu()
      
    # run inference spotlight
    losses = epoch_losses_inference[epoch - 1, :]
    min_weight = int(train_len * HYPERP['spotlight_min_weight_fraction'])
    barrier_min_x = 0.05 * min_weight
    barrier_x_schedule = np.geomspace(train_len - min_weight, barrier_min_x, HYPERP['spotlight_num_steps'])
    print('Running spotlight...')
    (final_weights, _, _, _, _) = run_adversary(
        embeddings = embeddings, 
        losses = losses,
        min_weight = min_weight,
        spherical_adversary = True,
        barrier_x_schedule = barrier_x_schedule,
        barrier_scale = 1,
        learning_rate = HYPERP['spotlight_learning_rate'],
        device = 'cuda'
    )
    epoch_weights_inference[epoch - 1, :] = torch.Tensor(final_weights)
    
    # get epoch weights:
    # - in spotlight mode: run spotlight; use output weights
    # - in random mode: use uniform weights
    uniform_weights = torch.full((train_len,), 1/train_len)
    if args.adversary == 'spotlight':
        if epoch > 1 or args.spotlight_first_epoch:
            weights_epoch = (args.spotlight_strength * torch.Tensor(final_weights)) + (1 - args.spotlight_strength) * uniform_weights
        else:
            weights_epoch = uniform_weights
    elif args.adversary == 'random':
        weights_epoch = uniform_weights
    else:
        raise ValueError('Unrecognized adversary type: %s' % args.adversary)

    # run training loop
    print('Training...')
    model.train()
    for batch_idx, batch in enumerate(train_loader, 1):
        inp = batch[0].cuda()
        tgt = batch[2].cuda()
        batch_indices = batch[4]
        batch_weights = weights_epoch[batch_indices].cuda()

        optimizer.zero_grad()
        out = model(inp)
        losses = criterion(out, tgt)
        loss = losses @ batch_weights * train_len / HYPERP['batch_size']
        
        # add l2 penalty
        params_l2 = torch.tensor(0.0).cuda()
        for param in model.parameters():
            params_l2 += torch.norm(param)**2
        loss += args.l2_penalty * params_l2
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        step += 1
        optimizer.step()

        _, pred = torch.max(out, dim=1)
        correct = (pred == tgt)

        epoch_losses_train[epoch - 1, batch_indices] = losses.detach().cpu()
        epoch_accs_train[epoch - 1, batch_indices] = correct.detach().cpu().float()

        if batch_idx % PRINT_EVERY == 0:
            train_acc = (pred == tgt).sum().float() / HYPERP["batch_size"]
            print_train(epoch, batch_idx, timeSince(start), train_acc, loss.detach())

        train_losses.append(loss.item())


    # run validation (entire validation set) and checkpoint
    print('Validating...')
    embeddings = torch.full((len(fairface_val), 512), 0.0)
    model.eval()
    with torch.no_grad():
        valid_loss = 0.
        valid_acc = 0.
        for vbatch_idx, vbatch in enumerate(val_loader, 1):
            inp = vbatch[0].cuda()
            tgt = vbatch[2].cuda()
            batch_indices = vbatch[4]

            out = model(inp)
            losses = criterion(out, tgt).detach()
            valid_loss += losses.sum()

            _, pred = torch.max(out, dim=1)
            correct = (pred == tgt)
            valid_acc += (pred == tgt).sum().float()

            epoch_losses_val[epoch - 1, batch_indices] = losses.cpu()
            epoch_accs_val[epoch - 1, batch_indices] = correct.cpu().float()
            embeddings[batch_indices] = hidden_layers['last_layer'].cpu()

        valid_loss /= len(fairface_val)
        valid_acc /= len(fairface_val)

    print_valid(epoch, timeSince(start), valid_acc, valid_loss)
    valid_losses.append(valid_loss.item())
    
    # run spotlight on validation set
    val_len = len(fairface_val)
    min_weight = int(val_len * HYPERP['spotlight_min_weight_fraction'])
    barrier_min_x = 0.05 * min_weight
    barrier_x_schedule = np.geomspace(val_len - min_weight, barrier_min_x, HYPERP['spotlight_num_steps'])
    print('Running spotlight...')
    (final_weights, _, _, _, _) = run_adversary(
        embeddings = embeddings, 
        losses = epoch_losses_val[epoch - 1],
        min_weight = min_weight,
        spherical_adversary = True,
        barrier_x_schedule = barrier_x_schedule,
        barrier_scale = 1,
        learning_rate = HYPERP['spotlight_learning_rate'],
        device = 'cuda'
    )
    epoch_weights_val[epoch - 1, :] = torch.Tensor(final_weights)

    # Save results
    torch.save({
        'epoch_weights_inference': epoch_weights_inference,
        'epoch_accs_inference': epoch_accs_inference,
        'epoch_losses_inference': epoch_losses_inference,
        'epoch_weights_val': epoch_weights_val,
        'epoch_accs_val': epoch_accs_val,
        'epoch_losses_val': epoch_losses_val,
        'epoch_accs_train': epoch_accs_train,
        'epoch_losses_train': epoch_losses_train,
    }, args.results_path)
    print('Updated training results.')
    
    # Save model checkpoints
    torch.save({
        'current_epoch': epoch,
        'step': step,
        'best_loss': best_loss,
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'time': start,
        'HYPERP': HYPERP,
    }, args.checkpoint_path)
    print("Wrote a checkpoint.")

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save({
            'current_epoch': epoch,
            'step': step,
            'best_loss': best_loss,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'time': start,
            'HYPERP': HYPERP,
        }, os.path.splitext(args.checkpoint_path)[0] + "_best.pt")
        print("New best checkpoint.")
