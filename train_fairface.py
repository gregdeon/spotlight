import os
import sys
import math
import time
import pandas
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parameter import Parameter
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models


def main():
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)
    
    # Parse arguments
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true')
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false')
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int)
    parser.add_argument('-r', '--relative', default=0, type=int)
    parser.add_argument('--master_addr', default='localhost')
    parser.add_argument('--master_port', default='12345')
    parser.set_defaults(checkpoint=True)
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.MODEL_SAVEFILE_NAME = "fairface_resnet18.pt"
    args.CHECKPOINT_PATH = args.checkpoint_dir+args.MODEL_SAVEFILE_NAME
    args.BEST_CHECKPOINT_PATH = args.checkpoint_dir+"best_"+args.MODEL_SAVEFILE_NAME
    args.MODEL_PATH = "./models/"+args.MODEL_SAVEFILE_NAME
    args.PRINT_EVERY = 20
    args.HYPERP = {
        "epochs": 20,
        "batch_size": 64,
        "num_workers": 0
    }
    
    print("CHECKPOINT_PATH:", args.CHECKPOINT_PATH, flush=True)
    
    # Multiprocessing
    
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train, nprocs=int(args.gpus), args=(args,))
    
    

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
def print_valid(epoch, time, acc, loss, acc_race, loss_race):
    print("E%4d %s \t acc: %.6f \t loss: %.6f"%(
          epoch, time, acc, loss),
          flush=True)
    print("Acc by race:", acc_race, flush=True)
    print("Loss by race:", loss_race, flush=True)
    


class FairFace(VisionDataset):

    age_classes = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
    gender_classes = ["Male", "Female"]
    race_classes = ["White", "Black", "Indian", "East Asian", "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]

    def __init__(
        self,
        root,
        train = True,
        transform = None
    ) -> None: 
        super(FairFace, self).__init__(root, transform=transform)
        self.train = train
        if train:
            label_csv = pandas.read_csv(os.path.join(root, "fairface_label_train.csv"), delim_whitespace=False)
        else:
            label_csv = pandas.read_csv(os.path.join(root, "fairface_label_val.csv"), delim_whitespace=False)
        
        self.filenames = label_csv.file.values
        self.age = label_csv.age.values
        self.age = np.vectorize(self.age_classes.index)(self.age)
        self.gender = label_csv.gender.values
        self.gender = np.vectorize(self.gender_classes.index)(self.gender)
        self.race = label_csv.race.values
        self.race = np.vectorize(self.race_classes.index)(self.race)
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        im = Image.open(os.path.join(self.root, self.filenames[index]))
        if self.transform is not None:
            im = self.transform(im)
        return im, self.age[index], self.gender[index], self.race[index]



def train(gpu, args):

    torch.set_printoptions(linewidth=10000)
    
    ########## Distributed setup #########
    
    rank = args.relative * args.gpus + gpu
    addr = 'tcp://' + os.getenv('MASTER_ADDR') + ':' + os.getenv('MASTER_PORT')
    dist.init_process_group(backend='nccl', init_method=addr, world_size=args.world_size, rank=rank)
    if rank == 0:
        print("torch", torch.__version__, flush=True)
        print("torchvision", torchvision.__version__, flush=True)
        print(args.HYPERP, flush=True)
    
    ########## Data loaders #########
    data_dir = os.environ['DATA_DIR'] 
    fairface_dir = os.path.join(data_dir, 'fairface')
    
    # Train #
    fairface_train = FairFace(
        fairface_dir,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4844, 0.3597, 0.3050), (0.2565, 0.2224, 0.2162))
        ]))
    #fairface_train_small = torch.utils.data.Subset(fairface_train, list(range(1024)))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        fairface_train,
    	num_replicas=args.world_size,
    	rank=rank,
        shuffle=True)
    
    train_loader = DataLoader(
        fairface_train,
        batch_size=args.HYPERP["batch_size"],
        pin_memory=True,
        num_workers=0,
        sampler=train_sampler)
    
    # Validation #
    
    fairface_valid = FairFace(
        fairface_dir,
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4844, 0.3597, 0.3050), (0.2565, 0.2224, 0.2162))
        ]))
    #fairface_valid_small = torch.utils.data.Subset(fairface_valid, list(range(1024)))
                            
    valid_loader = DataLoader(
        fairface_valid,
        batch_size=args.HYPERP["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=True)
    
    if rank == 0:
        print("Train size:", len(fairface_train), flush=True)
        print("Valid size:", len(fairface_valid), flush=True)
    
    ######### Model setup #########
    
    model = models.resnet34(pretrained=False, progress=False)
    model.fc = nn.Linear(512, 2) # Replace final linear layer
    model.to(gpu)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    if rank == 0 and args.checkpoint:
        if not os.path.isfile(args.CHECKPOINT_PATH):
            torch.save({
                'current_epoch': 1,
                'step': 1,
                'best_loss': 1e5,
                'train_losses': [],
                'valid_losses': [],
                'valid_losses_race': [],
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'time': time.time(),
                'HYPERP': args.HYPERP
            }, args.CHECKPOINT_PATH)

        checkpoint = torch.load(args.CHECKPOINT_PATH, map_location="cuda:"+str(gpu))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.current_epoch = checkpoint['current_epoch']
        args.start = checkpoint['time']
        args.step = checkpoint['step']
        args.best_loss = checkpoint['best_loss']
        args.train_losses = checkpoint['train_losses']
        args.valid_losses = checkpoint['valid_losses']
        args.valid_losses_race = checkpoint['valid_losses_race']
    else:
        args.current_epoch = 1
        args.step = 1
        args.start = time.time()
        args.best_loss = 1e5
        args.train_losses = []
        args.valid_losses = []
        args.valid_losses_race = []
        
    dist.barrier()
        
    ######### Training #########

    model.train()
    for epoch in range(args.current_epoch, args.HYPERP["epochs"] + 1):
        for batch_idx, batch in enumerate(train_loader, 1):
            inp = batch[0].cuda(gpu, non_blocking=True)
            tgt = batch[2].cuda(gpu, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(inp)
            loss = criterion(out, tgt).mean() 
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            #for i in range(len(optimizer.param_groups)):
            #    optimizer.param_groups[i]['lr'] = lr_scheduler(args)
            args.step += 1
            optimizer.step()
            
            if rank == 0 and batch_idx % args.PRINT_EVERY == 0:
                _, pred = torch.max(out, dim=1)
                train_acc = (pred == tgt).sum().float() / args.HYPERP["batch_size"]
                print_train(epoch, batch_idx, timeSince(args.start), train_acc, loss.detach())
                
            args.train_losses.append(loss.item())
            
        # Validation and checkpoint
        if rank == 0:
            model.eval()
            with torch.no_grad():
                valid_loss = 0.
                valid_acc = 0.
                valid_loss_race = torch.zeros(7)
                valid_acc_race = torch.zeros(7)
                counts_race = torch.zeros(7).int()
                for vbatch_idx, vbatch in enumerate(valid_loader, 1):
                    inp = vbatch[0].cuda(gpu, non_blocking=True)
                    tgt = vbatch[2].cuda(gpu, non_blocking=True)
                    
                    out = model(inp)
                    losses = criterion(out, tgt).detach()
                    valid_loss += losses.mean()
                    
                    _, pred = torch.max(out, dim=1)
                    valid_acc += (pred == tgt).sum().float()
                    
                    for i in range(args.HYPERP["batch_size"]):
                        valid_loss_race[vbatch[3][i]] += losses[i]
                        counts_race[vbatch[3][i]] += 1
                        valid_acc_race[vbatch[3][i]] += (pred[i] == tgt[i]).float()
                
                valid_loss /= (len(fairface_valid) // args.HYPERP["batch_size"])
                valid_loss_race /= counts_race
                valid_acc /= len(fairface_valid)
                valid_acc_race /= counts_race
                    
            print_valid(epoch, timeSince(args.start), valid_acc, valid_loss, valid_acc_race, valid_loss_race)
            args.valid_losses.append(valid_loss.item())
            args.valid_losses_race.append(valid_loss_race)
            model.train()
            
            # Checkpoint
            if args.checkpoint:
                temp_path = os.path.join(args.checkpoint_dir, "temp.pt")
                torch.save({
                    'current_epoch': epoch,
                    'step': args.step,
                    'best_loss': args.best_loss,
                    'train_losses': args.train_losses,
                    'valid_losses': args.valid_losses,
                    'valid_losses_race': args.valid_losses_race,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'time': args.start,
                    'HYPERP': args.HYPERP,
                }, temp_path)
                os.replace(temp_path, args.CHECKPOINT_PATH)
                print("Wrote a checkpoint.", flush=True)
            
            if args.checkpoint and valid_loss < args.best_loss:
                args.best_loss = valid_loss
                temp_path = os.path.join(args.checkpoint_dir, "temp.pt")
                torch.save({
                    'current_epoch': epoch,
                    'step': args.step,
                    'best_loss': args.best_loss,
                    'train_losses': args.train_losses,
                    'valid_losses': args.valid_losses,
                    'valid_losses_race': args.valid_losses_race,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'time': args.start,
                    'HYPERP': args.HYPERP,
                }, temp_path)
                os.replace(temp_path, args.BEST_CHECKPOINT_PATH)
                print("New best checkpoint.", flush=True)
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()