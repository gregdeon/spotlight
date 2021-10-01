import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import torch

from torchvision.datasets import VisionDataset
from PIL import Image
from torch.utils.data import Dataset

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
            label_csv = pd.read_csv(os.path.join(root, "fairface_label_train.csv"), delim_whitespace=False)
        else:
            label_csv = pd.read_csv(os.path.join(root, "fairface_label_val.csv"), delim_whitespace=False)
        
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
        return im, self.age[index], self.gender[index], self.race[index], index


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
    
class Adult(Dataset):
    categorical_features = ["workclass", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    real_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    
    def __init__(
        self,
        root,
        train = True,
        normalize = True,
    ) -> None: 
        self.train = train
        
        df = pd.read_csv(root, sep='\t', compression='gzip')
        df = df.drop('fnlwgt', 1)
        
        if normalize:
            df[self.real_features] = (df[self.real_features] - df[self.real_features].mean()) / df[self.real_features].std()
        df = pd.get_dummies(df, columns=self.categorical_features)
        
        if train:
            df = df[:int(0.8 * len(df.index))]
        else:
            df = df[int(0.8 * len(df.index)):]
            
        self.data = torch.Tensor(df.drop('target', axis=1).values)
        self.label = torch.Tensor(df['target'].values).long()
        self.features = df.columns
        self.df = df
    
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index: int):
        return self.data[index], self.label[index]
    
class Wine(Dataset):
    def __init__(
        self,
        root,
        train = True,
        normalize = True,
    ) -> None: 
        self.train = train
        
        df = pd.read_csv(root, sep='\t', compression='gzip')
        df_data = df.drop('target', axis=1)
        df_target = df['target']
        
        if normalize:
            df_data[:] = (df_data[:] - df_data[:].mean()) / df_data[:].std()
        
        if train:
            df_data = df_data[:int(0.8 * len(df_data.index))]
            df_target = df_target[:int(0.8 * len(df_target.index))]
        else:
            df_data = df_data[int(0.8 * len(df.index)):]
            df_target = df_target[int(0.8 * len(df.index)):]
            
        self.data = torch.Tensor(df_data.values)
        self.label = torch.Tensor(df_target.values)
    
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index: int):
        return self.data[index], self.label[index]
    
class SST2(Dataset):
    split_fnames = {
        'train': 'train.csv',
        'val':   'val.csv',
        'test':  'test.csv',
    }
    
    def __init__(
        self,
        root,
        split = 'train',
    ) -> None:
        if split not in self.split_fnames.keys():
            valid_splits = ', '.join(self.split_fnames.keys())
            raise ValueError('Unrecognized split; valid splits are %s' % valid_splits)
        
        fname = os.path.join(root, self.split_fnames[split])
        df = pd.read_csv(fname)
        
        self.data = df['sentence'].values
        self.labels = df['label'].values
        
    def __len__(self):
        return self.data.size
    
    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]
