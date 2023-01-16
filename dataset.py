import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import shutil
import os.path

import json


class My_Dataset(Dataset):
    def __init__(self, img_dir, split= "training", transform=None):
        self.img_dir = f"{img_dir}/{split}"
        self.labels=os.listdir(self.img_dir)
        self.labels_lenght=[len(os.listdir(f"{self.img_dir}/{label}")) for label in self.labels]
        self.transform = transform
        self.parameters = json.load(open("config.json"))

    def find_label(self, index, current_label=0):
        if(index> self.labels_lenght[current_label]-1):
            index-=self.labels_lenght[current_label]
            return self.find_label(index, current_label+1)
        else:
            return index, current_label

    def __len__(self):
        return np.array(self.labels_lenght).sum()
        
    def __getitem__(self, idx):
        index, label= self.find_label(idx)
     #   print(f"index{index}, label {label}")
        img_path = f"{self.img_dir}/{self.labels[label]}/{index}.jpg"
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        #label= torch.tensor([label], dtype=torch.int64)
        return image, label
             
        
class Datamodule(pl.LightningDataModule):
    def __init__(self, path, resolution):
        super().__init__()
        self.parameters = json.load(open("config.json"))
        self.cifar10_mean= (0.4914, 0.4822, 0.4465)
        self.cifar10_std = (0.2471, 0.2435, 0.2616)
        self.batch_size=self.parameters['batch-size']
        self.num_workers=self.parameters['workers']
        self.reshape_size=resolution
        self.path=path



        self.train_transform = transforms.Compose([
          #  transforms.RandomResizedCrop(32, scale=(parameter['scale'], 1.0), ratio=(1.0, 1.0)),
            transforms.Resize((self.reshape_size, self.reshape_size)),
           # transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandAugment(num_ops=self.parameters['ra-n'], magnitude=self.parameters['ra-m']),
            transforms.ColorJitter(self.parameters['jitter'], self.parameters['jitter'], self.parameters['jitter']),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar10_mean, self.cifar10_std),
            transforms.RandomErasing(p=self.parameters['reprob'])
            ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.reshape_size, self.reshape_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.cifar10_mean, self.cifar10_std)
            ])

        self.training_dataset= My_Dataset(img_dir=self.path,split="training", transform=self.train_transform)
        self.evaluate_dataset= My_Dataset(img_dir=self.path,split="evaluation", transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, 
                            batch_size= self.batch_size,
                            shuffle=True, 
                            num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.evaluate_dataset, 
                            batch_size= self.batch_size,
                           # shuffle=True, 
                            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.evaluate_dataset, 
                            batch_size= self.batch_size,
                     
                            num_workers=self.num_workers)        
