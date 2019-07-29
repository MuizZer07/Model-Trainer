import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
import PIL

class Preprocess:

    def __init__(self, df, batch_size, input_size):
        self.df = df
        self.dataset_sizes = {}
        self.train, self.val = self.data_partition()
        self.batch_size = batch_size
        self.input_size = input_size

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.RandomRotation(15, resample=PIL.Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.dataset_train = MyDataset(df_data=self.train, transform=self.data_transforms['train'])
        self.dataset_valid = MyDataset(df_data=self.val,transform=self.data_transforms['val'])

        self.dataloaders = {}
        self.dataloaders['train'] = DataLoader(dataset = self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.dataloaders['val'] = DataLoader(dataset = self.dataset_valid, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def data_partition(self):
        train, val = train_test_split(self.df, test_size=0.2)

        self.dataset_sizes['train'] = len(train)
        self.dataset_sizes['val'] = len(val)

        return train, val

    def get_data(self):
        return self.dataloaders

class MyDataset(Dataset):
    def __init__(self, df_data,transform=None):
        super().__init__()
        self.df = df_data.values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image, label = self.df[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, label
