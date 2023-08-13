import os
from tkinter import Label
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

import scanpy as sc
import random
from module.process_data import data_process,split_data
from sklearn.preprocessing import OneHotEncoder

class ATACDataset(Dataset):
    def __init__(self, adata):
        super(ATACDataset,self).__init__()
        self.X = torch.from_numpy(adata.X.copy())
        self.domain = torch.from_numpy(adata.obs['domain'].values.astype(np.float32))
        self.shape = adata.shape
        self.index = torch.from_numpy(np.array(range(len(adata)),dtype=np.int64))
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # x = self.X[idx].toarray().squeeze()
        x = self.X[idx]
        domain = self.domain[idx]
#         return x, domain_id, idx
        return x,domain

class LabelDataset(ATACDataset):
    def __init__(self, adata):
        super(LabelDataset,self).__init__(adata)
        self.label = torch.from_numpy(adata.obs["Label"].values)
    
    def __getitem__(self, idx):
        # x = self.X[idx].toarray().squeeze().astype(np.float32)
        x = self.X[idx]
#         return x, domain_id, idx
        l = self.label[idx]
        domain = self.domain[idx]
        index = self.index[idx]
        return x,l,domain,index

class OneHotDataset(ATACDataset):
    def __init__(self, adata, OneHotLabel):
        super(OneHotDataset,self).__init__(adata)
        self.label = torch.from_numpy(np.array(OneHotLabel,dtype=np.float32))
    
    def __getitem__(self, idx):
        # x = self.X[idx].toarray().squeeze().astype(np.float32)
        x = self.X[idx]
#         return x, domain_id, idx
        l = self.label[idx,:]
        domain = self.domain[idx]
        index = self.index[idx]
        return x,l,domain,index
    
class WeightDataset(ATACDataset):
    def __init__(self, adata, OneHotLabel,weight):
        super(WeightDataset,self).__init__(adata)
        self.label = torch.from_numpy(np.array(OneHotLabel,dtype=np.float32))
        self.weight = torch.from_numpy(np.array(weight,dtype=np.float32))
    
    def __getitem__(self, idx):
        # x = self.X[idx].toarray().squeeze().astype(np.float32)
        x = self.X[idx]
#         return x, domain_id, idx
        l = self.label[idx,:]
        domain = self.domain[idx]
        weight = self.weight[idx,:]
        index = self.index[idx]
        return x,l,domain,index,weight
    
class BatchDataset(LabelDataset):
    def __init__(self, adata):
        super(BatchDataset,self).__init__(adata)
        self.batch = adata.obs["batch"].values
        self.batch_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(self.batch).reshape(-1,1))
    
    def __getitem__(self, idx):
        x = self.X[idx].toarray().squeeze()
        # x = self.X[idx]
#         return x, domain_id, idx
        l = self.label[idx]
        b = self.batch[idx]
        b_onehot = self.batch_onehot[idx]
        return x,l,b,b_onehot

def load_train_dataset(adata,onehotlabel=None,batch_size = 128,shuffle = False,drop_last = False,num_workers = 0,sample = None,weight = None,device = 'cpu'):
    # adata = data_process(adata)
    if weight is not None:
        dataset = WeightDataset(adata,OneHotLabel=onehotlabel,weight = weight)
    elif onehotlabel is not None :
        dataset = OneHotDataset(adata,OneHotLabel=onehotlabel)
    else :
        dataset = LabelDataset(adata)

    trainloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sample,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return trainloader

def load_test_dataset(adata,onehotlabel=None,batch_size = 128,shuffle = False,drop_last = False,num_workers = 4):
    # adata = data_process(adata)
    dataset = ATACDataset(adata)

    testloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=drop_last, 
        shuffle=shuffle,
        num_workers=num_workers
    )

    return testloader