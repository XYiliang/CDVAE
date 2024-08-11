# coding = utf-8
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


def make_law_loaders(cfg):
    law_df = pd.read_csv('datasets/law_school/law_school_scaled.csv', index_col=0)
    train_df, test_df = train_test_split(law_df, train_size=0.8, random_state=cfg.seed)
    train_loader, dim_dict = law_trainloader(train_df, cfg.batch_size)
    test_loader = law_testloader(test_df, cfg.batch_size)

    loader_pack = {'train_loader': train_loader, 'test_loader': test_loader, 'dim_dict': dim_dict}

    return loader_pack


def law_trainloader(train_df, batch_size, shuffle_loader=True):
    xnc = np.zeros([train_df.shape[0], 1])
    xnb = train_df[['Amerindian', 'Asian', 'Black', 'Hispanic', 'Mexican', 'Other', 'Puertorican', 'White']].values
    xac = train_df[['LSAT', 'UGPA']].values
    xab = np.zeros([train_df.shape[0], 1])
    a = train_df['sex'].values
    y = train_df['ZFYA'].values

    xnc = torch.from_numpy(xnc).type(torch.FloatTensor)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dim_dict = {
        'xnc': 0,
        'xnb': xnb.shape[1],
        'xac': xac.shape[1],
        'xab': 0,
        'a': 1,
        'y': 1
    }

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return train_loader, dim_dict


def law_testloader(test_df, batch_size, shuffle_loader=False):
    xnc = np.zeros([test_df.shape[0], 1])
    xnb = test_df[['Amerindian', 'Asian', 'Black', 'Hispanic', 'Mexican', 'Other', 'Puertorican', 'White']].values
    xac = test_df[['LSAT', 'UGPA']].values
    xab = np.zeros([test_df.shape[0], 1])
    a = test_df['sex'].values
    y = test_df['ZFYA'].values

    xnc = torch.from_numpy(xnc).type(torch.FloatTensor)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return test_loader
