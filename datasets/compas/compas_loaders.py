# coding = utf-8
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def make_compas_loaders(cfg):
    compas_df = pd.read_csv('datasets/compas/compas_data_scaled.csv', index_col=0)
    train_df, test_df = train_test_split(compas_df, train_size=0.8, random_state=cfg.seed)
    train_loader, dim_dict = compas_trainloader(train_df, cfg.batch_size)
    test_loader = compas_testloader(test_df, cfg.batch_size)

    loader_pack = {'train_loader': train_loader, 'test_loader': test_loader, 'dim_dict': dim_dict}

    return loader_pack


def compas_trainloader(train_df, batch_size, shuffle_loader=True):
    xnc = train_df['age'].values
    xnb = train_df['sex'].values
    xac = train_df[['juv_misd_count', 'juv_fel_count', 'juv_other_count', 'priors_count']].values
    xab = train_df['c_charge_degree'].values
    a = train_df['race'].values
    y = train_df['two_year_recid'].values

    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dim_dict = {
        'xnc': 1,
        'xnb': 1,
        'xac': xac.shape[1],
        'xab': 1,
        'a': 1,
        'y': 1
    }

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return train_loader, dim_dict


def compas_testloader(test_df, batch_size, shuffle_loader=False):
    xnc = test_df['age'].values
    xnb = test_df[['sex']].values
    xac = test_df[['juv_misd_count', 'juv_fel_count', 'juv_other_count', 'priors_count']].values
    xab = test_df['c_charge_degree'].values
    a = test_df['race'].values
    y = test_df['two_year_recid'].values

    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return test_loader
