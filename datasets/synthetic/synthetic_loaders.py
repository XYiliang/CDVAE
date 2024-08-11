# coding = utf-8
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_loaders(cfg):
    if not cfg.is_transfer:
        data = pd.read_csv('datasets/synthetic/syn_source.csv', index_col=0).iloc[:20000, :]
        real_cf_data = pd.read_csv('datasets/synthetic/syn_source_counterfactual.csv', index_col=0).iloc[:20000, :]
        scaler = StandardScaler()
        scaler.fit(data[['xn1', 'xa1', 'xa2']].values)
        data_train, data_test, _ = preprocess_synthetic(data, scaler)
        _, real_cf_data_test, _ = preprocess_synthetic(real_cf_data, scaler)

        train_loader, dim_dict = synthetic_train_loader_single(data_train, cfg.batch_size)
        test_loader = synthetic_test_loader_single(data_test, cfg.batch_size)
        real_cf_test_loader = synthetic_test_loader_single(real_cf_data_test, cfg.batch_size)
        loader_pack = {'train_loader': train_loader, 'test_loader': test_loader,
                       'real_cf_test_loader': real_cf_test_loader, 'dim_dict': dim_dict}

        return loader_pack

    else:
        source_data = pd.read_csv('datasets/synthetic/syn_source.csv', index_col=0).iloc[:15000, :]
        target_data = pd.read_csv('datasets/synthetic/b1(16)b2(5)xn1(mean45).csv', index_col=0).iloc[:15000, :]

        scaler = StandardScaler()
        total = pd.concat([source_data, target_data], axis=0)
        scaler.fit(total[['xn1', 'xa1', 'xa2']].values)
        data_train_src, data_test_src, src_size = preprocess_synthetic(source_data, scaler)
        data_train_trg, data_test_trg, trg_size = preprocess_synthetic(target_data, scaler)

        if trg_size < src_size:
            src_batch_size = cfg.batch_size
            trg_batch_size = int((cfg.batch_size / src_size) * trg_size.shape[0])
        else:
            src_batch_size = trg_batch_size = cfg.batch_size

        src_train_loader, trg_train_loader, dim_dict = synthetic_train_loader_transfer(data_train_src,
                                                                                       data_train_trg,
                                                                                       src_batch_size,
                                                                                       trg_batch_size)
        src_test_loader, trg_test_loader = synthetic_test_loader_transfer(data_test_src,
                                                                          data_test_trg,
                                                                          src_batch_size,
                                                                          trg_batch_size)

        loader_pack = {
            'src_train_loader': src_train_loader,
            'trg_train_loader': trg_train_loader,
            'src_test_loader': src_test_loader,
            'trg_test_loader': trg_test_loader,
            'dim_dict':dim_dict
        }

        return loader_pack


def synthetic_train_loader_single(data_train, batch_size, shuffle_loader=True):
    xnc, xnb, xac, xab, a, y = data_train
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
        'xab': 0,
        'a': 1,
        'y': 1
    }
    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return train_loader, dim_dict


def synthetic_test_loader_single(data_test, batch_size, shuffle_loader=False):
    xnc, xnb, xac, xab, a, y = data_test
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return test_loader


def synthetic_train_loader_transfer(data_train_src, data_train_trg, src_batch_size, trg_batch_size,
                                    shuffle_loader=True):
    xnc, xnb, xac, xab, a, y = data_train_src
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
        'xab': 0,
        'a': 1,
        'y': 1
    }
    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    src_train_loader = DataLoader(dataset, batch_size=src_batch_size, shuffle=shuffle_loader)

    xnc, xnb, xac, xab, a, _ = data_train_trg
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y_dummy = torch.zeros_like(a).type(torch.FloatTensor)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y_dummy)
    trg_train_loader = DataLoader(dataset, batch_size=trg_batch_size, shuffle=shuffle_loader)

    return src_train_loader, trg_train_loader, dim_dict


def synthetic_test_loader_transfer(data_test_src, data_test_trg, src_batch_size, trg_batch_size, shuffle_loader=False):
    xnc, xnb, xac, xab, a, y = data_test_src
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    src_test_loader = DataLoader(dataset, batch_size=src_batch_size, shuffle=shuffle_loader)

    xnc, xnb, xac, xab, a, y = data_test_trg
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor).view(-1, 1)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor).view(-1, 1)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    trg_test_loader = DataLoader(dataset, batch_size=trg_batch_size, shuffle=shuffle_loader)

    return src_test_loader, trg_test_loader


def preprocess_synthetic(data, scaler):
    data[['xn1', 'xa1', 'xa2']] = scaler.transform(data[['xn1', 'xa1', 'xa2']].values)
    xnc = data['xn1'].values.reshape(-1, 1)
    xnb = data['xn2'].values.reshape(-1, 1)
    xac = data[['xa1', 'xa2']].values
    xab = np.zeros([data.shape[0], 1])
    a = data['a'].values.reshape(-1, 1)
    y = data['y'].values.reshape(-1, 1)
    sample_size = y.shape[0]

    xnc_train, xnc_test, xnb_train, xnb_test, xac_train, xac_test, xab_train, xab_test, a_train, a_test, y_train, \
        y_test = train_test_split(xnc, xnb, xac, xab, a, y, random_state=1, train_size=0.8)
    data_train = xnc_train, xnb_train, xac_train, xab_train, a_train, y_train
    data_test = xnc_test, xnb_test, xac_test, xab_test, a_test, y_test

    return data_train, data_test, sample_size
