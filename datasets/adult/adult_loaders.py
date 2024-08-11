# coding = utf-8
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import onehot

country_continent_mapping = {
    'United-States': 'US',
    'Canada': 'America',
    'England': 'Europe',
    'Germany': 'Europe',
    'France': 'Europe',
    'Italy': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Scotland': 'Europe',
    'Yugoslavia': 'Europe',
    'Holand-Netherlands': 'Europe',
    'India': 'Asia',
    'China': 'Asia',
    'Japan': 'Asia',
    'Vietnam': 'Asia',
    'Taiwan': 'Asia',
    'Iran': 'Asia',
    'Philippines': 'Asia',
    'Cambodia': 'Asia',
    'Laos': 'Asia',
    'Thailand': 'Asia',
    'Hong': 'Asia',
    'South': 'Asia',
    'Other': 'Other',
    'Cuba': 'America',
    'Jamaica': 'America',
    'Mexico': 'America',
    'Puerto-Rico': 'America',
    'Honduras': 'America',
    'Guatemala': 'America',
    'El-Salvador': 'America',
    'Dominican-Republic': 'America',
    'Haiti': 'America',
    'Peru': 'America',
    'Outlying-US(Guam-USVI-etc)': 'US',
    'Trinadad&Tobago': 'America',
    'Greece': 'Europe',
    'Nicaragua': 'America',
    'Ireland': 'Europe',
    'Hungary': 'Europe',
    'Ecuador': 'America',
    'Columbia': 'America',
}


def make_adult_loaders(cfg):
    adult_head = ["Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                  "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
                  "hours per week", "native-country", "income"]
    adult_data = pd.read_csv('datasets/adult/uci_adult/adult.data', names=adult_head, sep=', ', engine='python')
    adult_test = pd.read_csv('datasets/adult/uci_adult/adult.test', skiprows=1, names=adult_head, sep=', ', engine='python')
    data_train, data_test = process_adult_data(adult_data, adult_test, cfg)

    train_loader, dim_dict = adult_trainloader(data_train, cfg.batch_size)
    test_loader = adult_testloader(data_test, cfg.batch_size)

    loader_pack = {'train_loader': train_loader, 'test_loader': test_loader, 'dim_dict': dim_dict}

    return loader_pack


def adult_trainloader(data_train, batch_size, shuffle_loader=True):
    xnc, xnb, xac, xab, a, y = data_train

    xnc = torch.from_numpy(xnc).type(torch.FloatTensor)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)


    dim_dict = {
        'xnc': xnc.shape[1],
        'xnb': xnb.shape[1],
        'xac': xac.shape[1],
        'xab': xab.shape[1],
        'a': 1,
        'y': 1
    }

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return train_loader, dim_dict


def adult_testloader(data_test, batch_size, shuffle_loader=False):
    xnc, xnb, xac, xab, a, y = data_test
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)

    dataset = TensorDataset(xnc, xnb, xac, xab, a, y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return test_loader


def process_adult_data(adult_data, adult_test, cfg):
    scaler = StandardScaler()

    adult_total = pd.concat([adult_data, adult_test], axis=0)
    adult_total = adult_total[adult_total["workclass"] != '?']
    adult_total = adult_total[adult_total["occupation"] != '?']
    adult_total = adult_total[adult_total["native-country"] != '?']
    adult_total['native-country'] = adult_total['native-country'].map(country_continent_mapping)
    adult_total['gender'] = adult_total['gender'].map({'Male': 1, 'Female': 0})
    adult_total['income'] = adult_total['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}).values.reshape(-1, 1)

    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    for col in category_col:
        b, c = np.unique(adult_total[col], return_inverse=True)
        adult_total[col] = c

    adult_total[['Age', 'hours per week', 'capital gain', 'capital loss', 'education-num']] =\
        scaler.fit_transform(adult_total[['Age', 'hours per week', 'capital gain', 'capital loss', 'education-num']].values)

    race = onehot(adult_total['race'].values, 5)
    age = adult_total['Age'].values.reshape(-1, 1)
    native = onehot(adult_total['native-country'].values, 4)
    sex = adult_total['gender'].values.reshape(-1, 1)
    edu = onehot(adult_total['education'].values, 16)
    edu_num = adult_total['education-num'].values.reshape(-1, 1)
    gain = adult_total['capital gain'].values.reshape(-1, 1)
    loss = adult_total['capital loss'].values.reshape(-1, 1)
    hour = adult_total['hours per week'].values.reshape(-1, 1)
    work_class = onehot(adult_total['workclass'].values, 7)
    marital = onehot(adult_total['marital-status'].values, 7)
    occp = onehot(adult_total['occupation'].values, 14)
    relp = onehot(adult_total['relationship'].values, 6)
    income = adult_total['income'].values.reshape(-1, 1)

    xnc = age
    xnb = np.concatenate([race, native], axis=1)
    xac = np.concatenate([hour, edu_num, gain, loss], axis=1)
    xab = np.concatenate([edu, work_class, marital, occp, relp], axis=1)  # 50
    a = sex
    y = income

    xnc_train, xnc_test, xnb_train, xnb_test, xac_train, xac_test, xab_train, xab_test, a_train, a_test,\
        y_train, y_test = train_test_split(xnc, xnb, xac, xab, a, y, random_state=cfg.seed, train_size=0.8)

    data_train = (xnc_train, xnb_train, xac_train, xab_train, a_train, y_train)
    data_test = (xnc_test, xnb_test, xac_test, xab_test, a_test, y_test)

    return data_train, data_test

