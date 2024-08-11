# coding = utf-8
import numpy as np
import torch
from folktables import ACSDataSource, ACSIncome
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.myDataset import MyDatasetY, MyDatasetNoY
from torch.utils.data import DataLoader
from utils_test_fair import onehot, binning
import joblib

state_list = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
              'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
              'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
              'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
              'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']

_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                'PR': '72'}


def make_new_adult_loader(cfg):
    if not cfg.is_transfer:
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                    root_dir='datasets/ACSIncome/data')
        acs_data = data_source.get_data(states=[cfg.state], download=False)
        features, label, group = ACSIncome.df_to_pandas(acs_data)

        scaler = StandardScaler()
        scaler.fit(features[['AGEP', 'WKHP']].values)
        data_train, data_test = preprocess_new_adult(features, label, scaler)
        train_loader, dim_dict = new_adult_trainloader(data_train, cfg.batch_size)
        test_loader = new_adult_testloader(data_test, cfg.batch_size)

        return train_loader, test_loader, dim_dict
    else:
        src_state, trg_state = cfg.source_state, cfg.target_state
        data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                    root_dir='datasets/ACSIncome/data')

        src_data = data_source.get_data(states=src_state, download=False)
        src_features, src_label, src_group = ACSIncome.df_to_pandas(src_data)
        trg_data = data_source.get_data(states=trg_state, download=False)
        trg_features, trg_label, trg_group = ACSIncome.df_to_pandas(trg_data)

        scaler = StandardScaler()
        # scaler_age = StandardScaler()
        # scaler_wkhp = StandardScaler()
        feature_total = pd.concat([src_features, trg_features], axis=0)
        scaler.fit(feature_total[['AGEP', 'WKHP']].values)
        # scaler_age.fit(feature_total['AGEP'].values.reshape(-1, 1))
        # scaler_wkhp.fit(feature_total['WKHP'].values.reshape(-1, 1))
        # joblib.dump(scaler_age, 'NewAdultStandardscalerAge.pkl')
        # joblib.dump(scaler_wkhp, 'NewAdultStandardscalerWKHP.pkl')

        data_train_src, data_test_src = preprocess_new_adult(src_features, src_label, scaler, cfg)
        data_train_trg, data_test_trg = preprocess_new_adult(trg_features, trg_label, scaler, cfg)

        # To ensure source loader and target loader have same length
        if trg_features.shape[0] < src_features.shape[0]:
            src_batch_size = cfg.batch_size
            trg_batch_size = int((cfg.batch_size / src_features.shape[0]) * trg_features.shape[0])
        else:
            src_batch_size = trg_batch_size = cfg.batch_size

        src_train_loader, trg_train_loader, dim_dict = new_adult_trainloader_transfer(data_train_src,
                                                                                      data_train_trg,
                                                                                      src_batch_size,
                                                                                      trg_batch_size)
        src_test_loader, trg_test_loader = new_adult_testloader_transfer(data_test_src,
                                                                         data_test_trg,
                                                                         cfg.batch_size,
                                                                         trg_batch_size)

        return src_train_loader, trg_train_loader, src_test_loader, trg_test_loader, dim_dict


def new_adult_trainloader(data_train, batch_size, shuffle_loader=True):
    xnc, xnb, xac, xab, a, y = data_train
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dim_dict = {
        'xnc': 1,
        'xnb': xnb.shape[1],
        'xac': 1,
        'xab': xab.shape[1],
        'a': 1,
        'y': 1
    }

    dataset = MyDatasetY(xnc, xnb, xac, xab, a, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return train_loader, dim_dict


def new_adult_testloader(data_test, batch_size, shuffle_loader=False):
    xnc, xnb, xac, xab, a, y = data_test
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)
    dataset = MyDatasetY(xnc, xnb, xac, xab, a, y)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_loader)

    return test_loader


def new_adult_trainloader_transfer(data_train_src, data_train_trg, src_batch_size, trg_batch_size, shuffle_loader=True):
    xnc, xnb, xac, xab, a, y = data_train_src
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dim_dict = {'xnc': 1, 'xnb': xnb.shape[1], 'xac': 1, 'xab': xab.shape[1], 'a': 1, 'y': 1}
    dataset = MyDatasetY(xnc, xnb, xac, xab, a, y)
    src_train_loader = DataLoader(dataset, batch_size=src_batch_size, shuffle=shuffle_loader)

    xnc, xnb, xac, xab, a, y = data_train_trg
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)

    dataset = MyDatasetNoY(xnc, xnb, xac, xab, a)
    trg_train_loader = DataLoader(dataset, batch_size=trg_batch_size, shuffle=shuffle_loader)

    return src_train_loader, trg_train_loader, dim_dict


def new_adult_testloader_transfer(data_test_src, data_test_trg, src_batch_size, trg_batch_size, shuffle_loader=False):
    xnc, xnb, xac, xab, a, y = data_test_src
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)
    y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1)

    dataset = MyDatasetY(xnc, xnb, xac, xab, a, y)
    src_test_loader = DataLoader(dataset, batch_size=src_batch_size, shuffle=shuffle_loader)

    xnc, xnb, xac, xab, a, y = data_test_trg
    xnc = torch.from_numpy(xnc).type(torch.FloatTensor).view(-1, 1)
    xnb = torch.from_numpy(xnb).type(torch.FloatTensor)
    xac = torch.from_numpy(xac).type(torch.FloatTensor).view(-1, 1)
    xab = torch.from_numpy(xab).type(torch.FloatTensor)
    a = torch.from_numpy(a).type(torch.FloatTensor).view(-1, 1)

    dataset = MyDatasetY(xnc, xnb, xac, xab, a, y)
    trg_test_loader = DataLoader(dataset, batch_size=trg_batch_size, shuffle=shuffle_loader)

    return src_test_loader, trg_test_loader


def preprocess_new_adult(features, label, scaler, cfg=None):
    features[['AGEP', 'WKHP']] = scaler.transform(features[['AGEP', 'WKHP']].values)
    features['SEX'] = features['SEX'].map(lambda x: 1 if x == 1 else 0).astype(float)
    label = label.map(lambda x: 1 if x is True else 0).astype(float)

    # OCCP bins are ['MGR', 'BUS', 'FIN', 'CMM', 'ENG', 'SCI', 'CMS', 'LGL', 'EDU', 'ENT', 'MED', 'HLS', 'PRT', 'EAT',
    #                'CLN', 'PRS', 'SAL', 'OFF', 'FFF', 'CON', 'EXT', 'PPR', 'PRD', 'TRN', 'MIL', 'UNEMPLOYED']
    occp_bin = [440, 750, 960, 1240, 1560, 1980, 2060, 2180, 2555, 2920, 3550, 3655, 3960, 4160, 4255, 4655, 4965,
                5940, 6130, 6765, 6950, 7640, 8990, 9760, 9830]
    # POBP bins are ['US', 'Europe', 'Asia', 'America', 'Africa', 'Oceania', 'Other']
    pob_bin = [78, 169, 254, 399, 469, 527]

    # The values of features SCHL, COW, MAR, RACE start with 1 not 0
    occp = onehot(binning(features['OCCP'].values, occp_bin), len(occp_bin)+1)
    pobp = onehot(binning(features['POBP'].values, pob_bin), len(pob_bin)+1)
    schl = onehot(features['SCHL'].values.astype(int) - 1, 24)
    cow = onehot(features['COW'].values.astype(int) - 1, 9)
    mar = onehot(features['MAR'].values.astype(int) - 1, 5)
    relp = onehot(features['RELP'].values.astype(int), 18)
    race = onehot(features['RAC1P'].values.astype(int) - 1, 9)

    xnc = features['AGEP'].values.reshape(-1, 1)
    xnb = np.concatenate([race, pobp], axis=1)
    xac = features['WKHP'].values.reshape(-1, 1)
    xab = np.concatenate([schl, cow, mar, occp, relp], axis=1)
    a = features['SEX'].values
    y = label.values

    seed = 42 if cfg is None else cfg.seed
    xnc_tr, xnc_te, xnb_tr, xnb_te, xac_tr, xac_te, xab_tr, xab_te, a_tr, a_te, y_tr, y_te = \
        train_test_split(xnc, xnb, xac, xab, a, y, train_size=0.8, random_state=seed)
    data_train = xnc_tr, xnb_tr, xac_tr, xab_tr, a_tr, y_tr
    data_test = xnc_te, xnb_te, xac_te, xab_te, a_te, y_te

    return data_train, data_test



