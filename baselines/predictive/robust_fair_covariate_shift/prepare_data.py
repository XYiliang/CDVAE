import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from project.CDVAE.datasets.ACSIncome.ACSIncome_loaders import preprocess_new_adult
from datasets.synthetic.synthetic_loaders import preprocess_synthetic
from folktables import ACSDataSource, ACSIncome


def prepare_new_adult():
    src_state = ['CA']
    trg_state = ['AL', 'AK', 'AZ', 'AR', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
                  'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
                  'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
                  'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
                  'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'PR']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                root_dir='./datasets/ACSIncome/data')

    src_data = data_source.get_data(states=src_state, download=False)
    src_features, src_label, src_group = ACSIncome.df_to_pandas(src_data)
    trg_data = data_source.get_data(states=trg_state, download=False)
    trg_features, trg_label, trg_group = ACSIncome.df_to_pandas(trg_data)

    scaler = StandardScaler()
    feature_total = pd.concat([src_features, trg_features], axis=0)
    scaler.fit(feature_total[['AGEP', 'WKHP']].values)
    data_train_src, data_test_src = preprocess_new_adult(src_features, src_label, scaler)
    data_train_trg, data_test_trg = preprocess_new_adult(trg_features, trg_label, scaler)
    test_trg_ratio_idx = (data_train_trg[0].shape[0], data_train_trg[0].shape[0]+data_test_trg[0].shape[0])
    src_data, trg_data = [], []
    for i in range(len(data_train_src)):
        src_data.append(np.concatenate([data_train_src[i], data_test_src[i]], axis=0))
        trg_data.append(np.concatenate([data_train_trg[i], data_test_trg[i]], axis=0))
    src_dataX = np.concatenate(src_data[:4], axis=1)
    trg_dataX = np.concatenate(trg_data[:4], axis=1)
    src_dataA, src_dataY = src_data[-2], src_data[-1].squeeze()
    trg_dataA, trg_dataY = trg_data[-2], trg_data[-1].squeeze()
    return src_dataA, src_dataY, src_dataX, trg_dataA, trg_dataY, trg_dataX, test_trg_ratio_idx


def prepare_synthetic():
    src_data = pd.read_csv('datasets/synthetic/syn_source.csv', index_col=0).iloc[:15000, :]
    trg_data = pd.read_csv('datasets/synthetic/b1(16)b2(5)xn1(mean45).csv', index_col=0).iloc[:15000, :]
    total = pd.concat([src_data, trg_data], axis=0)
    scaler = StandardScaler()
    scaler.fit(total[['xn1', 'xa1', 'xa2']].values)
    data_train_src, data_test_src, src_size = preprocess_synthetic(src_data, scaler)
    data_train_trg, data_test_trg, trg_size = preprocess_synthetic(trg_data, scaler)
    src_data, trg_data = [], []
    for i in range(len(data_train_src)):
        src_data.append(np.concatenate([data_train_src[i], data_test_src[i]], axis=0))
        trg_data.append(np.concatenate([data_train_trg[i], data_test_trg[i]], axis=0))
    src_dataX = np.concatenate(src_data[:3], axis=1)
    trg_dataX = np.concatenate(trg_data[:3], axis=1)
    src_dataA, src_dataY = src_data[-2].squeeze(), src_data[-1].squeeze()
    trg_dataA, trg_dataY = trg_data[-2].squeeze(), trg_data[-1].squeeze()
    test_trg_ratio_idx = (data_train_trg[0].shape[0], data_train_trg[0].shape[0]+data_test_trg[0].shape[0])
    return src_dataA, src_dataY, src_dataX, trg_dataA, trg_dataY, trg_dataX, test_trg_ratio_idx


def prepare_compas(normalized=True):

    filePath = "robust_fair_covariate_shift/dataset/IBM_compas/"
    dataA = pd.read_csv(
        filePath + "IBM_compas_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_compas_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_compas_X.csv", sep="\t", index_col=0)
    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_german(normalized=True):
    filePath = "robust_fair_covariate_shift/dataset/A,Y,X/IBM_german/"

    dataA = pd.read_csv(
        filePath + "IBM_german_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "IBM_german_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "IBM_german_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_drug(normalized=True):
    filePath = "robust_fair_covariate_shift/dataset/drug/"

    dataA = pd.read_csv(filePath + "drug_A.csv", sep="\t", index_col=0, header=None)
    dataY = pd.read_csv(filePath + "drug_Y.csv", sep="\t", index_col=0, header=None)
    dataX = pd.read_csv(filePath + "drug_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_arrhythmia(normalized=True):
    filePath = "robust_fair_covariate_shift/dataset/arrhythmia/"

    dataA = pd.read_csv(
        filePath + "arrhythmia_A.csv", sep="\t", index_col=0, header=None
    )
    dataY = pd.read_csv(
        filePath + "arrhythmia_Y.csv", sep="\t", index_col=0, header=None
    )
    dataX = pd.read_csv(filePath + "arrhythmia_X.csv", sep="\t", index_col=0)

    if normalized:
        dataX = normalize(dataX)
    return dataA.iloc[:, 0], dataY.iloc[:, 0], dataX


def prepare_newAdult(normalized=True):
    pass


def normalize(X):
    for c in list(X.columns):
        if X[c].min() < 0 or X[c].max() > 1:
            mu = X[c].mean()
            s = X[c].std(ddof=0)
            X.loc[:, c] = (X[c] - mu) / s
    return X
