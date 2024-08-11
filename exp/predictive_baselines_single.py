# coding = utf-8
import torch
import random
import numpy as np
import pandas as pd
import os

from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from metrics import cfd_bin, cfd_reg, absolute_total_effect
from CDVAE_model.single_domain.VAE import CDVAE
from CDVAE_model.components.predictors import *
from CDVAE_model.single_domain.train_test import extract_data, extract_real_cf
from datetime import datetime
from utils import seed_everything, get_logger, summary, ini_baseline_res_df
from datasets.get_loaders import get_loaders

seed = np.random.randint(0, 100)

TOTAL_REPEAT = 5


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_baselines(vae_list, loaders, logger, cfg, total_repeat_time, df_baselines,
                   df_savepath):

    if cfg.dataset_name == 'synthetic':
        mlp_args_NoReg = {'lr': 1e-4, 'n_epochs': 200, 'hidden_layer_sizes': (128, 128)}
        mlp_args_clp = {'lr': 1e-4, 'n_epochs': 200, 'hidden_layer_sizes': (128, 128),
                        'parm_cf': 0.0, 'parm_clp': 1.0}
        mlp_args_cf_clp = {'lr': 1e-4, 'n_epochs': 200, 'hidden_layer_sizes': (128, 128),
                           'parm_cf': 1.0, 'parm_clp': 1.0}

        full_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        xnOnly = [None, None, NNClassifier(**mlp_args_NoReg)]
        full_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        full_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

        repr_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        repr_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        repr_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

    elif cfg.dataset_name == 'compas':
        mlp_args_NoReg = {'lr': 5e-4, 'n_epochs': 500, 'hidden_layer_sizes': (128, 128, 128)}
        mlp_args_clp = {'lr': 5e-4, 'n_epochs': 500, 'hidden_layer_sizes': (128, 128, 128),
                        'parm_cf': 0.0, 'parm_clp': 1.0}
        mlp_args_cf_clp = {'lr': 5e-4, 'n_epochs': 500, 'hidden_layer_sizes': (128, 128, 128),
                           'parm_cf': 1.0, 'parm_clp': 1.0}

        full_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        xnOnly = [None, None, NNClassifier(**mlp_args_NoReg)]
        full_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        full_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

        repr_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        repr_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        repr_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

    elif cfg.dataset_name == 'law_school':
        mlp_args_NoReg = {'lr': 1e-4, 'n_epochs': 1000, 'hidden_layer_sizes': (128, 128)}
        mlp_args_clp = {'lr': 1e-4, 'n_epochs': 1000, 'hidden_layer_sizes': (128, 128),
                        'parm_cf': 0.0, 'parm_clp': 1.0}
        mlp_args_cf_clp = {'lr': 1e-4, 'n_epochs': 1000, 'hidden_layer_sizes': (128, 128),
                           'parm_cf': 1.0, 'parm_clp': 1.0}

        full_NoReg = [None, None, NNRegressor(**mlp_args_NoReg)]
        xnOnly = [None, None, NNRegressor(**mlp_args_NoReg)]
        full_clpReg = [None, None, NNRegressorWithFairReg(**mlp_args_clp)]
        full_cfclpReg = [None, None, NNRegressorWithFairReg(**mlp_args_cf_clp)]
        repr_NoReg = [None, None, NNRegressor(**mlp_args_NoReg)]
        repr_clpReg = [None, None, NNRegressorWithFairReg(**mlp_args_clp)]
        repr_cfclpReg = [None, None, NNRegressorWithFairReg(**mlp_args_cf_clp)]

    elif cfg.dataset_name == 'adult':
        mlp_args_NoReg = {'lr': 1e-3, 'n_epochs': 100, 'hidden_layer_sizes': (512, 512, 512)}
        mlp_args_clp = {'lr': 1e-3, 'n_epochs': 100, 'hidden_layer_sizes': (512, 512, 512),
                        'parm_cf': 0.0, 'parm_clp': 1.0}
        mlp_args_cf_clp = {'lr': 1e-3, 'n_epochs': 100, 'hidden_layer_sizes': (512, 512, 512),
                           'parm_cf': 1.0, 'parm_clp': 1.0}

        full_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        xnOnly = [None, None, NNClassifier(**mlp_args_NoReg)]
        full_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        full_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

        repr_NoReg = [None, None, NNClassifier(**mlp_args_NoReg)]
        repr_clpReg = [None, None, NNClassifierWithFairReg(**mlp_args_clp)]
        repr_cfclpReg = [None, None, NNClassifierWithFairReg(**mlp_args_cf_clp)]

    else:
        raise ValueError('Wrong dataset name.')

    # Full: take (xn, xa, a) as input
    # xnOnly: take xn as input
    # repr: take VAE encoded latent code (uxn, uxa) as input
    # NoReg/clpReg/cfclpReg: whether to add Counterfactual Logistic Pair regular term and Counterfactual term(pfohl)
    # in the loss function
    settings = [full_NoReg, xnOnly, full_clpReg, full_cfclpReg, repr_NoReg, repr_clpReg, repr_cfclpReg]
    settings_names = ['full_NoReg', 'xnOnly', 'full_clpReg', 'full_cfclpReg', 'repr_NoReg', 'repr_clpReg', 'repr_cfclpReg']

    for setting_name, setting in zip(settings_names, settings):
        for i in range(len(vae_list)):
            dims = loaders['dim_dict']
            state_dict_path = f'result/single_baseline_compare/{cfg.dataset_name}/pretrained/{vae_list[i]}'
            state_dict = torch.load(state_dict_path)
            trained_vae = CDVAE(dims['xnc'], dims['xnb'], dims['xac'], dims['xab'], dims['a'], dims['y'], cfg)
            trained_vae.load_state_dict(state_dict)
            
            trained_vae.eval()
            real_train, gen_factual_train, gen_counter_train, distilled_train = extract_data(loaders['train_loader'],
                                                                                            trained_vae,
                                                                                            loaders['dim_dict'])
            real_test, gen_factual_test, gen_counter_test, distilled_test = extract_data(loaders['test_loader'],
                                                                                        trained_vae,
                                                                                        loaders['dim_dict'])

            if cfg.dataset_name == 'synthetic':
                real_cf = extract_real_cf(loaders['real_cf_test_loader'], trained_vae, loaders['dim_dict'])
                real_cf_test_input = {
                    'full_NoReg': np.concatenate([real_cf['x'], real_cf['a']], axis=1),
                    'xnOnly': real_cf['xn'],
                    'full_clpReg':  np.concatenate([real_cf['x'], real_cf['a']], axis=1),
                    'full_cfclpReg': np.concatenate([real_cf['x'], real_cf['a']], axis=1),
                    'repr_NoReg': real_cf['U'],
                    'repr_clpReg': real_cf['U'],
                    'repr_cfclpReg': real_cf['U']
                }

            train_input = {
                'full_NoReg': [np.concatenate([real_train['x'], real_train['a']], axis=1), real_train['y']],
                'xnOnly': [real_train['xn'], real_train['y']],
                'full_clpReg': [np.concatenate([real_train['x'], real_train['a']], axis=1),
                                np.concatenate([gen_counter_train['x'], 1 - real_train['x']], axis=1),
                                real_train['y']],
                'full_cfclpReg': [np.concatenate([real_train['x'], real_train['a']], axis=1), 
                                  np.concatenate([gen_counter_train['x'], 1 - real_train['x']], axis=1),
                                  real_train['y']],
                'repr_NoReg': [gen_factual_train['U'], real_train['y']],
                'repr_clpReg': [gen_factual_train['U'], distilled_train['U'], real_train['y']],
                'repr_cfclpReg': [gen_factual_train['U'], distilled_train['U'], real_train['y']],
            }

            test_input = {
                'full_NoReg': [np.concatenate([gen_factual_test['x'], real_test['a']], axis=1), 
                np.concatenate([gen_counter_test['x'], 1 - real_test['a']], axis=1),
                               real_test['y']],
                'xnOnly': [real_test['xn'], gen_counter_test['x'], real_test['y']],
                'full_clpReg': [np.concatenate([gen_factual_test['x'], real_test['a']], axis=1), 
                np.concatenate([gen_counter_test['x'], 1 - real_test['a']], axis=1),
                                real_test['y']],
                'full_cfclpReg': [np.concatenate([gen_factual_test['x'], real_test['a']], axis=1), 
                np.concatenate([gen_counter_test['x'], 1 - real_test['a']], axis=1),
                                  real_test['y']],
                'repr_NoReg': [gen_factual_test['U'], distilled_test['U'], real_test['y']],
                'repr_clpReg': [gen_factual_test['U'], distilled_test['U'], real_test['y']],
                'repr_cfclpReg': [gen_factual_test['U'], distilled_test['U'], real_test['y']]
            }

            logger.info(f'Current baseline setting is {setting_name}, progress:{i + 1}/{total_repeat_time}')
            res = []

            for name, model in zip(['LR', 'SVM', 'MLP'], setting):
                if model is None:
                    continue
                
                logger.info(f'\t{setting_name} {name} train and test')

                if setting_name in ['full_NoReg', 'xnOnly', 'repr_NoReg']:
                    train_data = train_input[setting_name][0]
                    train_target = train_input[setting_name][1]
                    model.fit(train_data, train_target)

                else:
                    train_data_factual = train_input[setting_name][0]
                    train_data_counter = train_input[setting_name][1]
                    train_target = train_input[setting_name][2]
                    model.fit(train_data_factual, train_data_counter, train_target)

                test_data_factual = test_input[setting_name][0]
                test_data_counter = test_input[setting_name][1]
                test_target = test_input[setting_name][2]

                model = model.cpu()
                if cfg.dataset_name == 'law_school':
                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(test_data_counter)

                    mse = mean_squared_error(test_target, pred_factual) ** 0.5
                    te = absolute_total_effect(pred_factual, pred_counter)
                    cfd = cfd_reg(pred_factual, pred_counter)

                    res += [mse, te, cfd]
                else:
                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(test_data_counter)
                    pred_factual_prob = model.predict_proba(test_data_factual)
                    pred_counter_prob = model.predict_proba(test_data_counter)

                    acc = accuracy_score(test_target, pred_factual)
                    auc = roc_auc_score(test_target, pred_factual)
                    te = absolute_total_effect(pred_factual_prob, pred_counter_prob)
                    cfd = cfd_bin(pred_factual, pred_counter)

                    res += [acc, auc, te, cfd]

                if cfg.dataset_name == 'synthetic':
                    real_test_data_counter = real_cf_test_input[setting_name]

                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(real_test_data_counter)
                    pred_factual_prob = model.predict_proba(test_data_factual)
                    pred_counter_prob = model.predict_proba(real_test_data_counter)

                    te = absolute_total_effect(pred_factual_prob, pred_counter_prob)
                    cfd = cfd_bin(pred_factual, pred_counter)
                    res += [te, cfd]

            df_baselines[setting_name + f'_{i}'] = res
            df_baselines.to_csv(df_savepath)

        repeat_part = df_baselines.loc[:, df_baselines.columns.str.contains(setting_name + r'_\d+')]
        df_baselines[setting_name + '_summary'] = summary(repeat_part)
        df_baselines.to_csv(df_savepath)

    logger.info(f'{cfg.dataset_name} baselines comparision exp finished.')


def test_compas(cfg):
    df_baselines, path = get_df(cfg, 'classification')
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    vae_models = ['0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '1_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '2_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '4_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '5_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg, TOTAL_REPEAT,
                   df_baselines, path)


def test_law(cfg):
    df_baselines, path = get_df(cfg, 'regression')
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    vae_models = ['1_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seed42).pth',
                  '2_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '3_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '4_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth',
                  '5_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css10.0_uxn3_uxa3_seedNone).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg, TOTAL_REPEAT,
                   df_baselines, path)


def test_adult(cfg):
    df_baselines, path = get_df(cfg, 'classification')
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    vae_models = ['0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed5).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed35).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed42).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed71).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed80).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg, TOTAL_REPEAT,
                   df_baselines, path)


def test_synthetic(cfg):
    df_baselines, path = get_df(cfg, 'classification')
    seed_everything(cfg)
    loaders = get_loaders(cfg)

    vae_models = ['0_vae_single(xn1.0_xa1.0_y1.0_kl0.1_tc0.5_css5.0_uxn3_uxa3_seed42).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.1_tc0.5_css5.0_uxn3_uxa3_seed1).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.1_tc0.5_css5.0_uxn3_uxa3_seed2).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.1_tc0.5_css5.0_uxn3_uxa3_seed3).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.1_tc0.5_css5.0_uxn3_uxa3_seed4).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg, TOTAL_REPEAT,
                   df_baselines, path)


def get_df(cfg, task):
    path = f'result/single_baseline_compare/{cfg.dataset_name}/compare.csv'
    if not os.path.isfile(path):
        df = ini_baseline_res_df(cfg.dataset_name, task=task, is_transfer=False)
    else:
        df = pd.read_csv(path, index_col=[0, 1])
    return df, path

def main(cfg):
    if cfg.dataset_name == 'synthetic':
        test_synthetic(cfg)
    elif cfg.dataset_name == 'compas':
        test_compas(cfg)
    elif cfg.dataset_name == 'law_school':
        test_law(cfg)
    elif cfg.dataset_name == 'adult':
        test_adult(cfg)
    else:
        raise ValueError
