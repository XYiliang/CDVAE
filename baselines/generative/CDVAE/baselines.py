# coding = utf-8
import os
from omegaconf import OmegaConf

import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from metrics import cfd_bin, cfd_reg, total_effect, clp_bin, clp_reg
from CDVAE_model.components.predictors import NNClassifier, NNRegressor, NNRegressorWithFairReg, NNClassifierWithFairReg
from CDVAE_model.single_domain.train_test import extract_data_single

import torch
from datetime import datetime

from utils import seed_everything, get_logger, summary
from datasets.get_loaders import get_loaders

seed = 42

TOTAL_REPEAT = 5


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_baselines(vae_list, loaders, logger, cfg, total_repeat_time, df_baselines,
                   df_savepath):

    train_loader, test_loader, dim_dict = loaders['train_loader'], loaders['test_loader'], loaders['dim_dict']

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
            trained_vae = torch.load(f'result/{cfg.dataset_name}/vae_single/{vae_list[i]}')
            trained_vae.eval()
            data_train = extract_data_single(train_loader, trained_vae, dim_dict)
            data_test = extract_data_single(test_loader, trained_vae, dim_dict)

            una_tr, cf_una_tr, cf_una_dis_tr, x_tr, _, cf_r_x_tr, a_tr, y_tr, xn_tr, cf_r_xn_tr = data_train
            una_te, cf_una_te, cf_una_dis_te, x_te, _, cf_r_x_te, a_te, y_te, xn_te, cf_r_xn_te = data_test

            if cfg.dataset_name == 'synthetic':
                real_cf_test_loader = loaders['real_cf_test_loader']
                real_cf_test = extract_data_single(real_cf_test_loader, trained_vae, dim_dict)
                real_cf_una_te, _, _, real_x_te, _, _, real_cf_a_te, _, real_cf_xn_te, _ = real_cf_test
                real_cf_test_input = {
                    'full_NoReg': np.concatenate([real_x_te, real_cf_a_te], axis=1),
                    'xnOnly': real_cf_xn_te,
                    'full_clpReg':  np.concatenate([real_x_te, real_cf_a_te], axis=1),
                    'full_cfclpReg': np.concatenate([real_x_te, real_cf_a_te], axis=1),
                    'repr_NoReg': real_cf_una_te,
                    'repr_clpReg': real_cf_una_te,
                    'repr_cfclpReg': real_cf_una_te
                }

            train_input = {
                'full_NoReg': [np.concatenate([x_tr, a_tr], axis=1), y_tr],
                'xnOnly': [xn_tr, y_tr],
                'full_clpReg': [np.concatenate([x_tr, a_tr], axis=1), np.concatenate([cf_r_x_tr, 1 - a_tr], axis=1),
                                y_tr],
                'full_cfclpReg': [np.concatenate([x_tr, a_tr], axis=1), np.concatenate([cf_r_x_tr, 1 - a_tr], axis=1),
                                  y_tr],
                'repr_NoReg': [una_tr, y_tr],
                'repr_clpReg': [una_tr, cf_una_dis_tr, y_tr],
                'repr_cfclpReg': [una_tr, cf_una_dis_tr, y_tr],
            }

            test_input = {
                'full_NoReg': [np.concatenate([x_te, a_te], axis=1), np.concatenate([cf_r_x_te, 1 - a_te], axis=1),
                               y_te],
                'xnOnly': [xn_te, cf_r_xn_te, y_te],
                'full_clpReg': [np.concatenate([x_te, a_te], axis=1), np.concatenate([cf_r_x_te, 1 - a_te], axis=1),
                                y_te],
                'full_cfclpReg': [np.concatenate([x_te, a_te], axis=1), np.concatenate([cf_r_x_te, 1 - a_te], axis=1),
                                  y_te],
                'repr_NoReg': [una_te, cf_una_dis_te, y_te],
                'repr_clpReg': [una_te, cf_una_dis_te, y_te],
                'repr_cfclpReg': [una_te, cf_una_dis_te, y_te]
            }

            logger.info(f'Current baseline setting is {setting_name}, progress:{i + 1}/{total_repeat_time}')
            res = []

            for name, model in zip(['LR', 'SVM', 'MLP'], setting):
                logger.info(f'\t{setting_name} {name} train and test')

                if model is None:
                    res += [np.nan] * 5
                    continue

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

                if cfg.dataset_name == 'law_school':
                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(test_data_counter)

                    acc = mean_squared_error(test_target, pred_factual) ** 0.5
                    auc = np.nan
                    te = total_effect(pred_factual, pred_counter)
                    cfd = cfd_reg(pred_factual, pred_counter)
                    clp = clp_reg(pred_factual, pred_counter)

                else:
                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(test_data_counter)
                    pred_factual_prob = model.predict_proba(test_data_factual)
                    pred_counter_prob = model.predict_proba(test_data_counter)

                    acc = accuracy_score(test_target, pred_factual)
                    auc = roc_auc_score(test_target, pred_factual)
                    te = total_effect(pred_factual_prob, pred_counter_prob)
                    cfd = cfd_bin(pred_factual, pred_counter)
                    clp = clp_bin(pred_factual_prob, pred_counter_prob)

                res += [acc, auc, te, cfd, clp]

                if cfg.dataset_name == 'synthetic':
                    real_test_data_counter = real_cf_test_input[setting_name]

                    pred_factual = model.predict(test_data_factual)
                    pred_counter = model.predict(real_test_data_counter)
                    pred_factual_prob = model.predict_proba(test_data_factual)
                    pred_counter_prob = model.predict_proba(real_test_data_counter)

                    te = total_effect(pred_factual_prob, pred_counter_prob)
                    cfd = cfd_bin(pred_factual, pred_counter)
                    clp = clp_bin(pred_factual_prob, pred_counter_prob)
                    res += [te, cfd, clp]

            df_baselines[setting_name + f'_{i}'] = res
            df_baselines.to_csv(df_savepath)

        repeat_part = df_baselines.loc[:, df_baselines.columns.str.contains(setting_name + r'_\d+')]
        df_baselines[setting_name + '_summary'] = summary(repeat_part)
        df_baselines.to_csv(df_savepath)

    logger.info(f'{cfg.dataset_name} baselines comparision exp finished.')


def test_compas():
    cfg = OmegaConf.load('config/cfg_compas.yaml')
    df_baselines = pd.read_csv('result/compas/compas_baseline_compare.csv', index_col=[0, 1])
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
                   df_baselines, 'result/compas/compas_baseline_compare.csv')


def test_law():
    cfg = OmegaConf.load('config/cfg_law_school.yaml')
    df_baselines = pd.read_csv('result/law_school/law_baseline_compare.csv', index_col=[0, 1])
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
                   df_baselines, 'result/law_school/law_baseline_compare.csv')


def test_adult():
    cfg = OmegaConf.load('config/cfg_adult.yaml')
    df_baselines = pd.read_csv('result/adult/adult_baseline_compare.csv', index_col=[0, 1])
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    # '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css7.5_uxn5_uxa10_seed29).pth'
    vae_models = ['0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed5).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed35).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed42).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed71).pth',
                  '0_vae_single(xn1.0_xa1.0_y1.0_kl0.01_tc0.5_css1.0_uxn5_uxa10_seed80).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg, TOTAL_REPEAT,
                   df_baselines, 'result/adult/adult_baseline_compare.csv')


def test_synthetic():
    cfg = OmegaConf.load('config/cfg_synthetic_single.yaml')
    df_baselines = pd.read_csv('result/synthetic/synthetic_baseline_compare_single.csv', index_col=[0, 1])
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
                   df_baselines, 'result/synthetic/synthetic_baseline_compare_single.csv')

