# coding = utf-8
import os

import dill
from omegaconf import OmegaConf
# os.chdir('../')
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from metrics import cfd_bin, cfd_reg, total_effect, clp_bin, clp_reg
from single_domain_model.predictors import NNClassifier, NNRegressor, NNRegressorWithFairReg, NNClassifierWithFairReg
from transfer_model.train_test import extract_data_transfer

import torch
from datetime import datetime

from utils import seed_everything, get_logger, summary
from datasets.get_loaders import get_loaders
from robust_fair_covariate_shift.create_shift import estimate_ratio

seed = 1
TOTAL_REPEAT = 5


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_baselines(vae_list, train_loader_src, test_loader_trg, logger, cfg, dim_dict, df_baselines, df_savepath):
    set_seed(seed)

    if cfg.dataset_name == 'synthetic':
        mlp_args_NoReg = {'lr': 1e-4, 'n_epochs': 500, 'hidden_layer_sizes': (128, 128, 128)}
        mlp_args_clpReg = {'lr': 1e-4, 'n_epochs': 500, 'hidden_layer_sizes': (128, 128, 128), 'parm_cf': 0.0,
                           'parm_clp': 1.0}

        full_NoReg = NNClassifier(**mlp_args_NoReg)
        full_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)
        repr_NoReg = NNClassifier(**mlp_args_NoReg)
        repr_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)

        transfer_fairness = torch.load('transfer_fairness/result/transfer_fairness_clf_synthetic.pth').cpu()
        with open('robust_fair_covariate_shift/trained_models/robust_cov_clf_synthetic.pkl', 'rb') as f:
            robust_fair_covs = dill.load(f)

    elif cfg.dataset_name == 'ACSIncome':
        mlp_args_NoReg = {'lr': 5e-5, 'n_epochs': 150, 'hidden_layer_sizes': (256, 512, 256)}
        mlp_args_clpReg = {'lr': 5e-5, 'n_epochs': 150, 'hidden_layer_sizes': (256, 512, 256), 'parm_cf': 0.0,
                           'parm_clp': 1.0}

        full_NoReg = NNClassifier(**mlp_args_NoReg)
        full_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)
        repr_NoReg = NNClassifier(**mlp_args_NoReg)
        repr_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)

        transfer_fairness = torch.load('transfer_fairness/result/transfer_fairness_clf_new_adult.pth').cpu()
        with open('robust_fair_covariate_shift/trained_models/robust_cov_clf_new_adult.pkl', 'rb') as f:
            robust_fair_covs = dill.load(f)

    else:
        raise ValueError('Wrong dataset name.')

    settings = [full_NoReg, full_clpReg, repr_NoReg, repr_clpReg, transfer_fairness, robust_fair_covs]
    settings_names = ['full_NoReg', 'full_clpReg', 'repr_NoReg', 'repr_clpReg', 'transfer_fairness', 'robust_fair_covs']

    for model, setting_name in zip(settings, settings_names):
        if setting_name != 'robust_fair_covs': continue
        for i in range(len(vae_list)):
            # if i != 0: continue
            trained_vae = torch.load(f'result/{cfg.dataset_name}/vae_transfer/{vae_list[i]}')
            trained_vae.eval()
            una_tr_src, _, cf_una_dis_tr_src, x_tr_src, _, cf_r_x_tr_src, a_tr_src, y_tr_src = extract_data_transfer(
                train_loader_src, trained_vae, dim_dict)
            una_te_trg, _, cf_una_dis_te_trg, x_te_trg, _, cf_r_x_te_trg, a_te_trg, y_te_trg = extract_data_transfer(
                test_loader_trg, trained_vae, dim_dict)

            train_input = {
                'full_NoReg': [np.concatenate([x_tr_src, a_tr_src], axis=1), y_tr_src],
                'full_clpReg': [np.concatenate([x_tr_src, a_tr_src], axis=1),
                                np.concatenate([cf_r_x_tr_src, 1 - a_tr_src], axis=1),
                                y_tr_src],
                'repr_NoReg': [una_tr_src, y_tr_src],
                'repr_clpReg': [una_tr_src, cf_una_dis_tr_src, y_tr_src],
                'transfer_fairness': None,
                'robust_fair_covs': None,
            }

            trg_ratio = np.loadtxt(f'robust_fair_covariate_shift/{cfg.dataset_name}_trg_test_ratio.csv')
            test_input = {
                'full_NoReg': [np.concatenate([x_te_trg, a_te_trg], axis=1),
                               np.concatenate([cf_r_x_te_trg, 1 - a_te_trg], axis=1),
                               y_te_trg],
                'full_clpReg': [np.concatenate([x_te_trg, a_te_trg], axis=1),
                                np.concatenate([cf_r_x_te_trg, 1 - a_te_trg], axis=1),
                                y_te_trg],
                'repr_NoReg': [una_te_trg, cf_una_dis_te_trg, y_te_trg],
                'repr_clpReg': [una_te_trg, cf_una_dis_te_trg, y_te_trg],
                'transfer_fairness': [x_te_trg, cf_r_x_te_trg, y_te_trg],
                'robust_fair_covs': [np.hstack((x_te_trg, np.ones((x_te_trg.shape[0], 1)))), a_te_trg,
                                     np.hstack((cf_r_x_te_trg, np.ones((cf_r_x_te_trg.shape[0], 1)))), 1 - a_te_trg,
                                     y_te_trg,
                                     trg_ratio],
            }

            logger.info(f'Current baseline setting is {setting_name}, progress:{i + 1}/{len(vae_list)}')

            if setting_name in ['full_NoReg', 'repr_NoReg']:
                model.fit(train_input[setting_name][0], train_input[setting_name][1])
            if setting_name in ['full_clpReg', 'repr_clpReg']:
                model.fit(train_input[setting_name][0], train_input[setting_name][1], train_input[setting_name][2])

            if setting_name in ['full_NoReg', 'full_clpReg', 'repr_NoReg', 'repr_clpReg', 'transfer_fairness']:
                test_data_factual = test_input[setting_name][0]
                test_data_counter = test_input[setting_name][1]
                test_target = test_input[setting_name][2]

                if setting_name == 'transfer_fairness':
                    test_data_factual = torch.from_numpy(test_data_factual)
                    test_data_counter = torch.from_numpy(test_data_counter)
                    test_target = torch.from_numpy(test_target)

                # print('model', next(model.parameters()).device)
                # print('data', test_data_factual.device)

                pred_factual = model.predict(test_data_factual)
                pred_counter = model.predict(test_data_counter)
                pred_factual_prob = model.predict_proba(test_data_factual)
                pred_counter_prob = model.predict_proba(test_data_counter)

                if setting_name == 'transfer_fairness':
                    pred_factual = pred_factual.detach().numpy()
                    pred_counter = pred_counter.detach().numpy()
                    pred_factual_prob = pred_factual_prob.detach().numpy()
                    pred_counter_prob = pred_counter_prob.detach().numpy()

                acc = accuracy_score(test_target, pred_factual)
                auc = roc_auc_score(test_target, pred_factual)
                te = total_effect(pred_factual_prob, pred_counter_prob)
                cfd = cfd_bin(pred_factual, pred_counter)
                clp = clp_bin(pred_factual_prob, pred_counter_prob)
            else:
                factual_x, factual_a = test_input[setting_name][0], test_input[setting_name][1].squeeze()
                counter_x, counter_a = test_input[setting_name][2], test_input[setting_name][3].squeeze()
                test_target = test_input[setting_name][4]
                trg_ratio = test_input[setting_name][5]

                pred_factual = model.predict(factual_x, factual_a, trg_ratio)
                pred_counter = model.predict(counter_x, counter_a, trg_ratio)
                pred_factual_prob = model.predict_proba(factual_x, factual_a, trg_ratio)
                pred_counter_prob = model.predict_proba(counter_x, counter_a, trg_ratio)

                acc = accuracy_score(test_target, pred_factual)
                auc = roc_auc_score(test_target, pred_factual)
                te = total_effect(pred_factual_prob, pred_counter_prob)
                cfd = cfd_bin(pred_factual, pred_counter)
                clp = clp_bin(pred_factual_prob, pred_counter_prob)

            df_baselines[setting_name + f'_{i}'] = [acc, auc, te, cfd, clp]
            df_baselines.to_csv(df_savepath)

        repeat_part = df_baselines.loc[:, df_baselines.columns.str.contains(setting_name + r'_\d+')]
        df_baselines[setting_name + '_summary'] = summary(repeat_part)
        df_baselines.to_csv(df_savepath)

    logger.info(f'{cfg.dataset_name} baselines comparison exp finished.')


def test_synthetic():
    cfg = OmegaConf.load('config/cfg_synthetic_transfer.yaml')
    df_baselines = pd.read_csv('result/synthetic/synthetic_baseline_compare_transfer.csv', index_col=[0, 1])
    seed_everything(cfg)
    src_train_loader, trg_train_loader, src_test_loader, trg_test_loader, dim_dict = get_loaders(cfg)
    vae_models = ['0_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '1_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '2_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '3_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '4_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, src_train_loader, trg_test_loader, logger, cfg, dim_dict,
                   df_baselines, 'result/synthetic/synthetic_baseline_compare_transfer.csv')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


def test_newadult():
    cfg = OmegaConf.load('config/cfg_new_adult_transfer.yaml')
    df_baselines = pd.read_csv('result/ACSIncome/newadult_baseline_compare.csv', index_col=[0, 1])
    seed_everything(cfg)
    src_train_loader, trg_train_loader, src_test_loader, trg_test_loader, dim_dict = get_loaders(cfg)
    vae_models = ['0_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '0_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1_uxn10_uxa50).pth',
                  '2_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '3_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '4_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth', ]
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, src_train_loader, trg_test_loader, logger, cfg, dim_dict,
                   df_baselines, 'result/ACSIncome/newadult_baseline_compare.csv')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


if __name__ == '__main__':
    test_synthetic()
    # test_newadult()

