# coding = utf-8
import os
import sys
sys.path.append("./baselines/predictive")
import random
import numpy as np
import pandas as pd
import dill
import torch

from sklearn.metrics import accuracy_score, roc_auc_score
from metrics import cfd_bin, absolute_total_effect
from CDVAE_model.components.predictors import NNClassifier, NNClassifierWithFairReg
from CDVAE_model.transfer.train_test import extract_data_transfer
from CDVAE_model.transfer.VAE import CDVAE
from datetime import datetime
from utils import seed_everything, get_logger, summary, ini_baseline_res_df
from datasets.get_loaders import get_loaders

seed = np.random.randint(0, 100)

TSF_SYNTHETIC = 'baselines/predictive/transfer_fairness/clf/transfer_fairness_clf_synthetic.pth'
TSF_ACSINCOME = 'baselines/predictive/transfer_fairness/clf/transfer_fairness_ACSIncome_adult.pth'
RBSTF_SYNTHETIC = 'baselines/predictive/robust_fair_covariate_shift/trained_models/robust_cov_clf_synthetic.pkl'
RBSTF_ACSINCOME = 'baselines/predictive/robust_fair_covariate_shift/trained_models/robust_cov_clf_ACSIncome.pkl'


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test_baselines(vae_list, loaders, logger, cfg, df_baselines, df_savepath):
    set_seed(seed)

    if cfg.dataset_name == 'synthetic':
        mlp_args_NoReg = {'lr': 1e-4, 'n_epochs': 1, 'hidden_layer_sizes': (128, 128, 128)}
        mlp_args_clpReg = {'lr': 1e-4, 'n_epochs': 1, 'hidden_layer_sizes': (128, 128, 128), 'parm_cf': 0.0,
                           'parm_clp': 1.0}

        full_NoReg = NNClassifier(**mlp_args_NoReg)
        full_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)
        repr_NoReg = NNClassifier(**mlp_args_NoReg)
        repr_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)

        tsf_model = torch.load(TSF_SYNTHETIC).cpu()
        with open(RBSTF_SYNTHETIC, 'rb') as f:
            rbstf_model = dill.load(f)

    elif cfg.dataset_name == 'ACSIncome':
        mlp_args_NoReg = {'lr': 5e-5, 'n_epochs': 150, 'hidden_layer_sizes': (256, 512, 256)}
        mlp_args_clpReg = {'lr': 5e-5, 'n_epochs': 150, 'hidden_layer_sizes': (256, 512, 256), 'parm_cf': 0.0,
                           'parm_clp': 1.0}

        full_NoReg = NNClassifier(**mlp_args_NoReg)
        full_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)
        repr_NoReg = NNClassifier(**mlp_args_NoReg)
        repr_clpReg = NNClassifierWithFairReg(**mlp_args_clpReg)

        tsf_model = torch.load(TSF_ACSINCOME).cpu()
        with open(RBSTF_ACSINCOME, 'rb') as f:
            rbstf_model = dill.load(f)

    else:
        raise ValueError('Wrong dataset name.')

    settings = [full_NoReg, full_clpReg, repr_NoReg, repr_clpReg, tsf_model, rbstf_model]
    settings_names = ['full_NoReg', 'full_clpReg', 'repr_NoReg', 'repr_clpReg', 'tsf', 'rbstf']

    for model, setting_name in zip(settings, settings_names):
        for i in range(len(vae_list)):

            state_dict_path = f'result/transfer_baseline_compare/{cfg.dataset_name}/pretrained/{vae_list[i]}'
            state_dict = torch.load(state_dict_path)
            trained_vae = CDVAE(loaders['dim_dict'], cfg)
            trained_vae.load_state_dict(state_dict)

            real_src_train, gen_factual_src_train, gen_counter_src_train, distilled_src_train = extract_data_transfer(
                loaders['src_train_loader'], trained_vae, loaders['dim_dict'])
            real_trg_test, gen_factual_trg_test, gen_counter_trg_test, distilled_trg_test = extract_data_transfer(
                loaders['trg_test_loader'], trained_vae, loaders['dim_dict'])

            train_input = {
                'full_NoReg': [np.concatenate([real_src_train['x'], real_src_train['a']], axis=1), real_src_train['y']],
                'full_clpReg': [np.concatenate([real_src_train['x'], real_src_train['a']], axis=1),
                                np.concatenate([gen_counter_src_train['x'], 1 - real_src_train['a']], axis=1),
                                real_src_train['y']],
                'repr_NoReg': [gen_factual_src_train['U'], real_src_train['y']],
                'repr_clpReg': [gen_factual_src_train['U'], distilled_src_train['U'], real_src_train['y']],
                'tsf': None,
                'rbstf': None,
            }

            trg_ratio = np.loadtxt(f'baselines/predictive/robust_fair_covariate_shift/{cfg.dataset_name}_trg_test_ratio.csv')
            test_input = {
                'full_NoReg': [np.concatenate([real_trg_test['x'], real_trg_test['a']], axis=1),
                               np.concatenate([gen_counter_trg_test['x'], 1 - real_trg_test['a']], axis=1),
                               real_trg_test['y']],
                'full_clpReg': [np.concatenate([real_trg_test['x'], real_trg_test['a']], axis=1),
                                np.concatenate([gen_counter_trg_test['x'], 1 - real_trg_test['a']], axis=1),
                                real_trg_test['y']],
                'repr_NoReg': [gen_factual_trg_test['U'], distilled_trg_test['U'], real_trg_test['y']],
                'repr_clpReg': [gen_factual_trg_test['U'], distilled_trg_test['U'], real_trg_test['y']],
                'tsf': [real_trg_test['x'], gen_counter_trg_test['x'], real_trg_test['y']],
                'rbstf': [np.hstack((real_trg_test['x'], np.ones((real_trg_test['x'].shape[0], 1)))), 
                          real_trg_test['a'],
                          np.hstack((gen_counter_trg_test['x'], np.ones((gen_counter_trg_test['x'].shape[0], 1)))), 1 - real_trg_test['a'],
                          real_trg_test['y'],
                          trg_ratio],
            }

            logger.info(f'Current baseline setting is {setting_name}, progress:{i + 1}/{len(vae_list)}')

            if setting_name in ['full_NoReg', 'repr_NoReg']:
                model.fit(train_input[setting_name][0], train_input[setting_name][1])
            if setting_name in ['full_clpReg', 'repr_clpReg']:
                model.fit(train_input[setting_name][0], train_input[setting_name][1], train_input[setting_name][2])

            if setting_name in ['full_NoReg', 'full_clpReg', 'repr_NoReg', 'repr_clpReg', 'tsf']:
                test_data_factual = test_input[setting_name][0]
                test_data_counter = test_input[setting_name][1]
                test_target = test_input[setting_name][2]

                if setting_name == 'tsf':
                    test_data_factual = torch.from_numpy(test_data_factual)
                    test_data_counter = torch.from_numpy(test_data_counter)
                    test_target = torch.from_numpy(test_target)
                
                model = model.cpu()
                pred_factual = model.predict(test_data_factual)
                pred_counter = model.predict(test_data_counter)
                pred_factual_prob = model.predict_proba(test_data_factual)
                pred_counter_prob = model.predict_proba(test_data_counter)

                if setting_name == 'tsf':
                    pred_factual = pred_factual.detach().numpy()
                    pred_counter = pred_counter.detach().numpy()
                    pred_factual_prob = pred_factual_prob.detach().numpy()
                    pred_counter_prob = pred_counter_prob.detach().numpy()

                acc = accuracy_score(test_target, pred_factual)
                auc = roc_auc_score(test_target, pred_factual)
                te = absolute_total_effect(pred_factual_prob, pred_counter_prob)
                cfd = cfd_bin(pred_factual, pred_counter)
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
                te = absolute_total_effect(pred_factual_prob, pred_counter_prob)
                cfd = cfd_bin(pred_factual, pred_counter)

            df_baselines[setting_name + f'_{i}'] = [acc, auc, te, cfd]
            df_baselines.to_csv(df_savepath)

        repeat_part = df_baselines.loc[:, df_baselines.columns.str.contains(setting_name + r'_\d+')]
        df_baselines[setting_name + '_summary'] = summary(repeat_part)
        df_baselines.to_csv(df_savepath)

    logger.info(f'{cfg.dataset_name} baselines comparison exp finished.')


def test_synthetic(cfg):
    df_baselines, path = get_df(cfg, 'classification')
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    vae_models = ['0_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '1_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '2_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '3_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth',
                  '4_vae(xn1.0_xa1.0_y1.0_kl0.1_tc1.0_css10.0_tf0.001_uxn3_uxa3).pth']
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg,
                   df_baselines, path)
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


def test_acsincome(cfg):
    df_baselines, path = get_df(cfg, 'classification')
    seed_everything(cfg)
    loaders = get_loaders(cfg)
    vae_models = ['0_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '0_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1_uxn10_uxa50).pth',
                  '2_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '3_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth',
                  '4_vae_CA_TX(xn1.0_xa1.0_y1.0_kl0.01_tc1.0_css1.0_tf0.1).pth', ]
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Baselines test  EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    test_baselines(vae_models, loaders, logger, cfg,
                   df_baselines, path)
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)

def get_df(cfg, task):
    path = f'result/transfer_baseline_compare/{cfg.dataset_name}/compare.csv'
    if not os.path.isfile(path):
        df = ini_baseline_res_df(cfg.dataset_name, task=task, is_transfer=True)
    else:
        df = pd.read_csv(path, index_col=[0, 1])
    return df, path


def main(cfg):
    if cfg.dataset_name == 'synthetic':
        test_synthetic(cfg)
    elif cfg.dataset_name == 'ACSIncome':
        test_acsincome(cfg)
    else:
        raise ValueError

