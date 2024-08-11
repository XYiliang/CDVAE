# coding = utf-8
import os
import random
import sys
import joblib
import pandas as pd
import torch
import itertools

from datetime import datetime
from torch import nn
from tqdm import tqdm
from omegaconf import OmegaConf
from CDVAE_model.single_domain.VAE import CDVAE
from CDVAE_model.single_domain.train_test import train, test
from utils import seed_everything, get_writer, get_logger, get_result_dataframe, save_result_dataframe, save_vae, save_clf
from datasets.get_loaders import get_loaders


def get_cfg(cfg_path):
    return OmegaConf.load(cfg_path)

def main(cfg):
    res_df = pd.DataFrame()
    seed_everything(cfg)
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
    vae_savename = f'CDVAE_D.pth'
    if cfg.dataset_name == 'new_adult':
        vae_savename = cfg.state + '_' + vae_savename

    loaders = get_loaders(cfg)
    dim_dict = loaders['dim_dict']

    writer = get_writer(cfg, vae_savename)
    logger, console_handler, file_handler = get_logger('generative_models/')
    logger.info(f'EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Main EXP Config: Dataset:{cfg.dataset_name}|Random Seed:{cfg.seed}|VAE lr:{cfg.vae_lr}'
                f'|Discriminator lr:{cfg.disc_lr}|Epochs:{cfg.vae_epochs}|Loss Weights: Xn:{cfg.parm_r_xn}'
                f' Xa:{cfg.parm_r_xa} Y:{cfg.parm_r_y} KL:{cfg.parm_kl} TC:{cfg.parm_tc} CSS:{cfg.parm_css} '
                f'MLP Hidden Layers:{cfg.mlp_hidden_layers} MLP epochs: {cfg.clf_mlp_epochs}')

    # 训练VAE
    xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim = loaders['dim_dict'].values()
    model = CDVAE(xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim, cfg)
    trained_vae = train(model, vae_savename, cfg.vae_epochs, writer, loaders['train_loader'], logger, device, dim_dict, cfg)
    torch.save(trained_vae, f'generative_models/{vae_savename}')
    # trained_vae = torch.load('/mnt/cfs/SPEECH/xiayiliang/project/VAE/generative_models/CDVAE_D.pth')
    logger.info('--------------VAE Model saved--------------')

    # 测试clf
    test(trained_vae, loaders, vae_savename, res_df, logger, cfg)
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


