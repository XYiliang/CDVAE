# coding = utf-8
import os
import random
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from datetime import datetime
from tqdm import tqdm
from transfer_model.VAE import CDVAE
from utils import seed_everything, get_writer, get_logger, get_result_dataframe, save_result_dataframe, save_vae, \
    save_clf
from datasets.get_loaders import get_loaders
from transfer_model.train_test import train, test
import itertools

CONFIG_PATH = 'config/cfg_new_adult_transfer.yaml'


def get_cfg(cfg_path):
    cfg_dict = OmegaConf.load(cfg_path)
    return OmegaConf.create(cfg_dict)


def main(cfg, i):
    # Initial
    # res_df = get_result_dataframe(cfg)
    res_df = pd.read_csv('result/ACSIncome/ablation_new/ablation_CA_TX(uxn10uxa50).csv', index_col=[0, 1])
    seed_everything(cfg)
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")

    if cfg.dataset_name == 'ACSIncome':
        vae_savename = f'vae_{cfg.source_state[0]}_{cfg.target_state[0]}(xn{cfg.parm_r_xn}_xa{cfg.parm_r_xa}_y{cfg.parm_r_y}' \
                       f'_kl{cfg.parm_kl}_tc{cfg.parm_tc}_css{cfg.parm_css}_tf{cfg.parm_tf}_uxn{cfg.uxn_dim}' \
                       f'_uxa{cfg.uxa_dim}_seed{cfg.seed}).pth'
    else:
        vae_savename = f'vae(xn{cfg.parm_r_xn}_xa{cfg.parm_r_xa}_y{cfg.parm_r_y}_kl{cfg.parm_kl}_tc{cfg.parm_tc}' \
                       f'_css{cfg.parm_css}_tf{cfg.parm_tf}_uxn{cfg.uxn_dim}_uxa{cfg.uxa_dim}).pth'

    writer = get_writer(cfg, vae_savename)
    logger, console_handler, file_handler = get_logger(cfg.log_dir)

    logger.info(f'EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if cfg.dataset_name == 'ACSIncome':
        logger.info(f'----------------Current Source State:{cfg.source_state}-------------')
        logger.info(f'----------------Current Target State:{cfg.target_state}-------------')

    logger.info(f'Main EXP Config: Random Seed:{cfg.seed}|VAE lr:{cfg.lr}'
                f'|Discriminator lr:{cfg.disc_lr}|Epochs:{cfg.n_epochs}|Loss Weights: Xn:{cfg.parm_r_xn}'
                f' Xa:{cfg.parm_r_xa} Y:{cfg.parm_r_y} KL:{cfg.parm_kl} TC:{cfg.parm_tc} CSS:{cfg.parm_css}'
                f' TF:{cfg.parm_tf} ')
    src_train_loader, trg_train_loader, src_test_loader, trg_test_loader, dim_dict = get_loaders(cfg)

    # Train
    # css0.0 tf0.0
    model = CDVAE(dim_dict, cfg)
    trained_vae = train(model, src_train_loader, trg_train_loader, vae_savename, cfg.n_epochs, writer,
                        logger, device, dim_dict, cfg)
    # save_vae(trained_vae, vae_savename, cfg)
    torch.save(trained_vae, f'result/ACSIncome/ablation_new/vae/{i}_{vae_savename}')
    # Test
    metrics_dict, result_log_df, trained_clf = test(trained_vae, src_train_loader, src_test_loader, trg_test_loader,
                                                    vae_savename, res_df, cfg, dim_dict)
    result_log_df.to_csv('result/ACSIncome/ablation_new/ablation_CA_TX(uxn10uxa50).csv')
    #
    # save_result_dataframe(res_df, cfg)
    # save_clf(trained_clf, vae_savename, cfg)
    # result_log_df.to_csv('result/ACSIncome/ablation_CA_TX(uxn10uxa50).csv')

    # logger.info(f'Result: {metrics_dict}')
    # logger.info(f'{vae_savename} EXP end\n\n')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)


if __name__ == '__main__':
    cfg = get_cfg('config/cfg_new_adult_transfer.yaml')
    for css, tf in [(5.0, 0.1), (0.0, 0.0), (0.0, 0.1), (5.0, 0.0)]:
        for i in range(5):
            if i == 0:
                if css == 5.0 and tf == 0.1:
                    continue
                cfg.draw_tSNE = True
            else:
                cfg.draw_tSNE = True
            cfg.parm_css = css
            cfg.parm_tf = tf
            cfg.seed = None
            main(cfg, i)
    # for i in tqdm(range(1, 5)):
    #     if i == 1:
    #         cfg.draw_tSNE = False
    #     else:
    #         cfg.draw_tSNE = False
    #     cfg.seed = None
    #     main(cfg, i)
    # cfg = get_cfg('config/cfg_new_adult_transfer.yaml')
