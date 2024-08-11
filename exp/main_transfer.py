# coding = utf-8
import torch

from omegaconf import OmegaConf
from datetime import datetime
from utils import seed_everything, get_writer, get_logger, get_result_dataframe, \
    save_result_dataframe, save_vae, save_predictors
from datasets.get_loaders import get_loaders
from CDVAE_model.transfer.VAE import CDVAE
from CDVAE_model.transfer.train_test import train, test


def get_cfg(cfg_path):
    cfg_dict = OmegaConf.load(cfg_path)
    return OmegaConf.create(cfg_dict)


def main(cfg):
    # Initial
    res_df = get_result_dataframe(cfg)
    seed_everything(cfg)
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")

    if cfg.dataset_name == 'ACSIncome':
        vae_savename = f'vae_{cfg.source_state[0]}_{cfg.target_state[0]}(xn{cfg.parm_r_xn}_xa{cfg.parm_r_xa}_y{cfg.parm_r_y}' \
                       f'_kl{cfg.parm_kl}_tc{cfg.parm_tc}_css{cfg.parm_css}_tf{cfg.parm_tf}_uxn{cfg.un_dim}' \
                       f'_uxa{cfg.ua_dim}_seed{cfg.seed}).pth'
    else:
        vae_savename = f'vae(xn{cfg.parm_r_xn}_xa{cfg.parm_r_xa}_y{cfg.parm_r_y}_kl{cfg.parm_kl}_tc{cfg.parm_tc}' \
                       f'_css{cfg.parm_css}_tf{cfg.parm_tf}_uxn{cfg.un_dim}_uxa{cfg.ua_dim}).pth'

    writer = get_writer(cfg.tensorboard_log_dir, cfg.dataset_name, vae_savename)
    logger, console_handler, file_handler = get_logger(cfg.log_dir)

    logger.info(f'EXP Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    if cfg.dataset_name == 'ACSIncome':
        logger.info(f'----------------Current Source State:{cfg.source_state}-------------')
        logger.info(f'----------------Current Target State:{cfg.target_state}-------------')

    logger.info(f'Main EXP Config: Random Seed:{cfg.seed}|VAE lr:{cfg.vae_lr}'
                f'|Discriminator lr:{cfg.disc_lr}|Epochs:{cfg.vae_epochs}|Loss Weights: Xn:{cfg.parm_r_xn}'
                f' Xa:{cfg.parm_r_xa} Y:{cfg.parm_r_y} KL:{cfg.parm_kl} TC:{cfg.parm_tc} CSS:{cfg.parm_css}'
                f' TF:{cfg.parm_tf} ')
    
    loaders = get_loaders(cfg)

    # Train
    model = CDVAE(loaders['dim_dict'], cfg)
    trained_vae = train(model,
                        loaders,
                        writer,
                        logger, 
                        device,
                        cfg)
    save_vae(trained_vae, vae_savename, cfg)
    # Test
    metrics_dict, result_log_df, trained_clf = test(trained_vae, 
                                                    vae_savename,
                                                    loaders,
                                                    res_df, 
                                                    cfg)
    
    # Save result
    save_predictors(trained_clf, vae_savename, cfg)
    logger.info('--------------Predictor saved--------------')
    save_result_dataframe(result_log_df, cfg)
    logger.info('--------------Result Dataframe saved--------------')

    logger.info(f'Result: {metrics_dict}')
    logger.info(f'{vae_savename} EXP end\n\n')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
