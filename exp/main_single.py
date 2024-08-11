# coding = utf-8
import torch

from datetime import datetime
from CDVAE_model.single_domain.VAE import CDVAE
from CDVAE_model.single_domain.train_test import train, test
from utils import seed_everything, get_writer, get_logger, get_result_dataframe, save_result_dataframe, save_vae, save_predictors
from datasets.get_loaders import get_loaders


def main(cfg):
    res_df = get_result_dataframe(cfg)
    seed_everything(cfg)
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
    vae_savename = f'vae_single(xn{cfg.parm_r_xn}_xa{cfg.parm_r_xa}_y{cfg.parm_r_y}_kl{cfg.parm_kl}_' \
                   f'tc{cfg.parm_tc}_css{cfg.parm_css}_uxn{cfg.un_dim}_uxa{cfg.ua_dim}_seed{cfg.seed}).pth'
    if cfg.dataset_name == 'ACSIncome':
        vae_savename = cfg.state + '_' + vae_savename

    loaders = get_loaders(cfg)
    writer = get_writer(cfg.tensorboard_log_dir, vae_savename, cfg.dataset_name)
    logger, console_handler, file_handler = get_logger(cfg.log_dir)
    logger.info(f'Experiment Start Time:{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info(f'Main EXP Config: Dataset:{cfg.dataset_name}|Random Seed:{cfg.seed}|VAE Epochs:{cfg.vae_epochs}|'
                f'VAE lr:{cfg.vae_lr}| Discriminator lr:{cfg.disc_lr}|Loss Weights: Xn:{cfg.parm_r_xn} '
                f'Xa:{cfg.parm_r_xa} Y:{cfg.parm_r_y} KL:{cfg.parm_kl} TC:{cfg.parm_tc} CSS:{cfg.parm_css} '
                f'MLP Hidden Layers:{cfg.mlp_hidden_layers} MLP Epochs: {cfg.clf_mlp_epochs}')

    # Train
    xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim = loaders['dim_dict'].values()
    model = CDVAE(xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim, cfg)

    model = train(model, cfg.vae_epochs, writer, loaders['train_loader'], logger, device, loaders['dim_dict'], cfg)
    
    save_vae(model, vae_savename, cfg)
    logger.info('--------------VAE Model saved--------------')

    # Test
    metrics_dict, res_df, predictors = test(model, loaders, vae_savename, res_df, logger, cfg)

    save_predictors(predictors, vae_savename, cfg)
    logger.info('--------------Predictor saved--------------')
    save_result_dataframe(res_df, cfg)
    logger.info('--------------Result Dataframe saved--------------')

    for k, v in metrics_dict.items():
        logger.info(f'{k}:{v}\n')
    logger.info(f'{vae_savename} Experiment end\n\n')
    logger.removeHandler(file_handler)
    logger.removeHandler(console_handler)
