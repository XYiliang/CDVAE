# coding = utf-8
import sys
import torch
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from timeit import default_timer
from utils import filter_dummy_x, get_new_colname
from CDVAE_model.transfer.loss import CDVAELoss
from CDVAE_model.components.predictors import NNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from metrics import cfd_bin, compute_mmd, absolute_total_effect
from utils import draw_tSNE

import warnings
warnings.filterwarnings("ignore")


def train(model, loaders, writer, logger, device, cfg):
    model.to(device)
    loss_func = CDVAELoss(model, device, loaders['dim_dict'], cfg)
    # Train
    logger.info('--------------Start VAE Training--------------')
    start = default_timer()

    model.train()
    with tqdm(range(cfg.clf_mlp_epochs), position=0, unit='epoch') as t:
        for epoch in t:
            torch.cuda.empty_cache()
            len_loader = min(len(loaders['src_train_loader']), len(loaders['trg_train_loader']))
            src_loader_iter = iter(loaders['src_train_loader'])
            trg_loader_iter = iter(loaders['trg_train_loader'])
            recon_loss, y_r_loss, KLD, tc_loss, css_loss, domain_loss, VAE_loss, disc_tc_loss = 0, 0, 0, 0, 0, 0, 0, 0
            i = 0
            while i < len_loader:
                p = float(i + epoch * len_loader) / cfg.clf_mlp_epochs / len_loader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1

                model.zero_grad()
                # Get Source Data
                src_data = next(src_loader_iter)
                xnc_s, xnb_s, xac_s, xab_s, a_s, y_s = src_data
                domain_tag_s = torch.full((y_s.shape[0],), 0).type(torch.LongTensor)
                src_data = (xnc_s, xnb_s, xac_s, xab_s, a_s, y_s, domain_tag_s)

                # Get Target Data
                trg_data = next(trg_loader_iter)
                xnc_t, xnb_t, xac_t, xab_t, a_t, y_t = trg_data
                domain_tag_t = torch.full((y_t.shape[0],), 1).type(torch.LongTensor)
                trg_data = (xnc_t, xnb_t, xac_t, xab_t, a_t, y_t, domain_tag_t)

                losses = loss_func.optimize(src_data, trg_data, alpha, epoch)
                recon_loss_bat, y_r_loss_bat, KLD_bat, tc_loss_bat, css_loss_bat, domain_loss_bat, VAE_loss_bat, \
                    disc_tc_loss_bat = losses

                recon_loss += recon_loss_bat.detach().item()
                y_r_loss += y_r_loss_bat.detach().item()
                KLD += KLD_bat.detach().item()
                tc_loss += tc_loss_bat.detach().item()
                css_loss += css_loss_bat.detach().item()
                domain_loss += domain_loss_bat.detach().item()
                VAE_loss += VAE_loss_bat.detach().item()
                disc_tc_loss += disc_tc_loss_bat.detach().item()

                i += 1

            recon_loss /= len_loader
            y_r_loss /= len_loader
            KLD /= len_loader
            tc_loss /= len_loader
            css_loss /= len_loader
            domain_loss /= len_loader
            VAE_loss /= len_loader
            disc_tc_loss /= len_loader

            t.set_postfix({'VAE Loss': round(float(VAE_loss), 4)})
            losses = {'Reconstruction Loss': recon_loss, 'KLD': KLD, 'Total Correlation Loss': tc_loss,
                    'Consistency': css_loss, 'Transfer Loss': domain_loss, 'VAE Loss': VAE_loss,
                    'Discriminator TC Loss': disc_tc_loss, 'y_r_loss': y_r_loss}

            for k, v in losses.items():
                writer.add_scalar(k, v, epoch)

    writer.close()
    model.eval()

    delta_time = (default_timer() - start) / 60
    logger.info('--------------Model saved--------------')
    logger.info(f'Finished VAE training after {delta_time:.1f} min.')
    logger.info(f'--------------Start CLF training--------------')
    torch.cuda.empty_cache()

    return model


def test(trained_vae, vae_savename, loaders, result_log_df, cfg):
    trained_vae.eval()

    src_train_real, src_train_gen_factual, src_train_gen_counter, src_train_distilled = extract_data_transfer(
        loaders['src_train_loader'], 
        trained_vae, 
        loaders['dim_dict']
    )
    src_test_real, src_test_gen_factual, src_test_gen_counter, src_test_distilled = extract_data_transfer(
        loaders['src_test_loader'],
        trained_vae, 
        loaders['dim_dict']
    )
    trg_train_real, trg_train_gen_factual, trg_train_gen_counter, trg_train_distilled = extract_data_transfer(
        loaders['trg_train_loader'], 
        trained_vae, loaders['dim_dict']
    )
    trg_test_real, trg_test_gen_factual, trg_test_gen_counter, trg_test_distilled = extract_data_transfer(
        loaders['trg_test_loader'], 
        trained_vae, 
        loaders['dim_dict']
    )

    # MMD between latent code of source domain and of target domain
    mmd_src_trg = float(compute_mmd(torch.from_numpy(src_test_gen_factual['U']), torch.from_numpy(trg_test_gen_factual['U'])))

    if cfg.draw_tSNE:
        draw_tSNE(trg_test_gen_factual['U'], trg_test_distilled['U'], cfg,
                f'Distilled_{vae_savename[:-4]}.tif', comparison='distill', ylim=None)
        draw_tSNE(src_test_gen_factual['U'], trg_test_gen_factual['U'], cfg, 
                f'Src-Trg_{vae_savename[:-4]}.tif', comparison='src-trg', mmd_val=mmd_src_trg, ylim=None)

    U_src_tr, U_src_val, _, U_trg_val, y_src_tr, y_src_val, _, y_trg_val = train_test_split(
        src_train_distilled['U'],
        trg_train_distilled['U'],
        src_train_real['y'],
        trg_train_real['y']
    )

    if cfg.use_torch_mlp:
        mlp_model = NNClassifier(
            lr=cfg.mlp_lr, 
            n_epochs=cfg.clf_mlp_epochs, 
            hidden_layer_sizes=cfg.mlp_hidden_layers
            )

        mlp_model.fit_transfer(U_src_tr, y_src_tr, U_src_val, y_src_val, U_trg_val, y_trg_val)
    else:
        mlp_model = MLPClassifier(
            hidden_layer_sizes=cfg.mlp_hidden_layers, 
            learning_rate_init=cfg.mlp_lr,
            max_iter=cfg.clf_mlp_epochs, 
            random_state=cfg.seed, activation='relu', verbose=True
        )
        mlp_model.fit(U_src_tr, y_src_tr)

    # Test on source data
    src_metrics_dict, src_metrics_list = calc_metrics_transfer(
        predictor=mlp_model,
        U=src_test_gen_factual['U'],
        cf_U_dis=src_test_distilled['U'],
        x=src_test_real['x'],
        r_x=src_test_gen_factual['x'],
        cf_r_x=src_test_gen_counter['x'],
        y=src_test_real['y']
    )
    # Test on target data
    trg_metrics_dict, trg_metrics_list = calc_metrics_transfer(
        predictor=mlp_model,
        U=trg_test_gen_factual['U'],
        cf_U_dis=trg_test_distilled['U'],
        x=trg_test_real['x'],
        r_x=trg_test_gen_factual['x'],
        cf_r_x=trg_test_gen_counter['x'],
        y=trg_test_real['y']
    )

    metrics_dict = {'MMD_latent': mmd_src_trg, 'Source': src_metrics_dict, 'Target': trg_metrics_dict}

    metric_list = [mmd_src_trg] + src_metrics_list + trg_metrics_list
    col_name = vae_savename[:-5] + f'_hlayers{cfg.mlp_hidden_layers}).pth'
    col_name = get_new_colname(col_name, result_log_df)

    result_log_df[col_name] = metric_list

    return metrics_dict, result_log_df, {'MLP': mlp_model}

def calc_metrics_transfer(predictor, U, cf_U_dis, x, r_x, cf_r_x, y):
    predictor = predictor.cpu()
    pred_factual = predictor.predict(U)
    pred_counter_dis = predictor.predict(cf_U_dis)
    
    mmd_real_factual = compute_mmd(x, r_x)
    mmd_real_counter = compute_mmd(x, cf_r_x)
    mmd_counter_factual = compute_mmd(cf_r_x, r_x)

    pred_factual_prob = predictor.predict_proba(U)
    pred_counter_dis_prob = predictor.predict_proba(cf_U_dis)

    acc = accuracy_score(y, pred_factual)
    auc = roc_auc_score(y, pred_factual)
    te = absolute_total_effect(pred_factual_prob, pred_counter_dis_prob)
    cfd = cfd_bin(pred_factual, pred_counter_dis)
    
    metrics_dict = {'Acc': acc, 'AUC': auc, '|TE|': te, 'CFD': cfd,
                    'mmd_rf': mmd_real_factual,'mmd_rc': mmd_real_counter,
                    'mmd_cf': mmd_counter_factual}
    metrics_list = [acc, auc, te, cfd, mmd_real_factual, mmd_real_counter,
                    mmd_counter_factual]
    
    return metrics_dict, metrics_list


def extract_data_transfer(loader, trained_vae, dim_dict):
    with torch.no_grad():
        xn_list, xa_list, a_list, y_list = [], [], [], []
        for data in loader:
            xnc, xnb, xac, xab, a, y = data
            xn, xa = filter_dummy_x(xnc, xnb, xac, xab, dim_dict)
            xn_list.append(xn)
            xa_list.append(xa)
            a_list.append(a)
            y_list.append(y)
        xn, xa, a, y = torch.cat(xn_list, 0), torch.cat(xa_list, 0), torch.cat(a_list, 0), torch.cat(y_list, 0)

        trained_vae = trained_vae.cpu()
        factual, counter, cf_dis, _ = trained_vae(xn, xa, a)
        un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_un, cf_ua, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab = counter
        cf_un_dis, cf_ua_dis = cf_dis

        r_xnb = torch.sigmoid(r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        r_xab = torch.sigmoid(r_xab).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xnb = torch.sigmoid(cf_r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xab = torch.sigmoid(cf_r_xab).gt(0.5).byte().type(torch.FloatTensor)

        r_xn, r_xa = filter_dummy_x(r_xnc, r_xnb, r_xac, r_xab, dim_dict)
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, dim_dict)
        x = torch.cat([xn, xa], 1)
        r_x = torch.cat([r_xn, r_xa], 1)
        cf_r_x = torch.cat([cf_r_xn, cf_r_xa], 1)

        U = torch.cat([un, ua], 1)
        cf_U = torch.cat([cf_un, cf_ua], 1)
        cf_U_dis = torch.cat([cf_un_dis, cf_ua_dis], 1)

        U, cf_U, cf_U_dis = U.cpu().numpy(), cf_U.cpu().numpy(), cf_U_dis.cpu().numpy()
        x, r_x, cf_r_x = x.cpu().numpy(), r_x.cpu().numpy(), cf_r_x.cpu().numpy()
        a, y = a.cpu().numpy(), y.long().squeeze().cpu().numpy()

        real = {'xn': x, 'xa': xa, 'x': x, 'a': a, 'y': y}
        gen_factual = {'xn': r_xn, 'xa': r_xa, 'x': r_x, 'y': r_y, 'un': un, 'ua': ua, 'U': U}
        gen_counter = {'xn': cf_r_xn, 'xa': cf_r_xa, 'x': cf_r_x, 'un': cf_un, 'ua': cf_ua, 'U': cf_U}
        distilled = {'un': cf_un_dis, 'ua': cf_ua_dis, 'U': cf_U_dis}

        for pack in [real, gen_factual, gen_counter, distilled]:
            for k, v in pack.items():
                if isinstance(v, torch.Tensor):
                    pack[k] = v.cpu()

    return real, gen_factual, gen_counter, distilled
