# coding = utf-8
import sys
from timeit import default_timer
import torch
import joblib
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from utils import filter_dummy_x, get_new_colname
from CDVAE_model.transfer.loss import TCFVAELoss
from CDVAE_model.components.predictors import NNClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from metrics import cfd_bin, compute_mmd, total_effect, clp_bin
from utils import draw_tSNE

import warnings
warnings.filterwarnings("ignore")



def train(model, src_loader, trg_loader, vae_savename, n_epochs, writer, logger, device, dim_dict, cfg):
    model.to(device)
    loss_func = TCFVAELoss(model, device, dim_dict, cfg)
    # Train
    logger.info('--------------Start VAE Training--------------')
    start = default_timer()
    model.train()
    t = tqdm(range(n_epochs), leave=True, position=0, unit='epoch')

    for epoch in t:
        torch.cuda.empty_cache()
        len_loader = min(len(src_loader), len(trg_loader))
        src_loader_iter = iter(src_loader)
        trg_loader_iter = iter(trg_loader)
        recon_loss, y_r_loss, KLD, tc_loss, css_loss, domain_loss, VAE_loss, disc_tc_loss = 0, 0, 0, 0, 0, 0, 0, 0
        i = 0
        while i < len_loader:
            p = float(i + epoch * len_loader) / n_epochs / len_loader
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

        if epoch % 25 == 0:
            torch.save(model, f'result/ACSIncome/vae_transfer/Epoch{epoch}_{vae_savename}')

    t.close()
    writer.close()
    model.eval()

    delta_time = (default_timer() - start) / 60
    logger.info('--------------Model saved--------------')
    logger.info(f'Finished VAE training after {delta_time:.1f} min.')
    # logger.info(f'Best VAE Loss:{best_loss}')
    logger.info(f'--------------Start CLF training--------------')
    del src_loader, trg_loader
    torch.cuda.empty_cache()

    return model


def test(trained_vae, src_train_loader, src_test_loader, trg_test_loader, vae_savename, result_log_df,
         cfg, dim_dict):
    trained_vae.eval()

    data_train_src = extract_data_transfer(src_train_loader, trained_vae, dim_dict)
    data_test_src = extract_data_transfer(src_test_loader, trained_vae, dim_dict)
    data_test_trg = extract_data_transfer(trg_test_loader, trained_vae, dim_dict)

    # Fit on source and target data
    una_src_tr, cf_una_src_tr, cf_una_dis_src_tr, x_src_tr, r_x_src_tr, cf_r_x_src_tr, a_src_tr, y_src_tr = data_train_src
    una_src_te, cf_una_src_te, cf_una_dis_src_te, x_src_te, r_x_src_te, cf_r_x_src_te, a_src_te, y_src_te = data_test_src
    una_trg_te, cf_una_trg_te, cf_una_dis_trg_te, x_trg_te, r_x_trg_te, cf_r_x_trg_te, a_trg_te, y_trg_te = data_test_trg
    a_src_tr, a_trg_te = a_src_tr.reshape(-1, 1), a_trg_te.reshape(-1, 1)

    # MMD between latent code of source domain and of target domain
    mmd_src_trg = float(compute_mmd(torch.from_numpy(una_src_te), torch.from_numpy(una_trg_te)))

    if cfg.draw_tSNE:
        # draw_tSNE(una_trg_te, cf_una_trg_te, cfg, f'Do_{vae_savename[:-4]}.tif', comparison='do')
        draw_tSNE(una_trg_te, cf_una_dis_trg_te, cfg, f'Distilled_{vae_savename[:-4]}.tif', comparison='distill', ylim=None)
        draw_tSNE(una_src_te, una_trg_te, cfg, f'Src-Trg_{vae_savename[:-4]}.tif', comparison='src-trg', mmd_val=mmd_src_trg, ylim=None)

    # if cfg.is_fit:
    #     ca_mlp = NNClassifier(lr=cfg.mlp_lr, n_epochs=cfg.clf_mlp_epochs, hidden_layer_sizes=cfg.mlp_hidden_layers)
    #     ca_mlp.fit(np.hstack([x_src_tr, a_src_tr]), y_src_tr)
    #     torch.save(ca_mlp, 'result/ACSIncome/CA_mlp.pth')
    #     return
    # else:
    #     ca_mlp = torch.load('result/ACSIncome/CA_mlp.pth')
    #     df = pd.read_csv('result/ACSIncome/full_50states.csv', index_col=[0, 1])
    #     factual_p = ca_mlp.predict_proba(np.hstack([x_trg_te, a_trg_te]))
    #     counter_p = ca_mlp.predict_proba(np.hstack([cf_r_x_trg_te, 1-a_trg_te]))
    #     factual = ca_mlp.predict(np.hstack([x_trg_te, a_trg_te]))
    #     counter = ca_mlp.predict(np.hstack([cf_r_x_trg_te, 1 - a_trg_te]))
    #     acc = accuracy_score(y_trg_te, factual)
    #     auc = roc_auc_score(y_trg_te, factual)
    #     te = total_effect(factual_p, counter_p)
    #     cfd = cfd_bin(factual, counter)
    #     clp = clp_bin(factual_p, counter_p)
    #     df[cfg.target_state] = [acc, auc, te, cfd, clp]
    #     df.to_csv('result/ACSIncome/full_50states.csv')
    #     return

    # full_ablation(x_src_tr, a_src_tr, y_src_tr, x_src_te, a_src_te, y_src_te, cf_r_x_src_te, x_trg_te,
    #               a_trg_te, cf_r_x_trg_te, y_trg_te, cfg)

    if cfg.use_torch_mlp:
        mlp_model = NNClassifier(lr=cfg.mlp_lr, n_epochs=cfg.clf_mlp_epochs, hidden_layer_sizes=cfg.mlp_hidden_layers)
    else:
        mlp_model = MLPClassifier(hidden_layer_sizes=cfg.mlp_hidden_layers, learning_rate_init=cfg.mlp_lr,
                                  max_iter=cfg.clf_mlp_epochs, random_state=cfg.seed, activation='relu', verbose=True)

    mlp_model.fit(una_src_tr, y_src_tr, una_src_te, y_src_te, una_trg_te, y_trg_te)

    # Test on source data
    src_metrics_dict, src_metrics_list = calc_metrics(data_test_src, mlp_model)
    # Test on target data
    trg_metrics_dict, trg_metrics_list = calc_metrics(data_test_trg, mlp_model)

    metrics_dict = {'MMD_latent': mmd_src_trg, 'Source': src_metrics_dict, 'Target': trg_metrics_dict}

    metric_list = [mmd_src_trg] + src_metrics_list + trg_metrics_list
    col_name = vae_savename[:-5] + f'_hlayers{cfg.mlp_hidden_layers}).pth'
    col_name = get_new_colname(col_name, result_log_df)

    result_log_df[col_name] = metric_list

    return metrics_dict, result_log_df, mlp_model


def no_encode_test(x, cf_x, y, clf):
    # clf_single = torch.load('result/fair_consis/transfer_fairness_clf.pth')
    clf.cuda()
    x = torch.from_numpy(x).cuda()
    cf_x = torch.from_numpy(cf_x).cuda()

    with torch.no_grad():
        pred_factual = clf.predict(x)
        pred_counter = clf.predict(cf_x)
        pred_factual_p = clf.predict_proba(x)
        pred_counter_p = clf.predict_proba(cf_x)

        acc = accuracy_score(pred_factual, y)
        auc = roc_auc_score(pred_factual, y)
        te = total_effect(pred_factual_p, pred_counter_p)
        cfd = cfd_bin(pred_factual, pred_counter)
        clp = clp_bin(pred_factual_p, pred_counter_p)
    return [acc, auc, te, cfd, clp]


def calc_metrics(data_pack, clf):
    with torch.no_grad():
        una, cf_una, cf_una_dis, x, r_x, cf_r_x, a, y = data_pack
        pred_factual = clf.predict(una)
        pred_counter_u = clf.predict(cf_una)
        pred_counter_dis = clf.predict(cf_una_dis)

        # pred_factual_prob = clf_single.predict_proba(una)[:, 1]
        # pred_counter_u_prob = clf_single.predict_proba(cf_una)[:, 1]
        # pred_counter_dis_prob = clf_single.predict_proba(cf_una_dis)[:, 1]
        pred_factual_prob = clf.predict_proba(una)
        pred_counter_u_prob = clf.predict_proba(cf_una)
        pred_counter_dis_prob = clf.predict_proba(cf_una_dis)

        acc = accuracy_score(y, pred_factual)
        auc = roc_auc_score(y, pred_factual)
        te_u = total_effect(pred_factual_prob, pred_counter_u_prob)
        te_dis = total_effect(pred_factual_prob, pred_counter_dis_prob)
        cfd_u = cfd_bin(pred_factual, pred_counter_u)
        cfd_dis = cfd_bin(pred_factual, pred_counter_dis)
        clp_u = clp_bin(pred_factual_prob, pred_counter_u_prob)
        clp_dis = clp_bin(pred_factual_prob, pred_counter_dis_prob)

        mmd_real_factual = compute_mmd(x, r_x)
        mmd_real_counter = compute_mmd(x, cf_r_x)
        mmd_counter_factual = compute_mmd(cf_r_x, r_x)

        metrics_dict = {'Acc': acc, 'AUC': auc, 'TE_u': te_u, 'TE_dis': te_dis, 'CFD_u': cfd_u,
                        'CFD_dis': cfd_dis, 'CLP_u': clp_u, 'CLP_dis': clp_dis, 'mmd_rf': mmd_real_factual,
                        'mmd_rc': mmd_real_counter, 'mmd_cf': mmd_counter_factual}
        metrics_list = [acc, auc, te_u, te_dis, cfd_u, cfd_dis, clp_u, clp_dis, mmd_real_factual, mmd_real_counter,
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
        factual, counter, _, cf_dis = trained_vae(xn, xa, a)
        uxn, uxa, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_uxn, cf_uxa, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab = counter
        cf_uxn_dis, cf_uxa_dis = cf_dis

        r_xnb = torch.sigmoid(r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        r_xab = torch.sigmoid(r_xab).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xnb = torch.sigmoid(cf_r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xab = torch.sigmoid(cf_r_xab).gt(0.5).byte().type(torch.FloatTensor)

        r_xn, r_xa = filter_dummy_x(r_xnc, r_xnb, r_xac, r_xab, dim_dict)
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, dim_dict)
        x = torch.cat([xn, xa], 1)
        r_x = torch.cat([r_xn, r_xa], 1)
        cf_r_x = torch.cat([cf_r_xn, cf_r_xa], 1)

        una = torch.cat([uxn, uxa], 1)
        cf_una = torch.cat([cf_uxn, cf_uxa], 1)
        cf_una_dis = torch.cat([cf_uxn_dis, cf_uxa_dis], 1)

        una, cf_una, cf_una_dis = una.cpu().numpy(), cf_una.cpu().numpy(), cf_una_dis.cpu().numpy()
        x, r_x, cf_r_x = x.cpu().numpy(), r_x.cpu().numpy(), cf_r_x.cpu().numpy()
        a, y = a.cpu().numpy(), y.long().squeeze().cpu().numpy()

    return una, cf_una, cf_una_dis, x, r_x, cf_r_x, a, y


def full_ablation(x_src_tr, a_src_tr, y_src_tr, x_src_te, a_src_te, y_src_te, cf_r_x_src_te, x_trg_te,
                  a_trg_te, cf_r_x_trg_te, y_trg_te, cfg):
    df_full = pd.read_csv('result/synthetic/res_transfer_full.csv', index_col=[0, 1])
    mlp_full = NNClassifier(lr=1e-4, n_epochs=500, hidden_layer_sizes=cfg.mlp_hidden_layers)
    mlp_full.fit(np.concatenate([x_src_tr, a_src_tr.reshape(-1, 1)], axis=1), y_src_tr)
    src_fact_input = np.hstack([x_src_te, a_src_te.reshape(-1, 1)])
    src_counter_input = np.hstack([cf_r_x_src_te, 1-a_src_te.reshape(-1, 1)])
    trg_fact_input = np.hstack([x_trg_te, a_trg_te.reshape(-1, 1)])
    trg_counter_input = np.hstack([cf_r_x_trg_te, 1-a_trg_te.reshape(-1, 1)])
    res = no_encode_test(src_fact_input, src_counter_input, y_src_te, mlp_full)+no_encode_test(trg_fact_input, trg_counter_input, y_trg_te, mlp_full)
    df_full[f'{cfg.i}_full'] = res
    df_full.to_csv('result/synthetic/res_transfer_full.csv')


def fair_consis_test(i, x_src_te, cf_r_x_src_te, y_src_te, x_trg_te, cf_r_x_trg_te, y_trg_te, clf):
    fair_consis_df = pd.read_csv('result/fair_consis/fair_consis.csv', index_col=0)
    fair_consis_res = no_encode_test(x_src_te, cf_r_x_src_te, y_src_te, clf) + no_encode_test(x_trg_te, cf_r_x_trg_te, y_trg_te, clf)
    fair_consis_df[f'{i}_transfer_consis'] = fair_consis_res
    fair_consis_df.to_csv('result/fair_consis/fair_consis.csv')
