# coding = utf-8
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import warnings

from tqdm import tqdm
from timeit import default_timer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from gower import gower_matrix

from CDVAE_model.single_domain.loss import CDVAELoss
from CDVAE_model.components.predictors import NNClassifier, NNRegressor
from utils import filter_dummy_x, draw_tSNE, get_new_colname
from metrics import cfd_bin, cfd_reg, compute_mmd, absolute_total_effect, clp_reg


warnings.filterwarnings("ignore")


def train(model, vae_epochs, writer, train_loader, logger, device, dim_dict, cfg):
    model.to(device)
    loss_func = CDVAELoss(model, device, dim_dict, cfg)

    # Train
    logger.info('--------------Start VAE Training--------------')
    start = default_timer()
    model.train()
    t = tqdm(range(vae_epochs), leave=True, position=0, unit='epoch')

    for epoch in t:
        torch.cuda.empty_cache()
        recon_loss, KLD, tc_loss, css_loss, VAE_loss, disc_tc_loss = 0, 0, 0, 0, 0, 0
        for _, data in enumerate(train_loader):
            recon_loss_bat, KLD_bat, tc_loss_bat, css_loss_bat, VAE_loss_bat, disc_tc_loss_bat = \
                loss_func.optimize(data)
            recon_loss += recon_loss_bat.detach().item()
            KLD += KLD_bat.detach().item()
            tc_loss += tc_loss_bat.detach().item()
            css_loss += css_loss_bat.detach().item()
            VAE_loss += VAE_loss_bat.detach().item()
            disc_tc_loss += disc_tc_loss_bat.detach().item()

        N = len(train_loader)
        recon_loss /= N
        KLD /= N
        tc_loss /= N
        css_loss /= N
        VAE_loss /= N
        disc_tc_loss /= N


        t.set_postfix({'VAE Loss': round(float(VAE_loss), 4)})
        losses = {'Reconstruction Loss': recon_loss, 'KLD': KLD, 'Total Correlation Loss': tc_loss,
                  'Consistency': css_loss, 'VAE Loss': VAE_loss, 'Discriminator TC Loss': disc_tc_loss}
        for k, v in losses.items():
            writer.add_scalar(k, v, epoch)

    t.close()
    writer.close()
    torch.cuda.empty_cache()
    
    model.eval()
    with torch.no_grad():
        delta_time = (default_timer() - start) / 60
        logger.info(f'Finished training after {delta_time:.1f} min.')
        
    return model


def test(model, loaders, vae_savename, result_log_df, logger, cfg):
    model.eval()
    
    with torch.no_grad():
        real_train, gen_factual_train, gen_counter_train, distilled_train = extract_data(loaders['train_loader'], model, loaders['dim_dict'])
        real_test, gen_factual_test, gen_counter_test, distilled_test = extract_data(loaders['test_loader'], model, loaders['dim_dict'])
        if cfg.dataset_name == 'synthetic':
            real_cf_data_test = extract_real_cf(loaders['real_cf_test_loader'], model, loaders['dim_dict'])
            
    if cfg.draw_tSNE:
        # U and distilled U
        mmd_U_disU = compute_mmd(gen_factual_test['U'], distilled_test['U'])
        draw_tSNE(
            gen_factual_test['U'],
            distilled_test['U'],
            cfg,
            save_name=f'U - distilled U.tif',
            comparison='src-trg',
            mmd_val=mmd_U_disU
        )
        
    if cfg.task == 'regression':
        if cfg.use_torch_mlp:
            mlp_model = NNRegressor(
                lr=cfg.mlp_lr,
                n_epochs=cfg.clf_mlp_epochs,
                hidden_layer_sizes=cfg.mlp_hidden_layers
            )
        else:
            mlp_model = MLPRegressor(
                hidden_layer_sizes=cfg.mlp_hidden_layers,
                learning_rate_init=cfg.mlp_lr,
                max_iter=cfg.clf_mlp_epochs,
                random_state=cfg.seed,
                activation='relu',
                verbose=True
            )
        lr_model = LinearRegression()
        svm_model = SVR(kernel='poly')
    else:
        if cfg.use_torch_mlp:
            print('torch')
            mlp_model = NNClassifier(
                lr=cfg.mlp_lr,
                n_epochs=cfg.clf_mlp_epochs,
                hidden_layer_sizes=cfg.mlp_hidden_layers
            )
        else:
            print('no torch')
            mlp_model = MLPClassifier(
                hidden_layer_sizes=cfg.mlp_hidden_layers,
                learning_rate_init=cfg.mlp_lr,
                max_iter=cfg.clf_mlp_epochs,
                random_state=cfg.seed, activation='relu',
                verbose=True
            )
        lr_model = LogisticRegression(random_state=cfg.seed, penalty='l2')
        svm_model = SVC(kernel=cfg.svm_kernel, probability=True, random_state=cfg.seed)

    total_metrics_list, total_metrics_dict = [], {}
    # Modify this dict and dataframe in utils.ini_result_dataframe to add new predictive models
    models = {'MLP': mlp_model}

    # Train predictor
    for name, predictor in models.items():
        logger.info(f'{name} Train and test.')
        U_train, U_valid, y_train, y_valid = train_test_split(distilled_train['U'],
                                                              real_train['y'],
                                                              random_state=cfg.seed,
                                                              train_size=0.9)
        if name == 'MLP' and cfg.use_torch_mlp:
            predictor.fit(U_train, y_train, U_valid, y_valid)
        else:
            predictor.fit(U_train, y_train)
            
        # Test data
        predictor.eval()
        predictor = predictor.cpu()
        metrics_dict, metrics_list = calc_metrics_single(
            predictor=predictor,
            U=gen_factual_test['U'],
            cf_U_dis=distilled_test['U'],
            x=real_test['x'],
            r_x=gen_factual_test['x'],
            cf_r_x=gen_counter_test['x'],
            y=real_test['y'],
            task=cfg.task
        )

        total_metrics_dict[name] = metrics_dict
        total_metrics_list += metrics_list

    col_name = vae_savename.replace(').pth', f'_hlayers{cfg.mlp_hidden_layers}).pth')
    col_name = get_new_colname(col_name, result_log_df)

    if cfg.dataset_name == 'synthetic':
        total_metrics_list += calc_real_cf_metrics(
            mlp_model,
            gen_factual_test['U'],
            real_cf_data_test['U']
        )
    result_log_df[col_name] = total_metrics_list

    return total_metrics_dict, result_log_df, models


def extract_data(loader, model, dim_dict):
    """ Extract data from loaders
    Returns:
        dicts of 2-D numpy.ndarrays
    """
    device = next(model.parameters()).device
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
        xn, xa, a, y = xn.to(device), xa.to(device), a.to(device), y.to(device)

        factual, counter, cf_dis = model(xn, xa, a)
        un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_un, cf_ua, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, cf_r_y = counter
        cf_un_dis, cf_ua_dis = cf_dis

        r_xnb = torch.sigmoid(r_xnb).gt(0.5).byte()
        r_xab = torch.sigmoid(r_xab).gt(0.5).byte()
        cf_r_xnb = torch.sigmoid(cf_r_xnb).gt(0.5).byte()
        cf_r_xab = torch.sigmoid(cf_r_xab).gt(0.5).byte()
        r_xn, r_xa = filter_dummy_x(r_xnc, r_xnb, r_xac, r_xab, dim_dict)
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, dim_dict)
        
        r_y_p = torch.sigmoid(r_y).reshape(-1, 1)
        cf_r_y_p = torch.sigmoid(cf_r_y).reshape(-1, 1)
        r_y = r_y_p.gt(0.5).byte().reshape(-1, 1)
        cf_r_y = cf_r_y_p.gt(0.5).byte().reshape(-1, 1)
        
        x = torch.cat([xn, xa], 1)
        r_x = torch.cat([r_xn, r_xa], 1)
        cf_r_x = torch.cat([cf_r_xn, cf_r_xa], 1)

        U = torch.cat([un, ua], 1)
        cf_U = torch.cat([cf_un, cf_ua], 1)
        cf_U_dis = torch.cat([cf_un_dis, cf_ua_dis], 1)

        real = {'xn': x, 'xa': xa, 'x': x, 'a': a, 'y': y}
        gen_factual = {'xn': r_xn, 'xa': r_xa, 'x': r_x, 'y': r_y, 'un': un, 'ua': ua, 'U': U}
        gen_counter = {'xn': cf_r_xn, 'xa': cf_r_xa, 'x': cf_r_x, 'y': cf_r_y, 'un': cf_un, 'ua': cf_ua, 'U': cf_U}
        distilled = {'un': cf_un_dis, 'ua': cf_ua_dis, 'U': cf_U_dis}
        
        for pack in [real, gen_factual, gen_counter, distilled]:
            for k, v in pack.items():
                pack[k] = v.cpu()

        return real, gen_factual, gen_counter, distilled
    

def extract_real_cf(loader, model, dim_dict):
    device = next(model.parameters()).device
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
        xn, xa, a, y = xn.to(device), xa.to(device), a.to(device), y.to(device)
        un, ua, _, _ = model.encode(xn, xa, a)
        x = torch.cat([xn, xa], 1)
        U = torch.cat([un, ua], 1)
        real_cf = {'xn': xn, 'xa': xa, 'x': x, 'a': a, 'y': y, 'un': un, 'ua': ua, 'U': U}
        
        for k, v in real_cf.items():
            real_cf[k] = v.cpu()
            
        return real_cf


def calc_real_cf_metrics(predictor , U, real_cf_U):
    pred_factual = predictor.predict(U)
    pred_counter = predictor.predict(real_cf_U)
    pred_factual_prob = predictor.predict_proba(U)
    pred_counter_prob = predictor.predict_proba(real_cf_U)
    te = absolute_total_effect(pred_factual_prob, pred_counter_prob)
    cfd = cfd_bin(pred_factual, pred_counter)
    return [te, cfd]



def calc_metrics_single(predictor, U, cf_U_dis, x, r_x, cf_r_x, y, task='classification'):
    pred_factual = predictor.predict(U)
    pred_counter_dis = predictor.predict(cf_U_dis)
    
    mmd_real_factual = compute_mmd(x, r_x)
    mmd_real_counter = compute_mmd(x, cf_r_x)
    mmd_counter_factual = compute_mmd(cf_r_x, r_x)

    if task == 'classification':
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
    else:
        rmse = mean_squared_error(y, pred_factual) ** 0.5
        rmse_cf = clp_reg(pred_factual, pred_counter_dis) ** 0.5
        cfd = cfd_reg(pred_factual, pred_counter_dis)
        metrics_dict = {'RMSE': rmse, 'RMSE_cf': rmse_cf, 'CFD': cfd,
                        'mmd_rf': mmd_real_factual,'mmd_rc': mmd_real_counter,
                        'mmd_cf': mmd_counter_factual}
        metrics_list = [rmse, rmse_cf, cfd, mmd_real_factual, mmd_real_counter,
                        mmd_counter_factual]

    return metrics_dict, metrics_list
