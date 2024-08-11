# coding = utf-8
import sys
import torch
import numpy as np

from timeit import default_timer
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from project.CDVAE.CDVAE_model.components.predictors import NNClassifier, NNRegressor
from CDVAE_model.single_domain.loss import CDVAELoss
from gower import gower_matrix
from utils import filter_dummy_x, draw_tSNE, get_new_colname
from metrics import cfd_bin, cfd_reg, compute_mmd, total_effect, clp_bin, clp_reg, chi2_distance

import warnings
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
                loss_func.optimize(data, epoch)
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
            
    torch.cuda.empty_cache()
    t.close()
    writer.close()
    
    model.eval()
    with torch.no_grad():
        delta_time = (default_timer() - start) / 60
        logger.info(f'Finished training after {delta_time:.1f} min.')

    return model


def test(model, loaders, vae_savename, result_log_df, logger, cfg):
    model.eval()
    train_loader, test_loader, dim_dict = loaders['train_loader'], loaders['test_loader'], loaders['dim_dict']

    with torch.no_grad():
        data_train = extract_data_single(train_loader, model, dim_dict)
        data_test = extract_data_single(test_loader, model, dim_dict)
        if cfg.dataset_name == 'synthetic':
            real_cf_data_test = extract_data_single(loaders['real_cf_test_loader'], model, dim_dict)

    una_train, cf_una_train, cf_una_dis_train, _, _, _, _, y_train, _, _, _, _ = data_train
    una_test, cf_una_test, cf_una_dis_test, x_test, r_x_test, cf_r_x_test, a_test, y_test, xn_test, _, r_y_p_test, cf_r_y_p_test = data_test
    
    if cfg.dataset_name == 'synthetic':
        
        te = (cf_r_y_p_test - r_y_p_test) * np.where(a_test == 1, -1, 1)
        m = xn_test[:, 1].reshape(-1, 1)
        mask1 = (m == 0).all(axis=1)
        mask2 = (m == 1).all(axis=1)
        
        o1 = te[mask1]
        o2 = te[mask2]
        
        real_cf_una_test, _, _, real_cf_x, _, _, _, real_cf_y, _, _, _, _ = real_cf_data_test
        r_y_test = np.where(r_y_p_test >= 0.5, 1, 0)
        cf_r_y_test = np.where(cf_r_y_p_test >= 0.5, 1, 0)
        te_bin = (cf_r_y_test - r_y_test) * np.where(a_test == 1, -1, 1)
        o1_bin = te_bin[mask1]
        o2_bin = te_bin[mask2]
        
        generated_factual_data = np.hstack([r_x_test, r_y_test])
        generated_counter_data = np.hstack([cf_r_x_test, cf_r_y_test])
        real_factual_data = np.hstack([x_test, y_test])
        real_counter_data = np.hstack([real_cf_x, real_cf_y])
        generated_factual_data[:, [1, 3]] = generated_factual_data[:, [1, 3]].astype(np.int32)
        generated_counter_data[:, [1, 3]] = generated_counter_data[:, [1, 3]].astype(np.int32)
        real_factual_data[:, [1, 3]] = real_factual_data[:, [1, 3]].astype(np.int32)
        real_counter_data[:, [1, 3]] = real_counter_data[:, [1, 3]].astype(np.int32)
        
        chi2_real_factual = chi2_distance(real_factual_data, generated_factual_data)
        chi2_real_counter = chi2_distance(real_counter_data, generated_counter_data)
        mmd_real_factual = compute_mmd(real_factual_data, generated_factual_data, batch_size=float('inf'))
        mmd_real_counter = compute_mmd(real_counter_data, generated_counter_data, batch_size=float('inf'))
        gower_real_factual = gower_matrix(real_factual_data, generated_factual_data).mean().mean()
        gower_real_counter = gower_matrix(real_counter_data, generated_counter_data).mean().mean()
        
        te = np.sum(te) / te.shape[0]
        te_bin = np.sum(te_bin) / te_bin.shape[0]
        o1 = np.sum(o1) / o1.shape[0]
        o2 = np.sum(o2) / o2.shape[0]
        o1_bin = np.sum(o1_bin) / o1_bin.shape[0]
        o2_bin = np.sum(o2_bin) / o2_bin.shape[0]
        
        with open('/mnt/cfs/SPEECH/xiayiliang/project/VAE/generative_models_syn/CDVAE_ours/CDVAE_gen_result.csv', 'w') as f:
            f.write(f'cf: {te} o1:{o1} o2:{o2}\n')
            f.write(f'cf_bin:{te_bin} o1_bin:{o1_bin} o2_bin:{o2_bin}\n')
            f.write(f'real factual - generated factual: chi square = {chi2_real_factual}  MMD = {mmd_real_factual} GOWER = {gower_real_factual}\n')
            f.write(f'real counter - generated counter: chi square = {chi2_real_counter}  MMD = {mmd_real_counter}  GOWER = {gower_real_counter}')
        sys.exit()
        if cfg.draw_tSNE:
            # m = compute_mmd(generated_factual_data)
            # real factual - generated factual
            draw_tSNE(generated_factual_data, real_factual_data, cfg, save_name=f'real factual-generated factual.tif',comparison='src-trg', mmd_val=mmd_real_factual)
            draw_tSNE(generated_counter_data, real_counter_data, cfg, save_name=f'real counterfactual - generated counterfactual.tif',comparison='src-trg', mmd_val=mmd_real_counter)

        
        
    if cfg.dataset_name == 'law_school':
        if cfg.use_torch_mlp:
            mlp_model = NNRegressor(lr=cfg.mlp_lr, n_epochs=cfg.clf_mlp_epochs,
                                    hidden_layer_sizes=cfg.mlp_hidden_layers)
        else:
            mlp_model = MLPRegressor(hidden_layer_sizes=cfg.mlp_hidden_layers, learning_rate_init=cfg.mlp_lr,
                                     max_iter=cfg.clf_mlp_epochs, random_state=cfg.seed, activation='relu',
                                     verbose=True)
        lr_model = LinearRegression()
        svm_model = SVR(kernel='poly')

    else:
        if cfg.use_torch_mlp:
            mlp_model = NNClassifier(lr=cfg.mlp_lr, n_epochs=cfg.clf_mlp_epochs,
                                     hidden_layer_sizes=cfg.mlp_hidden_layers)
        else:
            mlp_model = MLPClassifier(hidden_layer_sizes=cfg.mlp_hidden_layers, learning_rate_init=cfg.mlp_lr,
                                      max_iter=cfg.clf_mlp_epochs, random_state=cfg.seed, activation='relu',
                                      verbose=True)
        lr_model = LogisticRegression(random_state=cfg.seed, penalty='l2')
        svm_model = SVC(kernel=cfg.svm_kernel, probability=True, random_state=cfg.seed)  # 使用概率估计

    total_metrics_list, total_metrics_dict = [], {}
    models = {'MLP': mlp_model}
    total_metrics_list += [np.nan] * 22

    # Train
    for name, clf in models.items():
        logger.info(f'{name} Train and test.')
        # Train data
        if name == 'MLP' and cfg.use_torch_mlp:
            clf.fit(una_train, y_train, una_test, y_test)
        else:
            clf.fit(una_train, y_train)
        # Test data
        task = 'regression' if cfg.dataset_name == 'law_school' else 'classification'
        metrics_dict, metrics_list = calc_metrics_single(clf, una_test, cf_una_test, cf_una_dis_test, y_test, x_test,
                                                         r_x_test, cf_r_x_test, task)

        total_metrics_dict[name] = metrics_dict
        total_metrics_list += metrics_list

    col_name = vae_savename[:-5] + f'_hlayers{cfg.mlp_hidden_layers}).pth'
    col_name = get_new_colname(col_name, result_log_df)

    if cfg.dataset_name == 'synthetic':
        total_metrics_list += calc_real_cf_metrics(mlp_model, una_test, real_cf_una_test)

    result_log_df[col_name] = total_metrics_list
    models_dict = {'LR': lr_model, 'SVM': svm_model, 'MLP': mlp_model}

    return total_metrics_dict, result_log_df, models_dict


def extract_data_single(loader, trained_vae, dim_dict):
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
        factual, counter, cf_dis = trained_vae(xn, xa, a)
        uxn, uxa, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_uxn, cf_uxa, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, cf_r_y = counter
        cf_uxn_dis, cf_uxa_dis = cf_dis

        r_xnb = torch.sigmoid(r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        r_xab = torch.sigmoid(r_xab).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xnb = torch.sigmoid(cf_r_xnb).gt(0.5).byte().type(torch.FloatTensor)
        cf_r_xab = torch.sigmoid(cf_r_xab).gt(0.5).byte().type(torch.FloatTensor)
        r_y_p = torch.sigmoid(r_y).numpy()
        cf_r_y_p = torch.sigmoid(cf_r_y).numpy()

        r_xn, r_xa = filter_dummy_x(r_xnc, r_xnb, r_xac, r_xab, dim_dict)
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, dim_dict)

        x = torch.cat([xn, xa], 1)
        r_x = torch.cat([r_xn, r_xa], 1)
        cf_r_x = torch.cat([cf_r_xn, cf_r_xa], 1)

        una = torch.cat([uxn, uxa], 1)
        cf_una = torch.cat([cf_uxn, cf_uxa], 1)
        cf_una_dis = torch.cat([cf_uxn_dis, cf_uxa_dis], 1)

        una, cf_una, cf_una_dis = una.cpu().numpy(), cf_una.cpu().numpy(), cf_una_dis.cpu().numpy()
        x, r_x, cf_r_x = x.cpu(), r_x.cpu(), cf_r_x.cpu()
        a, y = a.cpu().numpy(), y.squeeze().cpu().numpy()

        return una, cf_una, cf_una_dis, x, r_x, cf_r_x, a, y.reshape(-1, 1), xn, cf_r_xn, r_y_p.reshape(-1, 1), cf_r_y_p.reshape(-1, 1)


# def full_ablation(input_train, input_test, input_test_counter, y_train, y_test, cfg):
#     clf = NNRegressor(lr=cfg.mlp_lr, n_epochs=cfg.clf_mlp_epochs, hidden_layer_sizes=cfg.mlp_hidden_layers)
#     clf.fit(input_train, y_train)
#
#     pred_factual_p = clf.predict_proba(input_test)
#     pred_counter_p = clf.predict_proba(input_test_counter)
#     pred_factual = clf.predict(input_test)
#     pred_counter = clf.predict(input_test_counter)
#
#     acc = accuracy_score(pred_factual, y_test)
#     auc = roc_auc_score(pred_factual, y_test)
#     cfd = cfd_bin()

def calc_real_cf_metrics(clf, una, real_cf_una):
    pred_factual = clf.predict(una)
    pred_counter = clf.predict(real_cf_una)
    pred_factual_prob = clf.predict_proba(una)
    pred_counter_prob = clf.predict_proba(real_cf_una)
    te = total_effect(pred_factual_prob, pred_counter_prob)
    cfd = cfd_bin(pred_factual, pred_counter)
    clp = clp_bin(pred_factual_prob, pred_counter_prob)
    return [te, cfd, clp]



def calc_metrics_single(clf, una, cf_una, cf_una_dis, y, x, r_x, cf_r_x, task='classification'):
    assert task in ['classification', 'regression'], 'Wrong Task'

    pred_factual = clf.predict(una)
    pred_counter_u = clf.predict(cf_una)
    pred_counter_dis = clf.predict(cf_una_dis)

    if task == 'classification':
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
    else:
        acc = mean_squared_error(y, pred_factual) ** 0.5  # RMSE
        auc = np.nan
        te_u = total_effect(pred_factual, pred_counter_u)  # 这个定义是Counterfactually Fair Representation给出的,就是
        te_dis = total_effect(pred_factual, pred_counter_dis)
        cfd_u = cfd_reg(pred_factual, pred_counter_u)
        cfd_dis = cfd_reg(pred_factual, pred_counter_dis)
        clp_u = clp_reg(pred_factual, pred_counter_u)
        clp_dis = clp_reg(pred_factual, pred_counter_dis)

    mmd_real_factual = compute_mmd(x, r_x)
    mmd_real_counter = compute_mmd(x, cf_r_x)
    mmd_counter_factual = compute_mmd(cf_r_x, r_x)

    metrics_dict = {'Acc': acc, 'AUC': auc, 'TE_u': te_u, 'TE_dis': te_dis, 'CFD_u': cfd_u,
                    'CFD_dis': cfd_dis, 'CLP_u': clp_u, 'clp_dis': clp_dis, 'mmd_rf': mmd_real_factual,
                    'mmd_rc': mmd_real_counter, 'mmd_cf': mmd_counter_factual}
    metrics_list = [acc, auc, te_u, te_dis, cfd_u, cfd_dis, clp_u, clp_dis, mmd_real_factual, mmd_real_counter,
                    mmd_counter_factual]

    return metrics_dict, metrics_list
