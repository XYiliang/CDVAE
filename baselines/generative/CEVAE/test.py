import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from gower import gower_matrix
from torch import nn

def guassian_kernel(X, Y, kernel_mul, kernel_num, fix_sigma=None):
    n_samples = int(X.shape[0]) + int(Y.shape[0])
    total = torch.cat([X, Y], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def compute_mmd(X, Y, batch_size=10000, part_size=0.1):
    import numpy as np
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
    loss = 0
    X = X[:batch_size, :] if X.shape[0] > batch_size else X
    Y = Y[:batch_size, :] if Y.shape[0] > batch_size else Y
    X_part_size = int(part_size * X.shape[0])
    Y_part_size = int(part_size * Y.shape[0])
    for i in range(10):
        x = X[int(X_part_size * i): int(X_part_size * (i+1)), :]
        y = Y[int(Y_part_size * i): int(Y_part_size * (i+1)), :]
        kernels = guassian_kernel(x, y, 1, 1)
        XX = torch.mean(kernels[:X_part_size, :X_part_size])
        YY = torch.mean(kernels[X_part_size:, X_part_size:])
        XY = torch.mean(kernels[:X_part_size, X_part_size:])
        loss += torch.mean(XX + YY - 2 * XY)
    return float(loss) / 10


def chi2_distance(A, B):
    tmp = np.square(A - B) / (A + B + 1e-10)
    chi = 0.5 * np.sum(tmp)

    return chi

def test(test_loader, real_cf_test_loader, args, logger):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    test_model = torch.load(model_path)
    test_model.to(device)
    test_model.eval()
    correct, _all, o1s, o2s, o3s, o4s, o1s_bin, o2s_bin, o3s_bin, o4s_bin, ys, ys_bin = \
        0, 0, None, None, None, None, None, None, None, None, None, None
    line = ''
    with torch.no_grad():
        for idx, (rc, rb, dc, db, a, y) in enumerate(test_loader):
            r = torch.cat([rc, rb], 1)
            d = torch.cat([dc, db], 1)
            loss_val, x_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, u_kl_loss_val, fair_loss_val\
                = test_model.calculate_loss(r.to(device), d.to(device), a.to(device), y.to(device))  # (*cur_batch)

            y_p_val = nn.Sigmoid()(y_p_val)
            y_p_counter_val = nn.Sigmoid()(y_p_counter_val)
            label_predicted = torch.eq(y_p_val.gt(0.5).byte(), y.to(device).byte())
            correct += torch.sum(label_predicted)
            _all += float(label_predicted.size(0))

            y_p_np = y_p_val.cpu().detach().numpy()
            y_cf_np = y_p_counter_val.cpu().detach().numpy()
            mask_a = np.where(a == 1, -1, 1)
            
            cf_effect = (y_cf_np - y_p_np) * mask_a
            cf_bin = (np.greater(y_cf_np, 0.5).astype(int) - np.greater(y_p_np, 0.5).astype(int)) * mask_a
            
            m = r.cpu().detach().numpy()[:, 1].reshape(-1, 1)
            mask1 = (m == 0).all(axis=1)
            mask2 = (m == 1).all(axis=1)

            o1 = cf_effect[mask1]
            o2 = cf_effect[mask2]

            o1s = np.concatenate((o1s, o1), axis=0) if idx != 0 else o1
            o2s = np.concatenate((o2s, o2), axis=0) if idx != 0 else o2

            o1_bin = cf_bin[mask1 == [True]]
            o2_bin = cf_bin[mask2 == [True]]

            o1s_bin = np.concatenate((o1s_bin, o1_bin), axis=0) if idx != 0 else o1_bin
            o2s_bin = np.concatenate((o2s_bin, o2_bin), axis=0) if idx != 0 else o2_bin
            ys = np.concatenate((ys, cf_effect), axis=0) if idx != 0 else cf_effect
            ys_bin = np.concatenate((ys_bin, cf_bin), axis=0) if idx != 0 else cf_bin


        logger.info('***data***')
        logger.info('cf: {:.4f}'.format(np.sum(ys) / ys.shape[0]))
        logger.info('o1: {:.8f}'.format(np.sum(o1s) / o1s.shape[0]))
        logger.info('o2: {:.8f}'.format(np.sum(o2s) / o2s.shape[0]))

        line += '############## generated result ##############\n'
        line += f'cf:{np.sum(ys) / ys.shape[0]:.4f}, o1:{np.sum(o1s) / o1s.shape[0]}, o2:{np.sum(o2s) / o2s.shape[0]}\n'
        line += f'cf_bin:{np.sum(ys_bin) / ys_bin.shape[0]}, o1_bin:{np.sum(o1s_bin) / o1s.shape[0]}, o2_bin:{np.sum(o2s_bin) / o2s.shape[0]}\n'
    
        test_df = extract_data(test_loader)
        generated_df = generate_data(test_loader, args)
        real_cf_df = extract_data(real_cf_test_loader)
        
        generated_factual_data = generated_df[['rc', 'rb', 'dc_0', 'dc_1', 'y']].to_numpy()
        generated_counter_data = generated_df[['rc', 'rb', 'dc_cf_0', 'dc_cf_1', 'y']].to_numpy()
        real_factual_data = test_df[['rc', 'rb', 'dc_0', 'dc_1', 'y']].to_numpy()
        real_counter_data = real_cf_df[['rc', 'rb', 'dc_0', 'dc_1', 'y']].to_numpy()
        generated_factual_data[:, [1, 3]] = generated_factual_data[:, [1, 3]].astype(np.int32)
        generated_counter_data[:, [1, 3]] = generated_counter_data[:, [1, 3]].astype(np.int32)
        real_factual_data[:, [1, 3]] = real_factual_data[:, [1, 3]].astype(np.int32)
        real_counter_data[:, [1, 3]] = real_counter_data[:, [1, 3]].astype(np.int32)
        
        factual_chi2 = chi2_distance(generated_factual_data, real_factual_data)
        counter_chi2 = chi2_distance(generated_counter_data, real_counter_data)
        factual_mmd = compute_mmd(generated_factual_data, real_factual_data, batch_size=float('inf'))
        counter_mmd = compute_mmd(generated_counter_data, real_counter_data, batch_size=float('inf'))
        factual_gower = gower_matrix(generated_factual_data, real_factual_data).mean().mean()
        counter_gower = gower_matrix(generated_counter_data, real_counter_data).mean().mean()
        real_total_effect = (real_cf_df['y'] - test_df['y']) * test_df['a'].apply(lambda x: -1 if x == 1 else 1)
        real_o1 = real_total_effect.loc[test_df[test_df['rb'] == 0].index]
        real_o2 = real_total_effect.loc[test_df[test_df['rb'] == 1].index]
        
        real_total_effect = np.sum(real_total_effect) / real_total_effect.shape[0]
        real_o1 = np.sum(real_o1) / real_o1.shape[0]
        real_o2 = np.sum(real_o2) / real_o2.shape[0]
        
        line += f'real total effect: {real_total_effect} real o1: {real_o1} real_o2: {real_o2}\n'
        line += f'real factual - generated factual: chi square = {factual_chi2}  MMD = {factual_mmd} GOWER = {factual_gower}\n'
        line += f'real counter - generated counter: chi square = {counter_chi2}  MMD = {counter_mmd} GOWER = {counter_gower}\n\n'
        
        result_dir = os.path.join('/mnt/cfs/SPEECH/xiayiliang/project/VAE/generative_models_syn/CEVAE/result', 'a_r_{:s}_a_d_{:s}_a_y_{:s}_a_f_{:s}_u_{:d}_run_{:d}'\
                                  .format(str(args.a_r), str(args.a_d), str(args.a_y), str(args.a_f), args.u_dim, args.run))
        file_dir = os.path.join(result_dir, 'whole_log.txt')
        
        if not os.path.exists(file_dir):
            f = open(file_dir, 'w')
        else:
            f = open(file_dir, 'a')
        
        f.write('a_r_{:s}_a_d_{:s}_a_y_{:s}_a_f_{:s}_u_{:d}_run_{:d}\n'\
                                  .format(str(args.a_r), str(args.a_d), str(args.a_y), str(args.a_f), args.u_dim, args.run))
        f.write(line)
        f.close()

def generate_data(loader, args):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # db为空
        rc_l, rb_l, dc_l, dc_cf_l, a_l, y_l, y_cf_l = [], [], [], [], [], [], []
        for idx, (rc, rb, dc, db, a, y) in enumerate(loader):
            rc_dim, dc_dim = rc.shape[1], dc.shape[1]
            r = torch.cat([rc, rb], 1)
            d = torch.cat([dc, db], 1)
            
            u_mu, u_logvar = model.q_u(r.to(device), d.to(device), a.to(device), y.to(device))
            u_prev = model.reparameterize(u_mu, u_logvar)
            r_hard, d_hard, d_cf_hard, y_hard, y_cf_hard = model.reconstruct_hard(u_prev, a.to(device))
            rc_hard, rb_hard = r_hard[:, :rc_dim], r_hard[:, rc_dim:]
            dc_hard = d_hard[:, :dc_dim]
            dc_cf_hard= d_cf_hard[:, :dc_dim]
            
            rc_l.append(rc_hard)
            rb_l.append(rb_hard)
            dc_l.append(dc_hard)
            dc_cf_l.append(dc_cf_hard)
            a_l.append(a)
            y_l.append(y_hard)
            y_cf_l.append(y_cf_hard)

        rc = torch.cat(rc_l, 0).cpu().numpy().squeeze()
        rb = torch.cat(rb_l, 0).cpu().numpy().squeeze()
        dc = torch.cat(dc_l, 0).cpu().numpy().squeeze()
        dc_cf = torch.cat(dc_cf_l, 0).cpu().numpy().squeeze()
        a = torch.cat(a_l, 0).cpu().numpy().squeeze()
        y = torch.cat(y_l, 0).cpu().numpy().squeeze()
        y_cf = torch.cat(y_cf_l, 0).cpu().numpy().squeeze()

        # for _ in [rc, rb, dc, dc_cf, a, y, y_cf]:
        #     print(_.shape)
        df = pd.DataFrame({'rc': rc, 'rb': rb, 'dc_0': dc[:,0].squeeze(), 'dc_1': dc[:,1].squeeze(), 'dc_cf_0': dc_cf[:,0].squeeze(), 'dc_cf_1': dc_cf[:,1].squeeze(), 'a': a, 'y': y, 'y_cf': y_cf})
        return df

def extract_data(loader):
    rc_l, rb_l, dc_l, y_l, a_l = [], [], [], [], []
    for idx, (rc, rb, dc, db, a, y) in enumerate(loader):
        rc_l.append(rc)
        rb_l.append(rb)
        dc_l.append(dc)
        a_l.append(a)
        y_l.append(y)
    rc = torch.cat(rc_l, 0).cpu().numpy().squeeze()
    rb = torch.cat(rb_l, 0).cpu().numpy().squeeze()
    dc = torch.cat(dc_l, 0).cpu().numpy().squeeze()
    a = torch.cat(a_l, 0).cpu().numpy().squeeze()
    y = torch.cat(y_l, 0).cpu().numpy().squeeze()

    df = pd.DataFrame({'rc': rc, 'rb': rb, 'dc_0': dc[:,0].squeeze(), 'dc_1': dc[:,1].squeeze(), 'a': a, 'y': y})
    return df