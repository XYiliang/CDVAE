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
    tmp = np.power((A - B), 2) / (A + B + 1e-10)
    chi = 0.5 * np.sum(tmp)

    return chi

def test(test_loader, real_cf_loader, args, logger):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    test_model = torch.load(model_path)
    test_model.to(device)
    test_model.eval()
    correct, _all, o1s, o2s, o3s, o4s, o1s_bin, o2s_bin, o3s_bin, o4s_bin, ys, ys_bin = \
        0, 0, None, None, None, None, None, None, None, None, None, None
    line = ''
    with torch.no_grad():
        for idx, (x, o, a, y) in enumerate(test_loader):
            loss_val, xo_recon_loss_val, y_recon_loss_val, y_p_val, y_p_counter_val, mmd_loss_val, mmd_a_loss_val, \
            fair_loss_val = test_model.calculate_loss(x.to(device), o.to(device), a.to(device), y.to(device))  # (*cur_batch)

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

            m = o.cpu().detach().numpy()
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
        line += f'cf_bin:{np.sum(ys_bin) / ys_bin.shape[0]}, o1_bin:{np.sum(o1s_bin) / o1s_bin.shape[0]}, o2_bin:{np.sum(o2s_bin) / o2s_bin.shape[0]}\n'
        
        test_df = extract_data(test_loader)
        generated_df = generate_data(test_loader, args)
        real_cf_df = extract_data(real_cf_loader)
        
        generated_factual_data = np.hstack([generated_df['x'][:, 0].reshape(-1, 1), generated_df['o'], generated_df['x'][:, 1:], generated_df['y']])
        generated_counter_data = np.hstack([generated_df['x_cf'][:, 0].reshape(-1, 1), generated_df['o_cf'], generated_df['x'][:, 1:], generated_df['y_cf']])
        real_factual_data = np.hstack([test_df['x'][:, 0].reshape(-1, 1), test_df['o'], test_df['x'][:, 1:], test_df['y']])
        real_counter_data = np.hstack([real_cf_df['x'][:, 0].reshape(-1, 1), real_cf_df['o'], test_df['x'][:, 1:], real_cf_df['y']])
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
        
        real_total_effect = (real_cf_df['y'] - test_df['y']) * np.where(test_df['a'] == 1, -1, 1)
        mask1 = np.where(test_df['o'] == 0)
        mask2 = np.where(test_df['o'] == 1)
        real_o1 = real_total_effect[mask1]
        real_o2 = real_total_effect[mask2]
        
        real_total_effect = np.sum(real_total_effect) / real_total_effect.shape[0]
        real_o1 = np.sum(real_o1) / real_o1.shape[0]
        real_o2 = np.sum(real_o2) / real_o2.shape[0]
        
        line += f'real total effect: {real_total_effect} real o1: {real_o1} real_o2: {real_o2}\n'
        line += f'real factual - generated factual: chi square = {factual_chi2}  MMD = {factual_mmd} GOWER = {factual_gower}\n'
        line += f'real counter - generated counter: chi square = {counter_chi2}  MMD = {counter_mmd} GOWER = {counter_gower}\n\n'
        
        save_path = os.path.join('/mnt/cfs/SPEECH/xiayiliang/project/VAE/generative_models_syn/mCEVAE/result/', 'a_x_{:s}_a_y_{:s}_a_f_{:s}_mmd_{:s}_mmd_a_{:s}_run_{:d}'.format(str(args.a_x), str(args.a_y), str(args.a_f), str(args.mmd), str(args.mmd_a), args.run))
        file_dir = os.path.join(save_path , 'whole_log.txt')
        if not os.path.exists(file_dir):
            f = open(file_dir, 'w')
        else:
            f = open(file_dir, 'a')
        
        f.write(line)
        f.close()


def generate_data(loader, args, dataset='train'):
    device = args.device
    model_path = os.path.join(args.save_path, 'model.pth')
    model = torch.load(model_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_l, o_l, a_l, y_l = [], [], [], []
        x_cf_l, o_cf_l, y_cf_l = [], [], []
        for idx, (x, o, a, y) in enumerate(loader):
            u_mu_encoder, u_logvar_encoder, u, u0, u1 = model.forward(o.to(device), a.to(device), x.to(device))
            o_hard, o_cf_hard, x_hard, x_cf_hard, y_hard, y_cf_hard = model.reconstruct_hard(a.to(device), u)

            x_l.append(x_hard)
            x_cf_l.append(x_cf_hard)
            o_l.append(o_hard)
            o_cf_l.append(o_cf_hard)
            a_l.append(a)
            y_l.append(y_hard)
            y_cf_l.append(y_cf_hard)
            

        x, x_cf = torch.cat(x_l, 0).cpu().numpy(), torch.cat(x_cf_l, 0).cpu().numpy()
        o, o_cf = torch.cat(o_l, 0).cpu().numpy(), torch.cat(o_cf_l, 0).cpu().numpy()
        a = torch.cat(a_l, 0).cpu().numpy()
        y, y_cf = torch.cat(y_l, 0).cpu().numpy(), torch.cat(y_cf_l, 0).cpu().numpy()
    
        df = {'x': x, 'x_cf': x_cf, 'o': o, 'o_cf': o_cf, 'a': a, 'y': y, 'y_cf': y_cf}
        return df
    
def extract_data(loader):
    x_l, o_l, a_l, y_l = [], [], [], []
    for idx, (x, o, a, y) in enumerate(loader):
        x_l.append(x)
        o_l.append(o)
        a_l.append(a)
        y_l.append(y)
    x = torch.cat(x_l, 0).cpu().numpy()
    o = torch.cat(o_l, 0).cpu().numpy()
    a = torch.cat(a_l, 0).cpu().numpy()
    y = torch.cat(y_l, 0).cpu().numpy()

    df = {'x': x, 'o': o, 'a': a, 'y': y}
    return df
        
