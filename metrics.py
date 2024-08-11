# coding = uft-8
import numpy
import numpy as np
import torch
from utils import tensor2numpy

@tensor2numpy
def equal_opportunity(y_true, y_pred, sens):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    sens = np.asarray(sens).reshape(-1, 1)

    data = np.concatenate((y_pred, y_true, sens), axis=1)

    y1_a0 = data[(data[:, 1] == 1) & (data[:, 2] == 0)]
    y1_a1 = data[(data[:, 1] == 1) & (data[:, 2] == 1)]

    TPR_a0 = (y1_a0[y1_a0[:, 0] == 1].shape[1] / y1_a0.shape[0] if y1_a0.shape[0] != 0 else 0)
    TPR_a1 = (y1_a1[y1_a1[:, 0] == 1].shape[1] / y1_a1.shape[0] if y1_a1.shape[0] != 0 else 0)

    return np.abs(TPR_a0 - TPR_a1)

@tensor2numpy
def equalized_odds(y_true, y_pred, sens):
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    sens = np.asarray(sens).reshape(-1, 1)

    data = np.concatenate((y_pred, y_true, sens), axis=1)

    y0_a0 = data[(data[:, 1] == 0) & (data[:, 2] == 0)]
    y1_a0 = data[(data[:, 1] == 1) & (data[:, 2] == 0)]
    y0_a1 = data[(data[:, 1] == 0) & (data[:, 2] == 1)]
    y1_a1 = data[(data[:, 1] == 1) & (data[:, 2] == 1)]

    TNR_a0 = (y0_a0[y0_a0[:, 0] == 0].shape[1] / y0_a0.shape[0] if y0_a0.shape[0] != 0 else 0)
    TNR_a1 = (y0_a1[y0_a1[:, 0] == 0].shape[1] / y0_a1.shape[0] if y0_a1.shape[0] != 0 else 0)
    TPR_a0 = (y1_a0[y1_a0[:, 0] == 1].shape[1] / y1_a0.shape[0] if y1_a0.shape[0] != 0 else 0)
    TPR_a1 = (y1_a1[y1_a1[:, 0] == 1].shape[1] / y1_a1.shape[0] if y1_a1.shape[0] != 0 else 0)

    return np.abs(TPR_a0 - TPR_a1) + np.abs(TNR_a0 - TNR_a1)

@tensor2numpy
def demographic_parity(y_pred, sens):
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    sens = np.asarray(sens).reshape(-1, 1)
    data = np.concatenate((y_pred, sens), 1)

    a0 = data[data[:, 1] == 0]
    a1 = data[data[:, 1] == 1]

    p_y1_a0 = (a0[a0[:, 0] == 1].shape[0] / a0.shape[0] if a0.shape[0] != 0 else 0)
    p_y1_a1 = (a1[a1[:, 0] == 1].shape[0] / a1.shape[0] if a1.shape[0] != 0 else 0)
    # print(a0, a1)
    # print(p_y1_a0, p_y1_a1)
    return np.abs(p_y1_a0 - p_y1_a1)

@tensor2numpy
def absolute_total_effect(y_factual_p, y_counter_p):
    # Absolute Difference in predicted PROBABILITY
    y_factual_p = y_factual_p.squeeze()
    y_counter_p = y_counter_p.squeeze()
    return np.mean(np.abs(y_factual_p - y_counter_p))

@tensor2numpy
def cfd_bin(y_factual, y_counter):
    # Difference in predicted label(classification task)
    y_factual = y_factual.squeeze().astype(int)
    y_counter = y_counter.squeeze().astype(int)

    return np.mean(np.abs(y_factual - y_counter))

@tensor2numpy
def cfd_reg(y_factual, y_counter):
    # Difference in predicted label(regression task)
    y_factual = y_factual.squeeze()
    y_counter = y_counter.squeeze()
    return np.sqrt(np.mean((y_factual - y_counter) ** 2))  # RMSE

@tensor2numpy
def clp_reg(y_factual, y_counter):
    return cfd_reg(y_factual, y_counter) ** 2  # MSE


def compute_mmd(X, Y, batch_size=10000, part_size=0.1):
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
    
    if isinstance(X, numpy.ndarray):
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

@tensor2numpy
def chi2_distance(A, B):
    tmp = np.power((A - B), 2) / (A + B + 1e-10)
    chi = 0.5 * np.sum(tmp)

    return chi



