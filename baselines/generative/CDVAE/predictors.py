# coding = utf-8
import numpy
import numpy as np
import torch
from torch import nn, optim, overrides
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm
import re
import torch.nn.functional as F
from utils import filter_dummy_x
from sklearn.metrics import accuracy_score
from utils import ini_weight
from sklearn.exceptions import NotFittedError
from overrides import overrides
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from metrics import cfd_bin, cfd_reg, compute_mmd, total_effect, clp_bin, clp_reg


class NNClassifier(nn.Module):
    def __init__(self, lr, n_epochs, hidden_layer_sizes=(100,), early_stop_threshold=None):
        super(NNClassifier, self).__init__()
        self.model = None
        self.h_layers = hidden_layer_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cuda')
        self.model_fitted = False
        self.early_stop_val = early_stop_threshold

        ini_weight(self.modules())

    def forward(self, x):
        return self.model(x)

    def fit(self, x, y, x_valid=None, y_valid=None):
        self._init_model(x.shape[1], 1)
        x = torch.from_numpy(x).type(torch.FloatTensor) if isinstance(x, numpy.ndarray) else x
        y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1) if isinstance(y, numpy.ndarray) else y
        x, y = x.to(self.device), y.to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        self.model.to(self.device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=10000, shuffle=True)

        if x_valid is not None:
            x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor).cuda() if isinstance(x_valid, numpy.ndarray) else x_valid
            y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor).cuda().view(-1, 1) if isinstance(y_valid, numpy.ndarray) else y_valid

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        self.model.train()
        for _ in t:
            acc, total_loss, N = 0, 0, 0
            for x, y in train_loader:
                y_pred_logit = self.model(x)

                loss = loss_fn(y_pred_logit, y)
                total_loss += loss

                y_pred_prob = torch.sigmoid(y_pred_logit)
                acc += accuracy(y_pred_prob, y, task='binary')

                self.model.zero_grad()
                loss.backward()
                opt.step()
                N += 1

            if x_valid is not None:
                valid_acc = accuracy(torch.sigmoid(self.model(x_valid)), y_valid, task='binary')
                t.set_postfix({'CLF Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4),
                               'Valid Acc Src': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'CLF Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4)})

        self.model.eval()
        self.model_fitted = True

    def predict(self, x):
        # if not self.model_fitted:
        #     raise NotFittedError("This MLP instance is not fitted yet. Call 'fit' with "
        #                          "appropriate arguments before using this estimator.")
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device) if isinstance(x, numpy.ndarray) else x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            return torch.sigmoid(out).gt(0.5).byte().cpu().numpy()

    def predict_proba(self, x):
        # if not self.model_fitted:
        #     raise NotFittedError("This MLP instance is not fitted yet. Call 'fit' with "
        #                          "appropriate arguments before using this estimator.")
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device) if isinstance(x, numpy.ndarray) else x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            return torch.sigmoid(out).cpu().numpy()

    def _init_weight(self):
        for _ in self.modules():
            if isinstance(_, nn.Linear) or isinstance(_, nn.Conv2d) or isinstance(_, nn.ConvTranspose2d):
                nn.init.xavier_normal_(_.weight.data)

    def _init_model(self, i_dim, o_dim):
        self.model = nn.Sequential(
            nn.Linear(i_dim, self.h_layers[0]),
            nn.ReLU(),
        )
        for i in range(len(self.h_layers) - 1):
            self.model.add_module(f"Hidden_{i+1}", nn.Linear(self.h_layers[i], self.h_layers[i+1]))
            self.model.add_module(f"ReLU_Hidden_{i+1}", nn.ReLU())
        self.model.add_module("Output", nn.Linear(self.h_layers[-1], o_dim))

        self._init_weight()


class NNClassifierWithFairReg(NNClassifier):
    def __init__(self, lr, n_epochs, hidden_layer_sizes=(100,), parm_cf=1.0, parm_clp=1.0):
        super(NNClassifierWithFairReg, self).__init__(lr, n_epochs, hidden_layer_sizes)
        self.parm_clp = parm_clp
        self.parm_cf = parm_cf

    def fit(self, x, cf_x, y, x_valid=None, y_valid=None):
        self._init_model(x.shape[1], 1)
        x = torch.from_numpy(x).type(torch.FloatTensor) if isinstance(x, numpy.ndarray) else x
        cf_x = torch.from_numpy(cf_x).type(torch.FloatTensor) if isinstance(cf_x, numpy.ndarray) else cf_x
        y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1) if isinstance(y, numpy.ndarray) else y
        x, cf_x, y = x.to(self.device), cf_x.to(self.device), y.to(self.device)

        yloss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        cf_loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        clp_loss_fn = nn.MSELoss(reduction='mean').to(self.device)

        self.model.to(self.device)
        train_loader = DataLoader(TensorDataset(x, cf_x, y), batch_size=10000, shuffle=True)

        if x_valid is not None:
            x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor).cuda() if isinstance(x_valid, numpy.ndarray) else x_valid
            y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor).cuda().view(-1, 1) if isinstance(y_valid, numpy.ndarray) else y_valid

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        self.model.train()
        for _ in t:
            acc, total_loss, N = 0, 0, 0
            for x, cf_x, y in train_loader:
                y_pred_factual_logit = self.model(x)
                y_pred_counter_logit = self.model(cf_x)

                pred_loss = yloss_fn(y_pred_factual_logit, y)
                cf_loss = cf_loss_fn(y_pred_counter_logit, y)
                clp_loss = clp_loss_fn(y_pred_factual_logit, y_pred_counter_logit)
                loss = pred_loss + self.parm_cf * cf_loss + self.parm_clp * clp_loss
                total_loss += loss

                y_pred_prob = torch.sigmoid(y_pred_factual_logit)
                acc += accuracy(y_pred_prob, y, task='binary')

                self.model.zero_grad()
                loss.backward()
                opt.step()
                N += 1

            if x_valid is not None:
                valid_acc = accuracy(torch.sigmoid(self.model(x_valid)), y_valid, task='binary')
                t.set_postfix({'CLF Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4),
                               'Valid Acc Src': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'CLF Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4)})

        self.model.eval()
        self.model_fitted = True


class NNRegressor(nn.Module):
    def __init__(self, lr=None, n_epochs=None, hidden_layer_sizes=(100,), batch_size=10000):
        super(NNRegressor, self).__init__()
        self.model = None
        self.h_layers = hidden_layer_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model_fitted = False

    def forward(self, x):
        return self.model(x)

    def fit(self, x, y, x_valid=None, y_valid=None):
        self._init_model(x.shape[1], 1)

        x = torch.from_numpy(x).type(torch.FloatTensor) if isinstance(x, numpy.ndarray) else x
        y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1) if isinstance(y, numpy.ndarray) else y
        x, y = x.to(self.device), y.to(self.device)

        loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        self.model.to(self.device)

        train_loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        if x_valid is not None:
            x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor).cuda() if isinstance(x_valid, numpy.ndarray) else x
            y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor).cuda().reshape(-1, 1) if isinstance(y_valid, numpy.ndarray) else y

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        self.model.train()

        for _ in t:
            acc, total_loss, N = 0, 0, 0
            for x, y in train_loader:
                y_pred = self.model(x)
                y_loss = loss_fn(y_pred, y)

                total_loss += y_loss

                acc += float(torch.sum((y_pred - y) ** 2))

                self.model.zero_grad()
                y_loss.backward()
                opt.step()
                N += y.shape[0]

            if x_valid is not None:
                valid_acc = torch.mean((self.model(x_valid) - y_valid) ** 2)
                t.set_postfix({'CLF Loss': float(total_loss), 'Train RMSE': round((float(acc) / N)**0.5, 4),
                               'Valid RMSE': round(float(valid_acc)**0.5, 4)})
            else:
                t.set_postfix({'CLF Loss': float(total_loss), 'Train RMSE': round((float(acc) / N)**0.5, 4)})

        self.model.eval()
        self.model_fitted = True

    def predict(self, x):
        if not self.model_fitted:
            raise NotFittedError("This MLP instance is not fitted yet. Call 'fit' with "
                                 "appropriate arguments before using this estimator.")
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device) if isinstance(x, numpy.ndarray) else x.to(self.device)
        self.model.to(self.device)
        with torch.no_grad():
            out = self.model(x)
            return out.cpu().numpy()

    def _init_weight(self):
        for _ in self.modules():
            if isinstance(_, nn.Linear) or isinstance(_, nn.Conv2d) or isinstance(_, nn.ConvTranspose2d):
                nn.init.xavier_normal_(_.weight.data)

    def _init_model(self, i_dim, o_dim):
        self.model = nn.Sequential(
            nn.Linear(i_dim, self.h_layers[0]),
            nn.ReLU(),
        )
        for i in range(len(self.h_layers) - 1):
            self.model.add_module(f"Hidden_{i+1}", nn.Linear(self.h_layers[i], self.h_layers[i+1]))
            self.model.add_module(f"ReLU_Hidden_{i+1}", nn.ReLU())
        self.model.add_module("Output", nn.Linear(self.h_layers[-1], o_dim))
        self._init_weight()


class NNRegressorWithFairReg(NNRegressor):
    def __init__(self, lr=None, n_epochs=None, hidden_layer_sizes=(100,), batch_size=10000, parm_cf=1.0, parm_clp=1.0):
        super(NNRegressorWithFairReg, self).__init__(lr, n_epochs, hidden_layer_sizes, batch_size)
        self.parm_cf = parm_cf
        self.parm_clp = parm_clp

    def fit(self, x, cf_x, y, x_valid=None, y_valid=None):
        self._init_model(x.shape[1], 1)

        x = torch.from_numpy(x).type(torch.FloatTensor) if isinstance(x, numpy.ndarray) else x
        cf_x = torch.from_numpy(cf_x).type(torch.FloatTensor) if isinstance(cf_x, numpy.ndarray) else cf_x
        y = torch.from_numpy(y).type(torch.FloatTensor).view(-1, 1) if isinstance(y, numpy.ndarray) else y
        x, cf_x, y = x.to(self.device), cf_x.to(self.device), y.to(self.device)

        yloss_fn = nn.MSELoss(reduction='mean').to(self.device)
        cf_loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        clp_loss_fn = nn.MSELoss(reduction='mean').to(self.device)

        self.model.to(self.device)

        train_loader = DataLoader(TensorDataset(x, cf_x, y), batch_size=self.batch_size, shuffle=True)
        if x_valid is not None:
            x_valid = torch.from_numpy(x_valid).type(torch.FloatTensor).cuda() if isinstance(x_valid, numpy.ndarray) else x
            y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor).cuda().reshape(-1, 1) if isinstance(y_valid, numpy.ndarray) else y

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        self.model.train()

        for _ in t:
            acc, total_loss, N = 0, 0, 0
            for x, cf_x, y in train_loader:
                y_pred_factual = self.model(x)
                y_pred_counter = self.model(cf_x)

                y_loss = yloss_fn(y_pred_factual, y)
                cf_loss = cf_loss_fn(y_pred_counter, y)
                clp_loss = clp_loss_fn(y_pred_factual, y_pred_counter)
                loss = y_loss + self.parm_cf * cf_loss + self.parm_clp * clp_loss

                total_loss += loss

                acc += float(torch.sum((y_pred_factual - y) ** 2))

                self.model.zero_grad()
                loss.backward()
                opt.step()
                N += y.shape[0]

            if x_valid is not None:
                valid_acc = torch.mean((self.model(x_valid) - y_valid) ** 2)
                t.set_postfix({'CLF Loss': float(total_loss), 'Train RMSE': round((float(acc) / N) ** 0.5, 4),
                               'Valid RMSE': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'CLF Loss': float(total_loss), 'Train RMSE': round((float(acc) / N) ** 0.5, 4)})

        self.model.eval()
        self.model_fitted = True


# class CFClassifier(nn.Module):
#     def __init__(self, i_dim, o_dim, hidden_layer_sizes=(100,)):
#         super(CFClassifier, self).__init__()
#         self.model = None
#         self.h_layers = hidden_layer_sizes
#         self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cuda')
#         self.model_fitted = False
#
#         self._init_model(i_dim, o_dim)
#
#     def forward(self, x):
#         return self.model(x)
#
#     def _init_weight(self):
#         for _ in self.modules():
#             if isinstance(_, nn.Linear) or isinstance(_, nn.Conv2d) or isinstance(_, nn.ConvTranspose2d):
#                 nn.init.xavier_normal_(_.weight.data)
#
#     def _init_model(self, i_dim, o_dim):
#         self.model = nn.Sequential(
#             nn.Linear(i_dim, self.h_layers[0]),
#             nn.ReLU(),
#         )
#         for i in range(len(self.h_layers) - 1):
#             self.model.add_module(f"Hidden_{i+1}", nn.Linear(self.h_layers[i], self.h_layers[i+1]))
#             self.model.add_module(f"ReLU_Hidden_{i+1}", nn.ReLU())
#         self.model.add_module("Output", nn.Linear(self.h_layers[-1], o_dim))
#         self.model.add_module("Sigmoid", nn.Sigmoid())
#         self._init_weight()

def calc_metrics_single(clf, p, task='classification'):
    una, cf_una, cf_una_dis, y, x, r_x, cf_r_x = p
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




