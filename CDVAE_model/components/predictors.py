# coding = utf-8
import numpy
import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm
from utils import ini_weight, numpy2tensor, sync_device


class NNClassifier(nn.Module):
    def __init__(self, lr, n_epochs, hidden_layer_sizes=(100,), early_stop_threshold=None):
        super(NNClassifier, self).__init__()
        self.model = None
        self.h_layers = hidden_layer_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cuda')
        self.early_stop_val = early_stop_threshold
        self.do_valid = False

        ini_weight(self.modules())

    def forward(self, x):
        return self.model(x)

    @numpy2tensor
    def fit(self, x, y, x_valid=None, y_valid=None):
        if x_valid is not None and y_valid is not None:
            self.do_valid = True  
        
        self._init_model(x.shape[1], 1)
        self.model.to(self.device)
        
        x, y = x.type(torch.FloatTensor).to(self.device), y.type(torch.FloatTensor).view(-1, 1).to(self.device)
        if self.do_valid:
            x_valid = x_valid.type(torch.FloatTensor).to(self.device)
            y_valid = y_valid.type(torch.FloatTensor).view(-1, 1).to(self.device)
            
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=10000, shuffle=True)

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        for _ in t:
            acc, total_loss, N = 0, 0, 0
            self.model.train()
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

            if self.do_valid:
                self.model.eval()
                with torch.no_grad():
                    valid_acc = accuracy(torch.sigmoid(self.model(x_valid)), y_valid, task='binary')
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4),
                               'Valid Acc Src': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4)})
                
        self.model.eval()

    @numpy2tensor
    def fit_transfer(self, x_src, y_src, x_src_valid=None, y_src_valid=None, x_trg_valid=None, y_trg_valid=None):
        if x_src_valid is not None and y_src_valid is not None and x_trg_valid is not None and y_trg_valid is not None:
            self.do_valid = True  
        
        self._init_model(x_src.shape[1], 1)
        self.model.to(self.device)
        
        x_src, y_src = x_src.type(torch.FloatTensor).to(self.device), y_src.type(torch.FloatTensor).view(-1, 1).to(self.device)
        if self.do_valid:
            x_src_valid = x_src_valid.to(self.device)
            y_src_valid = y_src_valid.view(-1, 1).to(self.device)
            x_trg_valid = x_trg_valid.to(self.device)
            y_trg_valid = y_trg_valid.view(-1, 1).to(self.device)
            
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        train_loader = DataLoader(TensorDataset(x_src, y_src), batch_size=10000, shuffle=True)

        opt = optim.Adam(params=self.model.parameters(), lr=self.lr)
        t = tqdm(range(self.n_epochs), leave=True, position=0, dynamic_ncols=True, unit='epoch')

        for _ in t:
            acc, total_loss, N = 0, 0, 0
            self.model.train()
            for x_src, y_src in train_loader:
                y_pred_logit = self.model(x_src)

                loss = loss_fn(y_pred_logit, y_src)
                total_loss += loss

                y_pred_prob = torch.sigmoid(y_pred_logit)
                acc += accuracy(y_pred_prob, y_src, task='binary')

                self.model.zero_grad()
                loss.backward()
                opt.step()
                N += 1

            if self.do_valid:
                self.model.eval()
                with torch.no_grad():
                    valid_acc_src = accuracy(torch.sigmoid(self.model(x_src_valid)), y_src_valid, task='binary')
                    valid_acc_trg = accuracy(torch.sigmoid(self.model(x_trg_valid)), y_trg_valid, task='binary')
                t.set_postfix({'Predictor Epoch Loss': float(total_loss),
                            'Train Acc': round(float(acc) / N, 4),
                            'Valid Acc Src': round(float(valid_acc_src), 4),
                            'Valid Acc Trg': round(float(valid_acc_trg), 4)})
                self.model = self.model.to(self.device)
            else:
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4)})
                
        self.model.eval()
                
    @numpy2tensor
    def predict(self, x):
        cur_device = next(self.model.parameters()).device
        x = sync_device(cur_device, x)
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
            return torch.sigmoid(out).gt(0.5).byte().cpu().numpy()
        
    @numpy2tensor
    def predict_proba(self, x):
        cur_device = next(self.model.parameters()).device
        x = sync_device(cur_device, x)
        self.model.eval()
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

    @numpy2tensor
    def fit(self, x, cf_x, y, x_valid=None, y_valid=None):
        if x_valid is not None and y_valid is not None:
            self.do_valid = True 
            
        self._init_model(x.shape[1], 1)
        
        x = x.type(torch.FloatTensor).to(self.device)
        cf_x = cf_x.type(torch.FloatTensor).to(self.device)
        y = y.type(torch.FloatTensor).view(-1, 1).to(self.device)
        if x_valid is not None:
            x_valid = x_valid.type(torch.FloatTensor).to(self.device) 
            y_valid = y_valid.type(torch.FloatTensor).to(self.device).view(-1, 1)

        yloss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        cf_loss_fn = nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
        clp_loss_fn = nn.MSELoss(reduction='mean').to(self.device)

        self.model.to(self.device)
        train_loader = DataLoader(TensorDataset(x, cf_x, y), batch_size=10000, shuffle=True)
        
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

            if self.do_valid:
                valid_acc = accuracy(torch.sigmoid(self.model(x_valid)), y_valid, task='binary')
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4),
                               'Valid Acc Src': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train Acc': round(float(acc) / N, 4)})

        self.model.eval()


class NNRegressor(nn.Module):
    def __init__(self, lr=None, n_epochs=None, hidden_layer_sizes=(100,), batch_size=10000):
        super(NNRegressor, self).__init__()
        self.model = None
        self.h_layers = hidden_layer_sizes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.do_valid = False

    def forward(self, x):
        return self.model(x)

    @numpy2tensor
    def fit(self, x, y, x_valid=None, y_valid=None):
        if x_valid is not None and y_valid is not None:
            self.do_valid = True  
        
        self._init_model(x.shape[1], 1)
        self.model.to(self.device)
        
        x, y = x.type(torch.FloatTensor).to(self.device), y.type(torch.FloatTensor).view(-1, 1).to(self.device)
        if self.do_valid:
            x_valid = x_valid.type(torch.FloatTensor).to(self.device)
            y_valid = y_valid.type(torch.FloatTensor).view(-1, 1).to(self.device)

        loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        self.model.to(self.device)

        train_loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)

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
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train RMSE': round((float(acc) / N)**0.5, 4),
                               'Valid RMSE': round(float(valid_acc)**0.5, 4)})
            else:
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train RMSE': round((float(acc) / N)**0.5, 4)})

        self.model.eval()

    @numpy2tensor
    def predict(self, x):
        cur_device = next(self.model.parameters()).device
        x = sync_device(cur_device, x)
        x = x.type(torch.FloatTensor).to(self.device)
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

    @numpy2tensor
    def fit(self, x, cf_x, y, x_valid=None, y_valid=None):
        if x_valid is not None and y_valid is not None:
            self.do_valid = True  
            
        self._init_model(x.shape[1], 1)

        x = x.type(torch.FloatTensor).to(self.device)
        cf_x = cf_x.type(torch.FloatTensor).to(self.device)
        y = y.type(torch.FloatTensor).view(-1, 1).to(self.device)
        if x_valid is not None:
            x_valid = x_valid.type(torch.FloatTensor).to(self.device) 
            y_valid = y_valid.type(torch.FloatTensor).to(self.device).view(-1, 1)

        yloss_fn = nn.MSELoss(reduction='mean').to(self.device)
        cf_loss_fn = nn.MSELoss(reduction='mean').to(self.device)
        clp_loss_fn = nn.MSELoss(reduction='mean').to(self.device)

        self.model.to(self.device)

        train_loader = DataLoader(TensorDataset(x, cf_x, y), batch_size=self.batch_size, shuffle=True)

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
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train RMSE': round((float(acc) / N) ** 0.5, 4),
                               'Valid RMSE': round(float(valid_acc), 4)})
            else:
                t.set_postfix({'Predictor Epoch Loss': float(total_loss), 'Train RMSE': round((float(acc) / N) ** 0.5, 4)})

        self.model.eval()
        self.model_fitted = True




