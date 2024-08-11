# coding = utf-8
import torch
from torch.utils.data import Dataset


class MyDatasetY(Dataset):
    def __init__(self, xnc, xnb, xac, xab, a, y):
        super(MyDatasetY, self).__init__()
        self.xnc = xnc
        self.xnb = xnb
        self.xac = xac
        self.xab = xab
        self.a = a
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, item):
        xnc = self.xnc[item, :]
        xnb = self.xnb[item, :]
        xac = self.xac[item, :]
        xab = self.xab[item, :]
        a = self.a[item, :]
        y = self.y[item, :]
        data = xnc, xnb, xac, xab, a, y
        return data


class MyDatasetNoY(Dataset):
    def __init__(self, xnc, xnb, xac, xab, a):
        super(MyDatasetNoY, self).__init__()
        self.xnc = xnc
        self.xnb = xnb
        self.xac = xac
        self.xab = xab
        self.a = a
        # We will not use y as input
        self.y = torch.tensor([-1]*self.a.shape[0]).reshape(-1, 1).type(torch.FloatTensor)

    def __len__(self):
        return self.xnc.shape[0]

    def __getitem__(self, item):
        xnc = self.xnc[item, :]
        xnb = self.xnb[item, :]
        xac = self.xac[item, :]
        xab = self.xab[item, :]
        a = self.a[item, :]
        y = self.y[item, :]
        data = xnc, xnb, xac, xab, a, y
        return data
