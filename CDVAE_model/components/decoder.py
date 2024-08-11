# coding = utf-8
import torch
from torch import nn


class DecoderX(nn.Module):
    def __init__(self, u_dim, h_dim, xc_dim, xb_dim, h_layers, act_fn):
        super(DecoderX, self).__init__()
        self.act_fn = act_fn
        self.xc_dim, self.xb_dim = xc_dim, xb_dim
        self.in_layer = nn.Linear(u_dim, h_dim)
        self.h_layer = nn.ModuleList()
        for _ in range(h_layers):
            self.h_layer.append(nn.Linear(h_dim, h_dim))
            self.h_layer.append(self.act_fn)
        # Output discrete and continuous separately.
        self.xc_rec = nn.Linear(h_dim, xc_dim) if self.xc_dim else nn.Linear(h_dim, 1)
        self.xb_rec = nn.Linear(h_dim, xb_dim) if self.xb_dim else nn.Linear(h_dim, 1)

    def forward(self, u):
        u = self.act_fn(self.in_layer(u))
        for layer in self.h_layer:
            u = layer(u)
        r_xc = self.xc_rec(u)
        r_xb = self.xb_rec(u)
        return r_xc, r_xb


class DecoderY(nn.Module):
    def __init__(self, u_dim, h_dim, y_dim, h_layers, act_fn):
        super(DecoderY, self).__init__()
        self.act_fn = act_fn
        self.in_layer = nn.Linear(u_dim, h_dim)
        self.h_layer = nn.ModuleList()
        for _ in range(h_layers):
            self.h_layer.append(nn.Linear(h_dim, h_dim))
            self.h_layer.append(self.act_fn)
        self.y_rec = nn.Linear(h_dim, y_dim)

    def forward(self, u):
        u = self.act_fn(self.in_layer(u))
        for layer in self.h_layer:
            u = layer(u)
        r_y = self.y_rec(u)

        return r_y
