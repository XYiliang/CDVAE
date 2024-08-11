# coding = utf-8
from torch import nn


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, u_dim, h_layers, act_fn):
        super(Encoder, self).__init__()
        self.act_fn = act_fn
        self.in_layer = nn.Linear(x_dim, h_dim)
        self.h_layer = nn.ModuleList()
        for _ in range(h_layers):
            self.h_layer.append(nn.Linear(h_dim, h_dim))
            self.h_layer.append(self.act_fn)
        self.mu_gen = nn.Linear(h_dim, u_dim)
        self.logvar_gen = nn.Linear(h_dim, u_dim)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.act_fn(self.in_layer(x))
        for layer in self.h_layer:
            x = layer(x)
        mu = self.mu_gen(x)
        logvar = self.logvar_gen(x)

        return mu, logvar
