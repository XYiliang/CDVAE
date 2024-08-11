# coding = utf-8
import torch
from torch import nn


class DiscriminatorTC(nn.Module):
    def __init__(self, z_dim, h_dim, h_layers=3, neg_slope=0.2):
        super(DiscriminatorTC, self).__init__()
        self.leaky_relu = nn.LeakyReLU(neg_slope, inplace=True)

        self.in_layer = nn.Linear(z_dim, h_dim)
        self.h_layer = nn.ModuleList()
        for i in range(h_layers):
            self.h_layer.append(nn.Linear(h_dim, h_dim))
            self.h_layer.append(self.leaky_relu)
        self.o_layer = nn.Linear(h_dim, 2)

    def forward(self, z):
        z = self.leaky_relu(self.in_layer(z))
        for layer in self.h_layer:
            z = layer(z)
        z = torch.softmax(self.o_layer(z), 1)
        return z


class DiscriminatorDomain(nn.Module):
    def __init__(self, z_dim, h_dim, h_layers=3, neg_slope=0.2):
        super(DiscriminatorDomain, self).__init__()
        self.leaky_relu = nn.LeakyReLU(neg_slope, inplace=True)

        self.in_layer = nn.Linear(z_dim, h_dim)
        self.h_layer = nn.ModuleList()
        for i in range(h_layers):
            self.h_layer.append(nn.Linear(h_dim, h_dim))
            self.h_layer.append(self.leaky_relu)
        self.o_layer = nn.Linear(h_dim, 2)

    def forward(self, z):
        z = self.leaky_relu(self.in_layer(z))
        for layer in self.h_layer:
            z = layer(z)
        z = torch.log_softmax(self.o_layer(z), dim=1)
        return z

