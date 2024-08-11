# coding = utf-8
import torch

from torch import nn
from CDVAE_model.components.encoder import Encoder
from CDVAE_model.components.decoder import DecoderX, DecoderY
from utils import ini_weight, filter_dummy_x


class CDVAE(nn.Module):
    def __init__(self, xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim, cfg):
        super(CDVAE, self).__init__()

        # dims
        self.dims = {'xnc': xnc_dim, 'xnb': xnb_dim, 'xac': xac_dim, 'xab': xab_dim, 'a': a_dim, 'y': y_dim}
        un_dim, ua_dim = cfg.un_dim, cfg.ua_dim

        # Activate function
        act_fn_dict = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(0.2, True), 'tahn': nn.Tanh()}
        self.act_fn = act_fn_dict[cfg.act_fn]

        # Initial Encoders and Decoders
        self.enc_xn = Encoder(xnc_dim + xnb_dim, cfg.h_dim, un_dim, cfg.h_layers, self.act_fn)
        self.enc_xa = Encoder(xac_dim + xab_dim + a_dim, cfg.h_dim, ua_dim, cfg.h_layers, self.act_fn)
        
        self.dec_uxn = DecoderX(un_dim, cfg.h_dim, xnc_dim, xnb_dim, cfg.h_layers, self.act_fn)
        self.dec_uxa = DecoderX(ua_dim + a_dim, cfg.h_dim, xac_dim, xab_dim, cfg.h_layers, self.act_fn)
        self.dec_uy = DecoderY(un_dim + ua_dim + a_dim, cfg.h_dim, y_dim, cfg.h_layers, self.act_fn)

        # Initial network weight
        ini_weight(self.modules())

    def forward(self, xn, xa, a):
        # encode
        un, ua, mu, logvar = self.encode(xn, xa, a)
        cf_un, cf_ua, cf_mu, cf_logvar = self.encode(xn, xa, 1 - a)
        # decode
        r_xnc, r_xnb, r_xac, r_xab, r_y = self.decode(un, ua, a)
        cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, cf_r_y = self.decode(cf_un, cf_ua, 1 - a)
        # disill
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, self.dims)
        cf_uxn_dis, cf_uxa_dis, _, _ = self.encode(cf_r_xn, cf_r_xa, 1 - a)

        factual = (un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y)
        counter = (cf_un, cf_ua, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, cf_r_y)
        disilled_cf = (cf_uxn_dis, cf_uxa_dis)

        return factual, counter, disilled_cf

    def encode(self, xn, xa, a):
        # Un
        mu_un, logvar_un = self.enc_xn(xn)
        un = self.reparameterize(mu_un, logvar_un)
        # Ua
        xa_a = torch.cat([xa, a], 1)
        mu_ua, logvar_ua = self.enc_xa(xa_a)
        ua = self.reparameterize(mu_ua, logvar_ua)
        # U
        mu = torch.cat([mu_un, mu_ua], 1)
        logvar = torch.cat([logvar_un, logvar_ua], 1)

        return un, ua, mu, logvar

    def decode(self, un, ua, a):
        # Reconstruct Xn
        r_xnc, rxn_b = self.dec_uxn(un)
        # Reconstruct Xa
        ua_a = torch.cat([ua, a], dim=1)
        r_xac, r_xab = self.dec_uxa(ua_a)
        # Reconstruct Y
        U_a = torch.cat([un, ua, a], dim=1)
        r_y = self.dec_uy(U_a)

        return r_xnc, rxn_b, r_xac, r_xab, r_y

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn_like(std)
        z = epsilon.mul(std).add_(mu)
        return z
