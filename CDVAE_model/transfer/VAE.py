# coding = utf-8
import torch
from torch import nn
from CDVAE_model.components.encoder import Encoder
from CDVAE_model.components.decoder import DecoderX, DecoderY
from CDVAE_model.components.discriminator import DiscriminatorDomain
from utils import ini_weight, get_dims, filter_dummy_x
from torch.autograd import Function


class CDVAE(nn.Module):
    def __init__(self, dim_dict, cfg):
        super(CDVAE, self).__init__()

        self.dims = dim_dict
        xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim = get_dims(dim_dict)
        un_dim, ua_dim, uy_dim = cfg.un_dim, cfg.ua_dim, cfg.uy_dim

        # Get Encoder and Decoder
        act_fn_dict = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(0.2, True), 'tahn': nn.Tanh()}
        self.act_fn = act_fn_dict[cfg.act_fn]

        self.enc_un = Encoder(xnc_dim + xnb_dim, cfg.h_dim, un_dim, cfg.h_layers, self.act_fn)
        self.enc_ua = Encoder(xac_dim + xab_dim + a_dim, cfg.h_dim, ua_dim, cfg.h_layers, self.act_fn)

        self.dec_xn = DecoderX(un_dim, cfg.h_dim, xnc_dim, xnb_dim, cfg.h_layers, self.act_fn)
        self.dec_xa = DecoderX(ua_dim + a_dim, cfg.h_dim, xac_dim, xab_dim, cfg.h_layers, self.act_fn)
        self.dec_y = DecoderY(un_dim + ua_dim + a_dim, cfg.h_dim, y_dim, cfg.h_layers, self.act_fn)

        self.domain_disc = DiscriminatorDomain(un_dim + ua_dim, cfg.disc_h_dim)

        # Initial Weight
        ini_weight(self.modules())

    def forward(self, xn, xa, a, alpha=0):
        un, ua, mu, logvar = self.encode(xn, xa, a)
        cf_un, cf_ua, cf_mu, cf_logvar = self.encode(xn, xa, 1 - a)

        # Gradient Reversal Layer (GRL)
        U = torch.cat([un, ua], 1)
        grad_reversed_U = ReverseLayerF.apply(U, alpha)
        domain_pred = self.domain_disc(grad_reversed_U)

        r_xnc, r_xnb, r_xac, r_xab, r_y = self.decode(un, ua, a)
        cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, cf_r_y = self.decode(cf_un, cf_ua, 1 - a)
        cf_r_xn, cf_r_xa = filter_dummy_x(cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab, self.dims)
        cf_un_dis, cf_ua_dis, _, _ = self.encode(cf_r_xn, cf_r_xa, 1-a)

        factual = (un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y)
        counter = (cf_un, cf_ua, cf_mu, cf_logvar, cf_r_xnc, cf_r_xnb, cf_r_xac, cf_r_xab)
        cf_dis = (cf_un_dis, cf_ua_dis)

        return factual, counter, cf_dis, domain_pred

    def encode(self, xn, xa, a):
        # un
        mu_un, logvar_un = self.enc_un(xn)
        un = self.reparameterize(mu_un, logvar_un)
        # ua
        xa_a = torch.cat([xa, a], 1)
        mu_ua, logvar_ua = self.enc_ua(xa_a)
        ua = self.reparameterize(mu_ua, logvar_ua)
        # U
        mu = torch.cat([mu_un, mu_ua], 1)
        logvar = torch.cat([logvar_un, logvar_ua], 1)

        return un, ua, mu, logvar

    def decode(self, un, ua, a):
        # Reconstruct Xn
        r_xnc, r_xnb = self.dec_xn(un)
        # Reconstruct Xa
        ua_a = torch.cat([ua, a], 1)
        r_xac, r_xab = self.dec_xa(ua_a)
        # Reconstruct Y
        U_a = torch.cat([un, ua, a], 1)
        r_y = self.dec_y(U_a)
        
        return r_xnc, r_xnb, r_xac, r_xab, r_y

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        epsilon = torch.randn_like(std)
        
        return epsilon.mul(std).add_(mu)


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
