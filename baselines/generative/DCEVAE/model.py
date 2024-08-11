import torch
from torch import nn
import torch.distributions as dists
import torch.nn.functional as F
import math
import sys

import random

class DCEVAE(nn.Module):
    def __init__(self, rc_dim, rb_dim, dc_dim, db_dim, sens_dim, label_dim, args):
        super(DCEVAE, self).__init__()
        '''random seed'''
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        if args.device == 'cuda':
            print("Current CUDA random seed", torch.cuda.initial_seed())
        else:
            print("Current CPU random seed", torch.initial_seed())

        """model structure"""
        self.device = args.device
        self.args = args
        
        self.r_dim = 2
        self.d_dim = 3
        self.rc_dim = rc_dim
        self.dc_dim = dc_dim
        self.label_dim = label_dim
        self.sens_dim = sens_dim
        ur_dim = args.ur_dim
        ud_dim = args.ud_dim
        u_dim = ur_dim + ud_dim
        self.ur_dim = ur_dim
        self.ud_dim = ud_dim
        self.u_dim = u_dim
        if args.act_fn == 'ReLU':
            act_fn = nn.LeakyReLU()
        elif args.act_fn == 'Tanh':
            act_fn = nn.Tanh()
        h_dim = args.h_dim

        """encoder (x_r, a, y) -> ur"""
        i_dim = (self.r_dim + sens_dim + label_dim)
        self.encoder_i_to_ur = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn, nn.Linear(h_dim, h_dim), act_fn)
        self.mu_i_to_ur = nn.Sequential(nn.Linear(h_dim, ur_dim))
        self.logvar_i_to_ur = nn.Sequential(nn.Linear(h_dim, ur_dim))

        i_dim = (self.d_dim + sens_dim + label_dim)
        """encoder (x_d, a, y) -> ud"""
        self.encoder_i_to_ud = nn.Sequential(nn.Linear(i_dim, h_dim), act_fn, nn.Linear(h_dim, h_dim), act_fn)
        self.mu_i_to_ud = nn.Sequential(nn.Linear(h_dim, ud_dim))
        self.logvar_i_to_ud = nn.Sequential(nn.Linear(h_dim, ud_dim))

        """decoder"""
        self.decoder_ur_to_rc = nn.Sequential(nn.Linear(ur_dim, h_dim), act_fn, nn.Linear(h_dim, 1))
        self.decoder_ur_to_rb = nn.Sequential(nn.Linear(ur_dim, h_dim), act_fn, nn.Linear(h_dim, 1))
        self.decoder_uda_to_dc = nn.Sequential(nn.Linear(ud_dim + sens_dim, h_dim), act_fn, nn.Linear(h_dim, 2))
        self.decoder_uda_to_db = nn.Sequential(nn.Linear(ud_dim + sens_dim, h_dim), act_fn, nn.Linear(h_dim, 1))
        self.p_ua_to_y = nn.Sequential(nn.Linear(u_dim + sens_dim, h_dim), act_fn, nn.Linear(h_dim, label_dim))

        """Discriminator Network"""
        d_dim = u_dim + sens_dim
        self.discriminator = nn.Sequential(
            nn.Linear(d_dim, h_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(h_dim, 2)
        )

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    @staticmethod
    def reparameterize(mu, logvar):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        std = torch.exp(logvar.mul(0.5))
        eps = torch.randn_like(std).to(device)
        u = mu + eps.mul(std)
        return u

    def D(self, z):
        z = self.discriminator(z).reshape(-1, 2)
        
        return z

    def q_u(self, r, d, a, y):
        # q(z|r,d,y)
        i = torch.cat([r, a, y], 1)
        intermediate = self.encoder_i_to_ur(i)
        ur_mu = self.mu_i_to_ur(intermediate)
        ur_logvar = self.logvar_i_to_ur(intermediate)

        # q(z|r,d,y)
        i = torch.cat([d, a, y], 1)
        intermediate = self.encoder_i_to_ud(i)
        ud_mu = self.mu_i_to_ud(intermediate)
        ud_logvar = self.logvar_i_to_ud(intermediate)

        u_mu = torch.cat([ur_mu, ud_mu], 1)
        u_logvar = torch.cat([ur_logvar, ud_logvar], 1)

        return u_mu, u_logvar

    def p_i(self, ur, ud, a):
        """
        VARIATIONAL PRIOR
        :param z, s, s_cf: input
        :return: output of p(o,x,y|z,s) & p(o,x,y|z,s_cf), (MB, hid_dim)
        """
        ua = torch.cat([ur, ud, a], 1)
        y = self.p_ua_to_y(ua)

        ua_cf = torch.cat([ur, ud, 1-a], 1)
        y_cf = self.p_ua_to_y(ua_cf)

        rc_mu = self.decoder_ur_to_rc(ur)
        rb_mu = self.decoder_ur_to_rb(ur)
        r_mu = torch.cat([rc_mu, rb_mu], 1)

        uda = torch.cat([ud, a], 1)
        uda_cf = torch.cat([ud, 1-a], 1)
        dc_mu = self.decoder_uda_to_dc(uda)
        dc_mu_cf = self.decoder_uda_to_dc(uda_cf)
        db_mu = self.decoder_uda_to_db(uda)
        db_mu_cf = self.decoder_uda_to_db(uda_cf)
        d_mu = torch.cat([dc_mu, db_mu], 1)
        d_mu_cf = torch.cat([dc_mu_cf, db_mu_cf], 1)
        return r_mu, d_mu, y, d_mu_cf, y_cf

    def reconstruct(self, u, a):
        ur, ud = torch.split(u, [self.ur_dim, self.ud_dim], 1)
        r_p, d_p, y_p, d_p_cf, y_p_cf = self.p_i(ur, ud, a)

        return r_p, d_p, y_p, d_p_cf, y_p_cf

    def reconstruct_hard(self, u, a):
        r_p, d_p, y_p, d_p_cf, y_p_cf = self.reconstruct(u, a)
        rc, rb_p = r_p[:, :self.rc_dim], r_p[:, self.rc_dim:]
        dc, db_p = d_p[:, :self.dc_dim], d_p[:, self.dc_dim:]
        dc_cf, db_cf_p = d_p_cf[:, :self.dc_dim], d_p_cf[:, :self.dc_dim]
        
        rb = dists.bernoulli.Bernoulli(torch.sigmoid(rb_p)).sample()
        db, db_cf = dists.bernoulli.Bernoulli(torch.sigmoid(db_p)).sample(), dists.bernoulli.Bernoulli(torch.sigmoid(db_cf_p)).sample()
        
        y_p = nn.Sigmoid()(y_p)
        y_p_cf = nn.Sigmoid()(y_p_cf)
        y_hard = dists.bernoulli.Bernoulli(y_p)
        y_hard = y_hard.sample()
        y_cf_hard = dists.bernoulli.Bernoulli(y_p_cf)
        y_cf_hard = y_cf_hard.sample()
        
        r_hard = torch.cat([rc, rb], 1)
        d_hard = torch.cat([dc, db], 1)
        d_cf_hard = torch.cat([dc_cf, db_cf], 1)

        return r_hard, d_hard, d_cf_hard, y_hard, y_cf_hard

    def diagonal(self, M):
        """
        If diagonal value is close to 0, it could makes cholesky decomposition error.
        To prevent this, I add some diagonal value which won't affect the original value too much.
        """
        new_M = torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
        return new_M

    def permute_dims(self, u):
        assert u.dim() == 2

        B, _ = u.size()
        perm_u = []
        idx = 0
        for u_j in torch.split(u, [self.ur_dim, self.ud_dim, self.sens_dim], 1):
            perm = torch.randperm(B).to(self.device)
            if idx == 0:
                perm_r = perm
            elif idx == 2:
                perm = perm_r
            perm_u_j = u_j[perm]
            perm_u.append(perm_u_j)
            idx += 1

        return torch.cat(perm_u, 1)

    def calculate_loss(self, r, d, a, y, r2, d2, a2, y2):

        MB = self.args.batch_size

        u_mu, u_logvar = self.q_u(r, d, a, y)
        u = self.reparameterize(u_mu, u_logvar)
        ur, ud = torch.split(u, [self.ur_dim, self.ud_dim], 1)

        r_mu, d_mu, y_p, d_mu_cf, y_p_cf = self.p_i(ur, ud, a)

        "reconstruction"
        loss_fn_cont = nn.MSELoss(reduction='sum')
        loss_fn_bin = nn.BCEWithLogitsLoss(reduction='sum')

        d_recon = loss_fn_cont(d_mu[:, :self.dc_dim], d[:, :self.dc_dim]) / MB
        r_recon = loss_fn_cont(r_mu[:, :self.rc_dim], r[:, :self.rc_dim]) / MB + loss_fn_bin(r_mu[:, self.rc_dim:], r[:, self.rc_dim:]) / MB 
        y_recon = loss_fn_bin(y_p, y) / MB

        recon = self.args.a_r * r_recon + self.args.a_d * d_recon + self.args.a_y * y_recon

        """KL loss"""
        # Prohibiting cholesky error
        u_logvar = self.diagonal(u_logvar)

        assert (torch.sum(torch.isnan(u_logvar)) == 0), 'u_logvar'

        u_dist = dists.MultivariateNormal(u_mu.flatten(), torch.diag(u_logvar.flatten().exp()))
        u_prior = dists.MultivariateNormal(torch.zeros(self.u_dim * u_mu.size()[0]).to(self.device),\
                                           torch.eye(self.u_dim * u_mu.size()[0]).to(self.device))
        u_kl = dists.kl.kl_divergence(u_dist, u_prior)/MB

        """independent Loss"""
        ones = torch.ones(u.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros(u.shape[0], dtype=torch.long, device=self.device)

        ua = torch.cat([u, a], 1)
        D_u = self.D(ua)
        vae_tc_loss = torch.mean(D_u[:, 0] - D_u[:, 1])

        u2_mu, u2_logvar = self.q_u(r2, d2, a2, y2)
        u2 = self.reparameterize(u2_mu, u2_logvar)
        ua2 = torch.cat([u2, a2], 1)
        u_perm = self.permute_dims(ua2)
        D_u_perm = self.D(u_perm)
        D_tc_loss = 0.5 * (F.cross_entropy(D_u, zeros) + F.cross_entropy(D_u_perm, ones))

        """fair loss"""
        # y_cf_sig = nn.Sigmoid()(y_p_cf)
        y_cf_sig = torch.sigmoid(y_p_cf)
        # y_p_sig = nn.Sigmoid()(y_p)
        y_p_sig = torch.sigmoid(y_p)
        #fair_l = ((y_cf_sig.log() * y_p_sig + (1 - y_cf_sig).log() * (1 - y_p_sig)).mean()) * (-1)
        fair_l = torch.sum(torch.norm(y_cf_sig - y_p_sig, p=2, dim=1)) / MB

        assert (torch.sum(torch.isnan(recon)) == 0), 'x_recon'
        assert (torch.sum(torch.isnan(y_recon)) == 0), 'y_recon'
        assert (torch.sum(torch.isnan(u_kl)) == 0), 'u_kl'

        ELBO = recon + self.args.u_kl * u_kl + self.args.a_h * vae_tc_loss + self.args.a_f * fair_l
        # ELBO = recon + self.args.u_kl * u_kl + self.args.a_f * fair_l + self.args.a_h * vae_tc_loss
        assert (torch.sum(torch.isnan(ELBO)) == 0), 'ELBO'

        return ELBO, recon, y_recon, y_p, y_p_cf, u_kl, vae_tc_loss, D_tc_loss, fair_l

