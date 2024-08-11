# coding = utf-8
import torch

from torch import nn, optim
from torch.nn import functional as F
from CDVAE_model.components.discriminator import DiscriminatorTC
from utils import filter_dummy_x
from torchmetrics.functional import accuracy

class CDVAELoss:
    def __init__(self, vae, device, dim_dict, cfg):
        self.vae = vae
        self.device = device
        self.vae_epochs = cfg.vae_epochs
        self.dims = dim_dict
        self.disc_tc = DiscriminatorTC(
            z_dim=cfg.un_dim + cfg.ua_dim + dim_dict['a'],
            h_dim=cfg.disc_h_dim,
            h_layers=cfg.disc_h_layers
            ).to(self.device)
        self.cfg = cfg

        # loss weights
        self.parm_r_xn, self.parm_r_xa, self.parm_r_y = cfg.parm_r_xn, cfg.parm_r_xa, cfg.parm_r_y
        self.parm_kl, self.parm_tc, self.parm_css = cfg.parm_kl, cfg.parm_tc, cfg.parm_css

        # loss function
        self.loss_fn_cont = nn.MSELoss(reduction='mean').to(device)
        self.loss_fn_bin = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        self.consistency = nn.L1Loss(reduction='mean').to(device)

        # optimizers
        self.vae_opt = optim.Adam(self.vae.parameters(), lr=cfg.vae_lr)
        self.disc_tc_opt = optim.Adam(self.disc_tc.parameters(), lr=cfg.disc_lr)
        
        # This dict is for removing loss term of dummy tensors
        self.rec_arg = {k: 1 if v != 0 else 0 for k, v in self.dims.items()}   

    def optimize(self, data):
        xnc, xnb, xac, xab, a, y = data
        xnc = xnc.to(self.device)
        xnb = xnb.to(self.device)
        xac = xac.to(self.device)
        xab = xab.to(self.device)
        a = a.to(self.device)
        y = y.to(self.device)

        xn, xa = filter_dummy_x(xnc, xnb, xac, xab, self.dims)
        shuf_idx = torch.randperm(a.shape[0])
        xn2, xa2, a2 = xn[shuf_idx], xa[shuf_idx], a[shuf_idx]

        # Forward
        factual, _, cf_dis = self.vae(xn, xa, a)
        un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_un_dis, cf_ua_dis = cf_dis
        logvar = self.diagonal(logvar)

        # VAE Optimize
        # Reconstruction Loss
        xnc_loss = self.loss_fn_cont(r_xnc, xnc) * self.rec_arg['xnc']
        xnb_loss = self.loss_fn_bin(r_xnb, xnb) * self.rec_arg['xnb']
        xac_loss = self.loss_fn_cont(r_xac, xac) * self.rec_arg['xac']
        xab_loss = self.loss_fn_bin(r_xab, xab) * self.rec_arg['xab']
        xn_loss = xnc_loss + xnb_loss
        xa_loss = xac_loss + xab_loss

        if self.cfg.task == 'regression':
            y_loss = self.loss_fn_cont(r_y, y)
        else:
            y_loss = self.loss_fn_bin(r_y, y)

        recon_loss = self.parm_r_xn * xn_loss + self.parm_r_xa * xa_loss + self.parm_r_y * y_loss

        # KL Divergence
        KLD = - 0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).mean(dim=0).mean()

        # Total Correlation
        U_a = torch.cat([un, ua, a], 1)
        D_U_a = self.disc_tc(U_a)
        tc_loss = torch.abs(D_U_a[:, 0] - D_U_a[:, 1]).mean()

        # Consistency
        U = torch.cat([un, ua], 1)
        cf_U_dis = torch.cat([cf_un_dis, cf_ua_dis], 1)
        css_loss = self.consistency(U, cf_U_dis)

        # VAE Loss
        VAE_loss = recon_loss + self.parm_kl * KLD + self.parm_tc * tc_loss + self.parm_css * css_loss
        self.vae_opt.zero_grad()
        VAE_loss.backward(retain_graph=True)

        # Reduce memory usage
        del xnc_loss, xnb_loss, xac_loss, xab_loss, xn_loss, xa_loss, y_loss, xn, xa, a, y, xnc, xnb, xac, xab

        # Optimize Discriminator_tc
        un2, ua2, _, _ = self.vae.encode(xn2, xa2, a2)

        U_a2 = torch.cat([un2, ua2, a2], 1)
        U_a2_perm = self._permute_dims(U_a2).detach()
        D_U_a2_perm = self.disc_tc(U_a2_perm)
        ones = torch.ones(ua.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones, dtype=torch.long, device=self.device)
        disc_tc_loss = 0.5 * (F.cross_entropy(D_U_a, zeros) + F.cross_entropy(D_U_a2_perm, ones))

        del un2, ua2, U_a2, U_a2_perm, D_U_a, D_U_a2_perm, ones, zeros

        self.disc_tc_opt.zero_grad()
        disc_tc_loss.backward()

        self.vae_opt.step()
        self.disc_tc_opt.step()

        assert (torch.sum(torch.isnan(recon_loss)) == 0), 'recon_loss is NaN'
        assert (torch.sum(torch.isnan(KLD)) == 0), 'KLD is NaN'
        assert (torch.sum(torch.isnan(tc_loss)) == 0), 'tc_loss is NaN'
        assert (torch.sum(torch.isnan(disc_tc_loss)) == 0), 'disc_tc_loss is NaN'

        return recon_loss, KLD, tc_loss, css_loss, VAE_loss, disc_tc_loss

    def _permute_dims(self, u):
        u_perm = torch.zeros_like(u)
        batch_size, dim_u = u_perm.size()
        for _ in range(dim_u):
            pi = torch.randperm(batch_size).to(self.device)
            u_perm[:, _] = u[pi, _]
        return u_perm

    def diagonal(self, M):
        return torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
