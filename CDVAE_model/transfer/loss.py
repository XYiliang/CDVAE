# coding = utf-8
import torch
from torch import nn, optim
from torch.nn import functional as F
from CDVAE_model.components.discriminator import DiscriminatorTC
from utils import filter_dummy_x


class CDVAELoss:
    def __init__(self, vae, device, dim_dict, cfg):
        self.vae = vae
        self.device = device
        self.n_epochs = cfg.vae_epochs
        self.dims = dim_dict
        self.rec_arg = {k: 1 if v != 0 else 0 for k, v in self.dims.items()}

        una_dim = cfg.un_dim + cfg.ua_dim
        a_dim = dim_dict['a']
        self.disc_tc = DiscriminatorTC(una_dim + a_dim, cfg.disc_h_dim, cfg.disc_h_layers).to(self.device)

        # Hyperparameters
        self.parm_r_xn, self.parm_r_xa, self.parm_r_y = cfg.parm_r_xn, cfg.parm_r_xa, cfg.parm_r_y
        self.parm_kl, self.parm_tc, self.parm_css, self.parm_tf = cfg.parm_kl, cfg.parm_tc, cfg.parm_css, cfg.parm_tf

        # Loss function
        self.loss_fn_cont = nn.MSELoss(reduction='mean').to(device)
        self.loss_fn_bin = nn.BCEWithLogitsLoss(reduction='mean').to(device)
        self.consistency = nn.L1Loss(reduction='mean').to(device)
        self.domain_loss = nn.NLLLoss(reduction='mean').to(device)

        # Optimizer
        self.vae_opt = optim.Adam(self.vae.parameters(), lr=cfg.vae_lr)
        self.disc_tc_opt = optim.Adam(self.disc_tc.parameters(), lr=cfg.disc_lr)

    def optimize(self, data_s, data_t, alpha, epoch):
        # Loss on source
        recon_loss_s, y_loss_s, KLD_s, tc_loss_s, css_loss_s, domain_loss_s, d_tc_loss_s\
            = self.calc_src_loss(data_s, alpha)
        # Loss on target
        recon_loss_t, _, KLD_t, tc_loss_t, css_loss_t, domain_loss_t, d_tc_loss_t\
            = self.calc_trg_loss(data_t, alpha)

        recon_loss = recon_loss_s + recon_loss_t
        y_loss = y_loss_s
        css_loss = css_loss_s + css_loss_t
        domain_loss = domain_loss_s + domain_loss_t
        # Considering total correlation loss and KLD on target domain harm the perfromance of CDVAE, so we discarded it.
        KLD = KLD_s
        tc_loss = tc_loss_s
        d_tc_loss = d_tc_loss_s

        VAE_loss = recon_loss + y_loss + self.parm_kl * KLD + self.parm_tc * tc_loss + self.parm_css * css_loss \
            + self.parm_tf * domain_loss

        self.vae_opt.zero_grad()
        VAE_loss.backward(retain_graph=True)

        self.disc_tc_opt.zero_grad()
        d_tc_loss.backward()

        self.vae_opt.step()
        self.disc_tc_opt.step()

        if epoch % 10 == 0:
            assert (torch.sum(torch.isnan(recon_loss)) == 0), 'recon_loss is NaN'
            assert (torch.sum(torch.isnan(y_loss)) == 0), 'y_loss is NaN'
            assert (torch.sum(torch.isnan(KLD)) == 0), 'KLD is NaN'
            assert (torch.sum(torch.isnan(tc_loss)) == 0), 'tc_loss is NaN'
            assert (torch.sum(torch.isnan(domain_loss)) == 0), 'domain_loss is NaN'
            assert (torch.sum(torch.isnan(d_tc_loss)) == 0), 'disc_tc_loss is NaN'

        return recon_loss, y_loss, KLD, tc_loss, css_loss, domain_loss, VAE_loss, d_tc_loss

    def calc_src_loss(self, data_s, alpha):
        xnc, xnb, xac, xab, a, y, domain_tag = data_s
        xnc = xnc.to(self.device)
        xnb = xnb.to(self.device)
        xac = xac.to(self.device)
        xab = xab.to(self.device)
        a = a.to(self.device)
        y = y.to(self.device)
        domain_tag = domain_tag.to(self.device)

        xn, xa = filter_dummy_x(xnc, xnb, xac, xab, self.dims)
        shuf_idx = torch.randperm(a.shape[0])
        xn2, xa2, a2 = xn[shuf_idx], xa[shuf_idx], a[shuf_idx]

        factual, _, cf_dis, domain_pred = self.vae(xn, xa, a, alpha)
        un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_U_dis, cf_ua_dis = cf_dis
        logvar = self.diagonal(logvar)

        # Reconstruction Loss
        recon_loss = self.calc_recon_loss(xnc, xnb, xac, xab, r_xnc, r_xnb, r_xac, r_xab)
        # label predict loss
        y_loss = self.parm_r_y * self.loss_fn_bin(r_y, y)
        # KL Divergence
        KLD = - 0.5 * (1 + logvar - logvar.exp() - mu.pow(2)).mean(dim=0).mean()
        # Total Correlation
        U_a = torch.cat([un, ua, a], 1)
        D_U_a = self.disc_tc(U_a)
        tc_loss = torch.abs(D_U_a[:, 0] - D_U_a[:, 1]).mean()
        # Consistency
        U = torch.cat([un, ua], 1)
        cf_U_dis = torch.cat([cf_U_dis, cf_ua_dis], 1)
        css_loss = self.consistency(U, cf_U_dis)
        # Domain Error
        domain_loss = self.domain_loss(domain_pred, domain_tag)
        # disc_tc_loss
        disc_tc_loss = self.calc_disc_tc_loss(xn2, xa2, a2, D_U_a, alpha)

        del xnc, xnb, xac, xab, a, y, domain_tag, xn, xa, factual,\
            domain_pred, logvar, xn2, xa2, a2, U_a, D_U_a

        return recon_loss, y_loss, KLD, tc_loss, css_loss, domain_loss, disc_tc_loss

    def calc_trg_loss(self, data_t, alpha):
        xnc, xnb, xac, xab, a, y, domain_tag = data_t
        xnc = xnc.to(self.device)
        xnb = xnb.to(self.device)
        xac = xac.to(self.device)
        xab = xab.to(self.device)
        a = a.to(self.device)
        domain_tag = domain_tag.to(self.device)
        xn, xa = filter_dummy_x(xnc, xnb, xac, xab, self.dims)

        # 目标域数据
        shuf_idx = torch.randperm(a.shape[0])
        xn2, xa2, a2 = xn[shuf_idx], xa[shuf_idx], a[shuf_idx]

        factual, counter, cf_dis, domain_pred = self.vae(xn, xa, a, alpha)
        un, ua, mu, logvar, r_xnc, r_xnb, r_xac, r_xab, r_y = factual
        cf_un_dis, cf_ua_dis = cf_dis
        logvar = self.diagonal(logvar)

        # Reconstruction Loss
        recon_loss = self.calc_recon_loss(xnc, xnb, xac, xab, r_xnc, r_xnb, r_xac, r_xab)
        # label predict loss
        y_loss = 0.
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
        # Domain Error
        domain_loss = self.domain_loss(domain_pred, domain_tag)
        # disc_tc_loss
        disc_tc_loss = self.calc_disc_tc_loss(xn2, xa2, a2, D_U_a, alpha)

        # Flush Cache
        del xnc, xnb, xac, xab, a, y, domain_tag, xn, xa, factual, counter, domain_pred,\
            logvar, xn2, xa2, a2, U_a, D_U_a

        return recon_loss, y_loss, KLD, tc_loss, css_loss, domain_loss, disc_tc_loss

    def calc_disc_tc_loss(self, xn2, xa2, a2, D_ua, alpha):
        """Optimize Discriminator_tc"""
        (un2, ua2, mu2, _, _, _, _, _, _), _, _, _ = self.vae(xn2, xa2, a2, alpha)

        una_a2 = torch.cat([un2, ua2, a2], 1)
        una_a2_perm = self._permute_dims(una_a2).detach()
        D_ua2_perm = self.disc_tc(una_a2_perm)
        ones = torch.ones(una_a2.shape[0], dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones, dtype=torch.long, device=self.device)
        disc_tc_loss = 0.5 * (F.cross_entropy(D_ua, zeros) + F.cross_entropy(D_ua2_perm, ones))

        return disc_tc_loss

    def calc_recon_loss(self, xnc, xnb, xac, xab, r_xnc, r_xnb, r_xac, r_xab):
        xnc_l = self.loss_fn_cont(r_xnc, xnc) * self.rec_arg['xnc']
        xnb_l = self.loss_fn_bin(r_xnb, xnb) * self.rec_arg['xnb']
        xac_l = self.loss_fn_cont(r_xac, xac) * self.rec_arg['xac']
        xab_l = self.loss_fn_bin(r_xab, xab) * self.rec_arg['xab']

        return self.parm_r_xn * (xnc_l + xnb_l) + self.parm_r_xa * (xac_l + xab_l)

    def _permute_dims(self, z):
        z_perm = torch.zeros_like(z)
        batch_size, dim_z = z_perm.size()
        for _ in range(dim_z):
            pi = torch.randperm(batch_size).to(self.device)
            z_perm[:, _] = z[pi, _]
        return z_perm

    def diagonal(self, M):
        return torch.where(torch.abs(M) < 1e-05, M + 1e-05 * torch.abs(M), M)
