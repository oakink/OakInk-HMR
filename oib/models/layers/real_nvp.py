import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn


def net_s(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, dim),
                         nn.Tanh())


def net_t(dim):
    return nn.Sequential(nn.Linear(dim, 64), nn.LeakyReLU(), nn.Linear(64, 64), nn.LeakyReLU(), nn.Linear(64, dim))


class RealNVP(nn.Module):

    def __init__(self, prior_dim):
        super(RealNVP, self).__init__()
        self.dim = prior_dim
        self.register_buffer("loc", torch.zeros(self.dim))
        self.register_buffer("cov", torch.eye(self.dim))

        mask_ = np.zeros(self.dim)
        mask_[:round(self.dim / 2)] = 1
        _mask = 1 - mask_
        mask = torch.from_numpy(np.concatenate([[mask_], [_mask]] * 3, axis=0).astype(np.float32))
        self.register_buffer('mask', mask)
        self.t = torch.nn.ModuleList([net_s(self.dim) for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([net_t(self.dim) for _ in range(len(mask))])

    def _init(self):
        for m in self.t:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)
        for m in self.s:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight, gain=0.01)

    def forward_p(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def backward_p(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = 1e-8 + (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self, x):
        prior = D.MultivariateNormal(self.loc, self.cov)
        z, logp = self.backward_p(x)

        z_is_nan = torch.isnan(z)
        if z_is_nan.sum() != 0:
            print("z has nan value")

        z = torch.nan_to_num(z)

        res = prior.log_prob(z) + logp
        res = torch.nan_to_num(res)
        return res

    def sample(self, batchSize):
        prior = D.MultivariateNormal(self.loc, self.cov)
        z = prior.sample((batchSize, 1))
        x = self.forward_p(z)
        return x

    def forward(self, x):
        return self.log_prob(x)
