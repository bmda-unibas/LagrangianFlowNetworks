import torch
import numpy as np

from experiments.bird_migration.models.util import device, build_scale_to_domain_transform
from experiments.bird_migration.models.template import DensityVelocityInterface

from experiments.gaussians.util.divergence_free.div_free import build_divfree_vector_field
from functorch import vmap
from experiments.gaussians.util.divergence_free.model import NeuralConservationLaw

def build_DFNN(num_layers, hidden_features, n_mixtures, **kwargs):
    return DivFreeNetwork(space_dim=3, num_layers=num_layers, hidden_features=hidden_features, n_mixtures=n_mixtures)


class DivFreeNetwork(torch.nn.Module, DensityVelocityInterface):
    def __init__(self, space_dim,
                 hidden_features,
                 num_layers,
                 n_mixtures=64):
        super().__init__()
        self.space_dim = space_dim
        self.in_features = self.out_features = space_dim + 1
        self._t_idx = 0
        self._x_idx = [i for i in range(self.in_features) if i != self._t_idx]

        self.network = NeuralConservationLaw(self.in_features, d_model=hidden_features, num_hidden_layers=num_layers,
                                             n_mixtures=n_mixtures).to(device)
        self.u_fn, self.params, self.A_fn = build_divfree_vector_field(self.network)
        self.u_fn_vmapped = vmap(self.u_fn, in_dims=(None, 0))
        self._e = None

    def parameters(self, recurse: bool = True):
        return self.params

    def get_antisymmetric_matrix(self, tx_samples):
        return vmap(self.A_fn, in_dims=(None, 0))(self.params, tx_samples)

    def get_divergence_free_vector(self, x):
        return self.u_fn_vmapped(self.params, x)

    def forward(self, _x, mods=None):
        x = _x
        div_free_vec = self.get_divergence_free_vector(x)
        rho = div_free_vec[..., 0].unsqueeze(-1) * 10

        flux = div_free_vec[..., 1:]
        return torch.concatenate([rho, flux], -1)  # * logabsdet.exp().view(-1, 1)
        # return self.network.forward(_x, mods)

    def density(self, x, t):
        return self.forward(torch.concatenate([t, x], -1))[..., [0]]

    def log_density(self, x, t):
        return self._log_density(torch.concatenate([t, x], -1))

    def _log_density(self, inputs):
        return (self.forward(inputs)[..., [0]]).log()

    def density_and_flux(self, x, t):
        return self._density_and_flux(torch.concatenate([t, x], -1))

    def _density_and_flux(self, inputs):
        return self.forward(inputs)

    def velocity(self, x, t):
        return self._velocity(torch.concatenate([t, x], -1))

    def _velocity(self, inputs):
        return self.forward(inputs)[..., 1:] / (self.forward(inputs)[..., [0]] + 1e-6)

    def flux(self, x, t):
        return self._flux(torch.concatenate([t, x], -1))

    def _flux(self, inputs):
        return self.forward(inputs)[..., 1:]
