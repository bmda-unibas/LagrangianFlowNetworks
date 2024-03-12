import numpy as np
import torch
import tqdm
from torch.nn.functional import softplus, mse_loss

from LFlow import SemiLagrangianFlow
from enflows.CNF import neural_odes
from enflows.utils.torchutils import np_to_tensor

from experiments.bird_migration.models.util import device, build_scale_to_domain_transform
from experiments.bird_migration.bird_utils import weighted_MSE
from experiments.bird_migration.models.template import DensityVelocityInterface
from experiments.bird_migration.models.MLP import VanillaNN


def build_SLDA(num_layers, hidden_features, activation, complete_dataset, init_log_scale_norm, **kwargs):
    odenet_flow = neural_odes.ODEnet(
        hidden_dims=tuple([hidden_features] * num_layers),
        input_shape=(3,),
        strides=None,
        conv=False,
        layer_type="concat_v2",
        nonlinearity=activation,
        act_norm=False,
        scale_output=1
    )
    return SemiLagrangianNODE(complete_dataset=complete_dataset, dim=3,
                              odenet_flow=odenet_flow, divergence_fn="approximate",
                              grid_len=20,
                              atol=1e-5, rtol=1e-5, solver="dopri5").to(device)


class SemiLagrangianNODE(SemiLagrangianFlow, DensityVelocityInterface):
    def __init__(self, complete_dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds = complete_dataset
        self.shift = torch.nn.Parameter(np_to_tensor(np.array([4., 2.92, 3.])).view(-1, 3), requires_grad=False)
        self.scale = torch.nn.Parameter(np_to_tensor(3 * np.array([1., 0.75, 3.5])).view(-1, 3), requires_grad=False)


        tmp = torch.ones(1, 2)
        tmp[:, 1] = 1.3
        self.uv_balance = torch.nn.Parameter(tmp, requires_grad=False)

    def transform_to_basegrid_range(self, x):
        return (x - self.shift) / self.scale

    def inv_transform_from_basegrid_range(self, x):
        return (x * self.scale) + self.shift

    def log_density(self, x, t, tol=None, method=None):
        # x_transformed, logabsdet_transform = self.domain_transform(x)
        return super().log_density(x, t, method=method, tol=tol)  # - logabsdet_transform

    def velocity(self, x, t):
        # x_transformed, logabsdet_transform = self.domain_transform(x)
        return super().velocity(x, t)  # * (1 / self.domain_transform_J)

    # def log_density_via_ode(self, x, t_now, t_reference, vel_scale):
    #     return super().log_density(x, t_now, tol=1e-7, method="dopri8")
    #
    # def data_independent_loss(self, *args, **kwargs):
    #     return 1e-6 * self.base_grid._base_grid.exp().mean()
