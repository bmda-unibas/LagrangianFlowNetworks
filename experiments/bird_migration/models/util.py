import numpy as np
import torch
from torch import tensor

from enflows.nn.nets import ResidualNet
from enflows.transforms import CompositeTransform, PointwiseAffineTransform
from siren_pytorch import SirenNet
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def build_scale_to_domain_transform(mins, maxs, max_abs_transformed=0.7, return_jacobian=False):
    _mins = tensor(mins).to(device)
    _maxs = tensor(maxs).to(device)
    _diff = _maxs - _mins

    last_scaling = torch.ones_like(_diff)
    last_scaling[0] = max_abs_transformed
    last_scaling[1] = max_abs_transformed
    last_scaling[2] = 0.5
    transform = CompositeTransform([
        PointwiseAffineTransform(shift=-_mins),
        PointwiseAffineTransform(scale=2 / _diff, shift=-torch.ones_like(_diff)),
        PointwiseAffineTransform(scale=last_scaling),
    ])
    if return_jacobian:
        return transform, (2/_diff)*last_scaling
    return transform


def matern32(dists, lengthscale=1.):
    dists = dists / lengthscale
    K = dists * np.sqrt(3)
    K = (1.0 + K) * torch.exp(-K)
    return K


def matern52(dists, lengthscale=1):
    K = (dists / lengthscale) * np.sqrt(5)
    K = (1.0 + K + K ** 2 / 3.0) * torch.exp(-K)
    return K


class ScaledResidualNet(ResidualNet):
    def __init__(
            self,
            min_time,
            max_time,
            in_features,
            out_features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            scaling_value=1.
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

        self.min_time = torch.nn.Parameter(torch.tensor(min_time, dtype=torch.float32),
                                           requires_grad=True)
        self.max_time = torch.nn.Parameter(torch.tensor(max_time, dtype=torch.float32),
                                           requires_grad=True)

        self.scaling_value = torch.nn.Parameter(torch.tensor(scaling_value, dtype=torch.float32),
                                                requires_grad=True)

    def forward(self, inputs, context=None):
        new_inputs = (inputs - self.min_time) / (self.max_time - self.min_time)
        return super().forward(new_inputs * self.scaling_value, context=context)

class ScaledSIREN(SirenNet):
    def __init__(
            self,
            min_time,
            max_time,
            in_features,
            out_features,
            hidden_features,
            num_blocks=2,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            scaling_value=1.
    ):
        super().__init__(
            dim_in=in_features,
            dim_out=out_features,
            dim_hidden=hidden_features,
            num_layers=num_blocks,
            w0_initial=2,
        )

        self.min_time = torch.nn.Parameter(torch.tensor(min_time, dtype=torch.float32),
                                           requires_grad=True)
        self.max_time = torch.nn.Parameter(torch.tensor(max_time, dtype=torch.float32),
                                           requires_grad=True)

        self.scaling_value = torch.nn.Parameter(torch.tensor(scaling_value, dtype=torch.float32),
                                                requires_grad=True)

    def forward(self, inputs, context=None):
        new_inputs = (inputs - self.min_time) / (self.max_time - self.min_time)
        return super().forward(new_inputs * self.scaling_value)

