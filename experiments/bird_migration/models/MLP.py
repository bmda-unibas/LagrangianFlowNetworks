import numpy as np
import torch

from experiments.bird_migration.models.util import device, build_scale_to_domain_transform
from experiments.bird_migration.models.template import DensityVelocityInterface
from enflows.nn.nets import ResidualNet
from enflows.utils.torchutils import np_to_tensor

from enflows.CNF.cnf import sample_rademacher_like, sample_gaussian_like, divergence_bf, divergence_approx
from torchdiffeq import odeint as odeint

def build_MLP(train_dataset, **kwargs):
    return VanillaNN(train_dataset).to(device)


class VanillaNN(ResidualNet, DensityVelocityInterface):
    def __init__(self, dataset):
        DIM = 3
        super().__init__(in_features=DIM + 1, out_features=DIM + 1, activation=torch.nn.functional.relu,
                         hidden_features=256,
                         num_blocks=10, use_batch_norm=True,
                         )

        # self.scale_to_domain_transform = build_scale_to_domain_transform(mins=dataset.mins,
        #                                                                  maxs=dataset.maxs,
        #                                                                  max_abs_transformed=0.4
        #                                                                  )
        self.train_mean_log1p = np_to_tensor(np.log1p(dataset.df.birds_per_km3.values).mean(), device=device,
                                             dtype=torch.float32)
        self.train_mean_vel = np_to_tensor(np.nanmean(dataset.df[["u_3035_km_per_min", "v_3035_km_per_min"]].values, 0),
                                           device=device,
                                           dtype=torch.float32)

        self.min_time = dataset.df[dataset.selected_time].min()
        self.max_time = dataset.df[dataset.selected_time].max()

    def log_density(self, x, t):
        return self._log_density_and_velocity(x, t)[0]

    def _log_density_and_velocity(self, x, t):
        try:
            output = self.forward(torch.concatenate([x, t], -1))
        except Exception:
            t = x.new_ones((x.shape[0], 1)) * t
            output = self.forward(torch.concatenate([x, t], -1))

        velocity = output[..., 1:]
        return output[..., :1], velocity

    def velocity(self, x, t):
        return self._log_density_and_velocity(x, t)[1]

    def forward(self, inputs, context=None):
        return super().forward(inputs)


    def odefunc_forward(self, t, states, vel_scale):
        assert len(states) >= 2
        y = states[0]

        # increment num evals

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).type_as(y)
        else:
            t = t.type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        try:
            if self._e is None:
                self._e = sample_rademacher_like(y)
        except AttributeError:
            self._e = sample_rademacher_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)

            dy = vel_scale * self.velocity(x=y, t=t * torch.ones(y.shape[0], 1).to(y))
            dy[..., -1] = 0.  # z velocity is nonsense.. no data for that
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = divergence_bf(dy, y, e=self._e).view(batchsize, 1)

        return tuple([dy, -divergence])


