import torch
from experiments.bird_migration.models.template import DensityVelocityInterface
from experiments.bird_migration.models.util import device, build_scale_to_domain_transform
from datasets.bird_data import BirdDatasetMultipleNightsForecast as BirdData

from siren_pytorch import SirenNet
from enflows.utils.torchutils import gradient, divergence
from enflows.transforms import ActNorm
from enflows.nn.nets import ResidualNet

def build_PINN(num_layers, hidden_features, sine_frequency, complete_dataset, **kwargs):
    return ResnetPINN(space_dim=3, hidden_features=hidden_features, sine_frequency=sine_frequency,
                      complete_dataset=complete_dataset, num_layers=num_layers).to(device)



class MassConsPINN(torch.nn.Module, DensityVelocityInterface):
    def __init__(self, space_dim,
                 hidden_features,
                 num_layers,
                 complete_dataset: BirdData,
                 **kwargs):
        super().__init__()
        self.data_gen = complete_dataset
        self.space_dim = space_dim
        self.in_features = self.out_features = space_dim + 1

        # self.domain_transform = build_scale_to_domain_transform(mins=self.data_gen.mins,
        #                                                         maxs=self.data_gen.maxs,
        #                                                         max_abs_transformed=0.1
        #                                                         )
        #
        self.domain_transform = ActNorm(3)
        min_time = complete_dataset.df[complete_dataset.selected_time].min(),
        max_time = complete_dataset.df[complete_dataset.selected_time].max(),
        self.min_time = torch.nn.Parameter(torch.tensor(min_time, dtype=torch.float32),
                                           requires_grad=True)
        self.max_time = torch.nn.Parameter(torch.tensor(max_time, dtype=torch.float32),
                                           requires_grad=True)

        self._t_idx = 0
        self._x_idx = [i for i in range(self.in_features) if i != self._t_idx]

        self.sampler = torch.quasirandom.SobolEngine(4, seed=1234)

        self.network_rho = SirenNet(
            dim_in=self.in_features,  # input dimension, ex. 2d coor
            dim_hidden=hidden_features,  # hidden dimension
            dim_out=1,  # output dimension, ex. rgb value
            num_layers=num_layers,  # number of layers
            final_activation=torch.nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=kwargs.get("sine_frequency", 30)
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )
        self.network_v = SirenNet(
            dim_in=self.in_features,  # input dimension, ex. 2d coor
            dim_hidden=hidden_features,  # hidden dimension
            dim_out=self.out_features - 1,  # output dimension, ex. rgb value
            num_layers=num_layers,  # number of layers
            final_activation=torch.nn.Identity(),  # activation of final layer (nn.Identity() for direct output)
            w0_initial=kwargs.get("sine_frequency", 30)
            # different signals may require different omega_0 in the first layer - this is a hyperparameter
        )

        self._num_evals = 0
        self._e = None

    def forward(self, x, mods=None):
        # _x = torch.empty_like(x)
        # time_transformed = (x[..., 0] - self.min_time.to(x)) / (self.max_time - self.min_time).to(x)
        # space_transformed, _ = self.domain_transform(x[..., 1:]*2 -1)
        time_transformed = x[..., 0]
        space_transformed = x[..., 1:]

        x_transformed = torch.concatenate([time_transformed.view(-1, 1), space_transformed], -1)
        vel = self.network_v(x_transformed, mods)
        rho = self.network_rho(x_transformed, mods)
        return torch.concatenate([rho, vel], -1)


    def log_density(self, x, t):
        return self._log_density_and_velocity(x, t)[0]

    def velocity(self, x, t):
        return self._log_density_and_velocity(x, t)[1]

    def _log_density_and_velocity(self, x, t):
        try:
            output = self.forward(torch.concatenate([x, t], -1))
        except Exception:
            t = x.new_ones((x.shape[0], 1)) * t
            output = self.forward(torch.concatenate([x, t], -1))

        velocity = output[..., 1:]
        log_density = output[..., :1]
        return log_density, velocity

    def pde_loss(self, n_samples=4096):
        t, x = self.sample_input_domain(n_samples)
        log_rho, pde_loss = self._pde_loss(x=x, t=t)
        pde_loss = pde_loss.mean().sqrt()
        # pde_term_sq = (drho_dt + div_massflux).pow(2)
        # sparsity_loss = 1e-5 * torch.nn.functional.softplus(log_rho).mean()
        return pde_loss  # + sparsity_loss

    def _pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        log_rho, velocity = self._log_density_and_velocity(x = x, t = t)
        rho = torch.exp(log_rho)
        mass_flux = rho * velocity
        # mass_flux[..., 0] /= self.data_gen.scale_space
        # mass_flux[..., 1] /= self.data_gen.scale_space
        div_massflux = divergence(mass_flux, x, x_offset=0)
        drho_dt = gradient(rho, t)[..., [0]]
        pde_loss = ((drho_dt + div_massflux)).pow(2)  # - 1e-2*rho.mean()
        return log_rho, pde_loss

    def sample_input_domain(self, n_samples: int):
        data_tx_raw = self.sampler.draw(n_samples).to(device)

        # data_t = self.sample_quasiuniform_time(n_samples, device=device)
        # data_x = self.domain_transform.inverse(data_tx_raw[..., 1:])[0]

        # - self.min_time.to(x)) / (self.max_time - self.min_time).to(x)
        data_t = data_tx_raw[..., :1] * (self.max_time - self.min_time).to(data_tx_raw) + self.min_time.to(data_tx_raw)
        _mins = torch.tensor(self.data_gen.mins).to(device)
        _maxs = torch.tensor(self.data_gen.maxs).to(device)
        _mins[:-1] *= self.data_gen.scale_space
        _maxs[:-1] *= self.data_gen.scale_space
        _mins = _mins * 0.8
        _maxs = _maxs * 1.2
        _diff = _maxs - _mins
        data_x = data_tx_raw[..., 1:] * _diff + _mins
        return data_t.detach(), data_x.detach()

    def data_independent_loss(self, pde_weight, collocation_points, **kwargs):
        return  pde_weight * self.pde_loss(collocation_points)




class ResnetPINN(MassConsPINN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = ResidualNet(in_features=self.in_features, out_features=self.out_features,
                                   activation=torch.nn.functional.relu,
                                   hidden_features=256,
                                   num_blocks=10, use_batch_norm=True,
                                   )

    def forward(self, x, mods=None):
        return self.network(x)

    def log_density(self, x, t):
        return self._log_density_and_velocity(x, t)[0]

    def velocity(self, x, t):
        return self._log_density_and_velocity(x, t)[1]

    def _log_density_and_velocity(self, x, t):
        try:
            output = self.forward(torch.concatenate([x, t], -1))
        except Exception:
            t = x.new_ones((x.shape[0], 1)) * t
            output = self.forward(torch.concatenate([x, t], -1))

        velocity = output[..., 1:]
        log_density = output[..., :1]
        return log_density, velocity

    def pde_loss(self, n_samples=4096):
        t, x = self.sample_input_domain(n_samples)
        log_rho, pde_loss = self._pde_loss(x=x, t=t)
        pde_loss = pde_loss.mean().sqrt()
        # pde_term_sq = (drho_dt + div_massflux).pow(2)
        # sparsity_loss = 1e-5 * torch.nn.functional.softplus(log_rho).mean()
        return pde_loss  # + sparsity_loss

    def _pde_loss(self, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        log_rho, velocity = self._log_density_and_velocity(x = x, t = t)
        rho = torch.exp(log_rho)
        mass_flux = rho * velocity
        # mass_flux[..., 0] /= self.data_gen.scale_space
        # mass_flux[..., 1] /= self.data_gen.scale_space
        div_massflux = divergence(mass_flux, x, x_offset=0)
        drho_dt = gradient(rho, t)[..., [0]]
        pde_loss = ((drho_dt + div_massflux)).pow(2)  # - 1e-2*rho.mean()
        return log_rho, pde_loss

    def sample_input_domain(self, n_samples: int):
        data_tx_raw = self.sampler.draw(n_samples).to(device)

        # data_t = self.sample_quasiuniform_time(n_samples, device=device)
        # data_x = self.domain_transform.inverse(data_tx_raw[..., 1:])[0]

        # - self.min_time.to(x)) / (self.max_time - self.min_time).to(x)
        data_t = data_tx_raw[..., :1] * (self.max_time - self.min_time).to(data_tx_raw) + self.min_time.to(data_tx_raw)
        _mins = torch.tensor(self.data_gen.mins).to(device)
        _maxs = torch.tensor(self.data_gen.maxs).to(device)
        _mins[:-1] *= self.data_gen.scale_space
        _maxs[:-1] *= self.data_gen.scale_space
        _diff = _maxs - _mins
        data_x = data_tx_raw[..., 1:] * _diff + _mins
        return data_t.detach(), data_x.detach()

    def data_independent_loss(self, pde_weight, collocation_points, **kwargs):
        return  pde_weight * self.pde_loss(collocation_points)
