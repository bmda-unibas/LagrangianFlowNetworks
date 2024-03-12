import abc
from abc import ABC

import enflows.transforms.base
from tqdm import tqdm
from typing import *
from numpy.typing import *
import numpy as np

from enflows.CNF.cnf import sample_rademacher_like, sample_gaussian_like, divergence_bf, divergence_approx
from torchdiffeq import odeint as odeint

import torch
from enflows.utils.torchutils import np_to_tensor, tensor_to_np

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


class DensityVelocityInterface(ABC):

    @abc.abstractmethod
    def log_density(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:
        """

        :param x: shape [mb_size, d]
        :param t: shape [mb_size, 1]
        :return: shape [mb_size]
        """
        pass

    @abc.abstractmethod
    def velocity(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:
        """

        :param x: shape [mb_size, d]
        :param t: shape [mb_size, 1]
        :return: shape [mb_size, d]
        """
        pass

    def log_density_np(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        x_torch = np_to_tensor(x)
        t_torch = np_to_tensor(t)
        return tensor_to_np(self.log_density(x_torch, t_torch))

    def velocity_np(self, x: ArrayLike, t: ArrayLike) -> ArrayLike:
        x_torch = np_to_tensor(x)
        t_torch = np_to_tensor(t)
        return tensor_to_np(self.velocity(x_torch, t_torch))

    @torch.no_grad()
    def predict_logdensity_split(self, XYZT, split_size=4_000):
        XYZ_3035_km = XYZT[..., :-1]
        time_condition = XYZT[..., -1:]

        densities = []
        for _xyz, _tcondition in zip(torch.split(np_to_tensor(XYZ_3035_km, device=device), split_size),
                                     torch.split(np_to_tensor(time_condition, device=device), split_size)):
            tmp_density = self.log_density(x=_xyz, t=_tcondition)
            densities.append(tensor_to_np(tmp_density))

        density_pred = np.concatenate(densities, 0)
        return density_pred.squeeze()

    @torch.no_grad()
    def predict_logdensity_viaode_split(self, XYZT, t_reference, split_size=4_000):
        XYZ_3035_km = XYZT[..., :-1]
        time_condition = XYZT[..., -1:]
        densities = []
        for _xyz, _tcondition, _t_reference in tqdm(zip(torch.split(np_to_tensor(XYZ_3035_km, device=device), split_size),
                                                   torch.split(np_to_tensor(time_condition, device=device), split_size),
                                                   torch.split(np_to_tensor(t_reference, device=device), split_size),
                                                   )):
            tmp_density = self.log_density_via_ode(x=_xyz, t_now=_tcondition,
                                                   t_reference=_t_reference,
                                                   vel_scale=1.)
            densities.append(tensor_to_np(tmp_density))

        density_pred = np.concatenate(densities, 0)
        return density_pred.squeeze()

    @torch.no_grad()
    def predict_vel_split(self, XYZT, split_size=4_000):
        XYZ_3035_km = XYZT[..., :-1]
        time_condition = XYZT[..., -1:]

        velocities = []
        for _xyz, _tcondition in zip(torch.split(np_to_tensor(XYZ_3035_km, device=device), split_size),
                                     torch.split(np_to_tensor(time_condition, device=device), split_size)):
            tmp_vel = self.velocity(x=_xyz, t=_tcondition)
            velocities.append(tensor_to_np(tmp_vel))

        velocity_pred = np.concatenate(velocities, 0)
        return velocity_pred.squeeze()

    def odefunc_forward(self, t, states, vel_scale):
        assert len(states) >= 2
        y = states[0]
        y = torch.clip(y, min=-3.99, max=3.99)
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
            # Hack for 2D data to use brute force divergence computation.
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                divergence = divergence_bf(dy, y, e=self._e).view(batchsize, 1)

        return tuple([dy, -divergence])

    def integrate(self, z, integration_times, vel_scale):
        _logpz = torch.zeros(z.shape[0], 1).to(z)
        state_t = odeint(
            lambda t, states: self.odefunc_forward(t, states, vel_scale),
            (z, _logpz),
            # torch.linspace(integration_times[0], integration_times[-1], 1_000).to(z),
            integration_times.to(z),
            atol=1e-5,
            rtol=1e-5,
            method="dopri8",
            # options=dict(step_size=1e-2)
            # step_size=self.solver_options["step_size"]
        )

        z_t, logpz_t = tuple(s[1] for s in state_t)
        return z_t, logpz_t

    def log_density_via_ode(self, x, t_now, t_reference, vel_scale):
        assert torch.allclose(t_now[0] * torch.ones_like(t_now), t_now)
        assert torch.allclose(t_reference[0] * torch.ones_like(t_now), t_reference)
        z_base, divergence_accum_tbase = self.integrate(x, torch.tensor([t_now[0], t_reference[0]]),
                                                        vel_scale=vel_scale)
        z_base = torch.clip(z_base, min=-3.999, max=3.999)

        density_at_z_base = self.log_density(z_base, t_reference * torch.ones(x.shape[0], 1).to(x)).view(-1)
        logp_x = density_at_z_base - divergence_accum_tbase.view(-1)
        return logp_x


    def evaluate_consistency_loss(self, data_gen, ode_split_size=4096):
        np.random.seed(1000)
        differences = []
        if data_gen.dim == 2:
            threshold = 0.1
        else:
            threshold = 0.01

        for t in tqdm(np.linspace(0.1, 1.2, 10)):
            XYZ_3035_km, data_t, logpdf, vel = data_gen.sample_data(200_000, t_subset="discrete_single_t", subset="all")
            idx = np.where(np.exp(logpdf)>threshold)[0][:5000]
            data_t = data_t * 0. + t
            XYZ_3035_km = XYZ_3035_km[idx]
            XYZ_3035_km = np.clip(XYZ_3035_km, -3.5, 3.5)

            data_t = data_t[idx]

            t_reference = np.zeros((XYZ_3035_km.shape[0], 1))
            # list_of_times = np.linspace(0 + 1e-3, 1.2, 10).tolist()
            # list_of_times = [0.25, 0.5, 0.75, 1.]
            time_condition = data_t

        # for _time_scalar in tqdm(list_of_times):
        #     time_condition = np.ones((XYZ_3035_km.shape[0], 1)) * _time_scalar
            XYZT = np.concatenate([XYZ_3035_km, time_condition[:, np.newaxis]], -1)
            density_odesolve = np.exp(self.predict_logdensity_viaode_split(
                XYZT,
                t_reference=t_reference,
                split_size=ode_split_size))
            density_model = np.exp(self.predict_logdensity_split(XYZT))

            assert density_odesolve.shape == density_model.shape
            mape = np.abs(density_model - density_odesolve) / (np.abs(density_model) + np.abs(density_odesolve) + 1e-6)
            differences.append(mape)
        difference = np.stack(differences, 0).mean()
        return difference

    # def evaluate_consistency_loss(self, data_gen, ode_split_size=4096):
    #     np.random.seed(1000)
    #     XYZ_3035_km, ___, _, __ = data_gen.sample_data(2_000, t_subset="all", subset="all")
    #     XYZ_3035_km = np.clip(XYZ_3035_km, -3.9, 3.9)
    #     t_reference = np.zeros((XYZ_3035_km.shape[0], 1))
    #     # list_of_times = np.linspace(0 + 1e-3, 1.2, 10).tolist()
    #     list_of_times = [0.25, 0.5, 0.75, 1.]
    #
    #     differences = []
    #     for _time_scalar in tqdm(list_of_times):
    #         time_condition = np.ones((XYZ_3035_km.shape[0], 1)) * _time_scalar
    #         XYZT = np.concatenate([XYZ_3035_km, time_condition], -1)
    #         density_odesolve = np.exp(self.predict_logdensity_viaode_split(
    #             XYZT,
    #             t_reference=t_reference,
    #             split_size=ode_split_size))
    #         density_model = np.exp(self.predict_logdensity_split(XYZT))
    #
    #         assert density_odesolve.shape == density_model.shape
    #         mape = np.abs(density_model - density_odesolve) / (np.abs(density_model) + np.abs(density_odesolve) + 1e-3)
    #         differences.append(mape)
    #
    #     return np.stack(differences, 0).mean()
