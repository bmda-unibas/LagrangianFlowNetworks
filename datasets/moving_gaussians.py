import numpy as np
import numpy.typing
import scipy as sp
from numpy._typing import ArrayLike
from scipy.stats import norm
import torch
import tqdm
from typing import *
from experiments.gaussians.density_velocity_interface import DensityVelocityInterface
from sklearn.metrics import r2_score

from pprint import pformat
import os
from enflows.utils.torchutils import batch_jacobian
import enflows.utils.torchutils as torchutils
from enflows.utils.torchutils import tensor_to_np, np_to_tensor


def gen_data_pinwheel(batch_size, num_classes=9):
    radial_std = 0.3
    tangential_std = 0.1
    num_per_class = batch_size // 5
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
               * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    x = 2 * np.einsum("ti,tij->tj", features, rotations)
    idx_permuted = np.random.permutation(np.arange(x.shape[0]))
    x_permuted = x[idx_permuted]
    label_permuted = labels[idx_permuted]
    label_permuted = label_permuted / label_permuted.max()
    return x_permuted, label_permuted


class MovingGaussians(DensityVelocityInterface):
    def __init__(self, num_timesteps=21, num_gaussians=4, scale=.3, n_samples_lagr=300):
        # self.radius_scalar = 1. / 3
        self.radius_scalar = 2. / 3
        self.radiuses_base = np_to_tensor(
            np.linspace(0, 1., num_gaussians) + self.radius_scalar)  # torch.ones(num_gaussians) * self.radius_scalar
        # self.radiuses_base = 1 - np.linspace(0., .8, num_gaussians)

        self.num_timesteps = num_timesteps
        self.num_gaussians = num_gaussians
        self.dim = 2

        self.timesteps = torch.linspace(0, 1, self.num_timesteps)
        self.timesteps_often = torch.linspace(0, 1, self.num_timesteps * 2)
        self.gauss_mixtures = torch.arange(num_gaussians)
        self.p_gauss_mixtures = torch.ones_like(self.gauss_mixtures) / len(self.gauss_mixtures)
        self.scale = scale

        # self.mixture_angles_radian = torch.linspace(0, 2 * np.pi, self.num_gaussians)
        self.mixture_angles_radian = np.linspace(0, 2 * np.pi, self.num_gaussians, endpoint=False)
        self.mixture_angles_radian = np_to_tensor(self.mixture_angles_radian)

        self.omega_x = .1
        self.omega_y = .1
        self.omega_xy = 0

        self.omega_rot = 4. / 2 * np.pi
        self.shift_dt = np_to_tensor(np.array([.6, -.6]))

        self.lim = 4 - 1e-3
        self.max_t_test = 1.25
        self.sampler = torch.quasirandom.SobolEngine(2, seed=1234)
        self.sampler_time = torch.quasirandom.SobolEngine(1, seed=1234)
        self.rng = np.random.default_rng(12345)

        # self.lagrangian_samples_rho0 = self.sample_from_rho0_subset(n_samples_lagr, subset="train")
        self.lagrangian_samples_rho0 = self.sample_from_rho0(n_samples_lagr)

    def score_density(self, model: DensityVelocityInterface,
                      scorer: Callable = r2_score, device="cpu",
                      n_space=1_000, n_times=20, t_max=1.2, split_size=1_000,
                      subset="test", verbose=False):
        assert subset in ["train", "val", "test", "all"]
        self.sampler.reset()
        self.sampler_time.reset()

        xy = self.sample_quasiuniform_space(n_space, device=device, subset=subset)
        t = self.sample_quasiuniform_time(n_times, t_max, device=device)
        xy_rep = xy.repeat(n_times, 1)
        t_rep = t.repeat_interleave(n_space, 0)
        # t_rep = t_rep[np.random.choice(t_rep.shape[0], t_rep.shape[0])]
        true_densities = []
        pred_densities = []
        for _xy, _t in tqdm.tqdm(zip(torch.split(xy_rep, split_size), torch.split(t_rep, split_size)),
                                 total=t_rep.shape[0] // split_size, disable=not verbose):
            true_density = np.exp(tensor_to_np(self.log_density(x=_xy, t=_t)))
            pred_density = np.exp(tensor_to_np(model.log_density(x=_xy, t=_t)))

            true_densities.append(true_density)
            pred_densities.append(pred_density)
        true_density = np.concatenate(true_densities, 0)
        pred_density = np.concatenate(pred_densities, 0)
        return scorer(true_density, pred_density)

    # def score_velocity(self, model: DensityVelocityInterface,
    #                    scorer: Callable = lambda y_t, y: r2_score(y_t, y, multioutput="raw_values"),device="cpu",
    #                    n_samples=1_000, t_max=1.2):
    #     self.sampler.reset()
    #     xy = self.sample_quasiuniform(n_samples, t_max, device=device)
    #     true_velocity = tensor_to_np(self.velocity(x=xy, t=t))
    #     pred_velocity = tensor_to_np(model.velocity(x=xy, t=t))
    #     return scorer(true_velocity, pred_velocity)

    def sample_uniform_time(self, n_samples, t_subset="discrete"):
        # assert t_subset in ["discrete", "all", "t0", "discrete_single_t"]
        if t_subset == "discrete":
            data_t = self.sample_time(n_samples)  # np.random.rand(mb_size)
        elif t_subset == "discrete_often":
            data_t = self.rng.choice(self.timesteps_often, n_samples)
        elif t_subset == "discrete_single_t":
            data_t = np.broadcast_to(np.random.choice(self.timesteps, 1), n_samples)
        elif t_subset == "t0":
            data_t = np.zeros(n_samples)
        elif t_subset == "all":
            data_t = np.broadcast_to(np.random.rand(n_samples) * self.max_t_test, n_samples)
        else:
            raise ValueError("Invalid Input")
        return data_t

    def sample_quasiuniform_space(self, n_samples, subset, device="cpu"):
        assert subset in ["train", "val", "test", "all", "right", "all_small"]
        raw_samples = self.sampler.draw(n_samples).to(device)
        # shuffle because quasi random samples are ordered
        idx = torch.randperm(raw_samples.shape[0])
        raw_samples = raw_samples[idx].view(raw_samples.size())

        if subset == "train":
            samples = self.uniform_to_train_space(raw_samples)
        elif subset == "val":
            samples = self.uniform_to_val_space(raw_samples)
        elif subset == "test":
            samples = self.uniform_to_test_space(raw_samples)
        elif subset == "right":
            samples = self.uniform_to_rightside(raw_samples)
        elif subset == "all_small":
            samples = self.uniform_to_domain(raw_samples) * 0.8
        else:
            samples = self.uniform_to_domain(raw_samples)
        return samples  # , t

    def uniform_to_domain(self, raw_samples):
        raw_samples[..., 0] = (raw_samples[..., 0] * 2 - 1) * self.lim
        raw_samples[..., 1] = (raw_samples[..., 1] * 2 - 1) * self.lim
        return raw_samples

    def uniform_to_rightside(self, raw_samples):
        raw_samples[..., 0] = (raw_samples[..., 0]) * self.lim
        raw_samples[..., 1] = (raw_samples[..., 1]) * self.lim
        return raw_samples

    def uniform_to_val_space(self, raw_samples):
        data_x_topleft = raw_samples[::2, ...] * self.lim
        data_x_topleft[..., 0] *= -0.2
        data_x_bottomright = -raw_samples[1::2, ...] * self.lim
        data_x_bottomright[..., 0] *= -0.2
        samples = torch.concatenate([data_x_topleft, data_x_bottomright], 0)
        return samples

    def uniform_to_test_space(self, raw_samples):
        data_x_topleft = raw_samples[::2, ...] * self.lim
        data_x_topleft[..., 0] = -0.8 * data_x_topleft[..., 0] - 0.2 * self.lim
        data_x_bottomright = -raw_samples[1::2, ...] * self.lim
        data_x_bottomright[..., 0] = -0.8 * data_x_bottomright[..., 0] + 0.2 * self.lim
        samples = torch.concatenate([data_x_topleft, data_x_bottomright], 0)
        return samples

    def uniform_to_train_space(self, raw_samples):
        data_x_topright = raw_samples[::2, ...] * self.lim
        data_x_bottomleft = -raw_samples[1::2, ...] * self.lim
        samples = torch.concatenate([data_x_topright, data_x_bottomleft], 0)
        return samples

    def sample_quasiuniform_time(self, n_samples, t_max, device="cpu"):
        samples = self.sampler_time.draw(n_samples).to(device)
        t = samples * t_max
        return t

    def velocity(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:
        return torch.stack(self.dxy_dt(x=x[..., 0].reshape(-1), y=x[..., 1].reshape(-1), time=t.reshape(-1)), -1)

    def log_density(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:
        return self.logp_at_t(position=x, time=t.reshape(-1))

    def xt_to_x0(self, x, t):
        x = np_to_tensor(x)
        t = np_to_tensor(t)
        x = 4 * torch.arctanh(x / 4)
        x /= (t.reshape(-1, 1) / 2 + 1)

        proj = self.get_projection_matrix(t).squeeze()
        # proj_inv = torch.linalg.inv(proj)
        # x = torch.arctanh(x / 2)
        # result = proj_inv @ (x - self.shift_dt * t)[..., None]
        x_shifted = (x)[..., None]
        result = torch.linalg.solve(proj, x_shifted).squeeze() - self.get_shift(t.reshape(-1, 1))
        return self.lim * torch.tanh(result / self.lim)

    def x0_to_xt(self, x0, t):
        x0 = np_to_tensor(x0)
        t = np_to_tensor(t)
        x_ = self.lim * torch.arctanh(x0 / self.lim)

        proj = self.get_projection_matrix(t).squeeze()
        result = (proj @ (x_ + self.get_shift(t.reshape(-1, 1)))[..., None]).squeeze()
        result *= (t.reshape(-1, 1) / 2 + 1)
        return self.lim * torch.tanh(result / self.lim)
        # return np.tanh(result) * 2

    def get_shift(self, t):
        return torch.sin(np.pi * t) * self.shift_dt.to(t)

    def get_projection_matrix(self, t):
        """
        Returns position of given particle at time t,
        :param t:
        :return:
        """
        proj = self.projection_matrix_rotate(t) @ self.projection_matrix_shear(t)
        return proj

    def projection_matrix_rotate(self, t):
        """
        Returns position of given particle at time t,
        :param t:
        :return:
        """
        w1, w2 = torch.cos(self.omega_rot * t), -torch.sin(self.omega_rot * t)
        w3, w4 = torch.sin(self.omega_rot * t), torch.cos(self.omega_rot * t)
        matrix = torch.stack([torch.stack([w1, w2], -1),
                              torch.stack([w3, w4], -1)], -2)

        return matrix

    def projection_matrix_shear(self, t):
        w1, w2, w3, w4 = (1 + t * self.omega_x), (t * self.omega_xy), torch.zeros_like(t), (1 + t * self.omega_y)
        matrix = torch.stack([torch.stack([w1, w2], -1), torch.stack([w3, w4], -1)], -2)
        return matrix

    def sample_time(self, batch_size):
        # return np.random.choice(self.timesteps, batch_size)
        return self.rng.choice(self.timesteps, batch_size)

    def get_mixture_center(self, mixture):
        angles = self.mixture_angles_radian[mixture]
        radiuses = self.radiuses_base[mixture]
        return self.polar_to_cartesian(radiuses, angles)

    @staticmethod
    def polar_to_cartesian(radiuses, angles):
        radiuses = np.array([radiuses]) if np.isscalar(radiuses) else radiuses
        angles = np.array([angles]) if np.isscalar(angles) else angles

        x = radiuses * np.cos(angles)
        y = radiuses * np.sin(angles)
        return np.stack([np.squeeze(x), np.squeeze(y)], -1)

    def logprob_mixture(self, mixture):
        return self.p_gauss_mixtures[mixture]

    def logprob_x_cond_on_mixture(self, positions, mixture):
        centers = self.get_mixture_center(mixture=mixture)
        log_probs = sp.stats.norm.logpdf(positions, centers, self.scale).sum(-1)
        return log_probs

    def logp_at_t0(self, position) -> numpy.typing.ArrayLike:
        # position = np.arctanh(position/4)*4
        assert position.shape[-1] == 2
        mixturewise_logprobs = []
        for mixture in self.gauss_mixtures:
            mixturewise_logprobs.append(
                self.logprob_x_cond_on_mixture(position, mixture=mixture))
        mixturewise_logprobs = np.array(mixturewise_logprobs)
        mixture_logprobs = np.log(self.p_gauss_mixtures)[..., np.newaxis]
        logsumexp = sp.special.logsumexp(mixture_logprobs + mixturewise_logprobs, axis=0)
        return logsumexp

    def logp_at_t(self, position, time):
        # position_at_t0 = (projection_matrix_inv @ ((position - self.shift_dt * time.squeeze())[..., None])).squeeze()
        with torch.enable_grad():
            position = np_to_tensor(position).requires_grad_(True)
            position_at_t0 = self.xt_to_x0(position, time.squeeze())
            logp_at_t0 = self.logp_at_t0(position_at_t0.detach().cpu().numpy())
            jac = batch_jacobian(position_at_t0, position)
            logabsdet = - torch.slogdet(jac)[1]
        # logabsdet = np.log(1+time*self.omega_x) + np.log(1+time*self.omega_y)
        # projection_matrix = self.get_projection_matrix(cast_to_tensor(time)).squeeze().numpy()
        # logabsdet_ref = np.linalg.slogdet(projection_matrix)[1]
        return logp_at_t0 - logabsdet.detach().cpu().numpy()

    def dxy_dt(self, x, y, time=None):
        with torch.enable_grad():
            x = np_to_tensor(x)
            y = np_to_tensor(y)
            time = np_to_tensor(time).requires_grad_(True)
            xy = torch.stack([x, y], -1)

            xy_0 = self.xt_to_x0(xy, t=time).detach().requires_grad_(True)
            xy_t = self.x0_to_xt(xy_0, t=time)
            vel_x = torchutils.gradient(xy_t[..., 0], time)
            vel_y = torchutils.gradient(xy_t[..., 1], time)
        return vel_x.detach(), vel_y.detach()

    def sample_data(self, n_samples, subset="train", t_subset="discrete"):
        assert subset in ["train", "val", "test", "all", "right", "all_small"]
        assert t_subset in ["discrete", "all", "t0", "discrete_single_t"]

        # data_x[:, 0] = np.abs(data_x[:, 0])
        data_x = self.sample_quasiuniform_space(n_samples=n_samples, subset=subset, device="cpu").numpy()
        data_t = self.sample_uniform_time(n_samples=n_samples, t_subset=t_subset)

        vel = np.stack(self.dxy_dt(*data_x.T, data_t), -1)
        vel = vel + 0.1 * self.rng.standard_normal(vel.shape) * 0.14  # ~10% std noise
        # logpdf = self.logp_at_t(data_x, time=data_t)
        logpdf = self.logp_at_t(data_x, time=data_t)
        logpdf = logpdf + 0.1 * self.rng.standard_normal(logpdf.shape) * 0.23  # ~10% std noise
        return data_x, data_t, logpdf, vel

    def sample_from_rho0(self, n_samples):
        mixture_samples = self.rng.choice(self.gauss_mixtures, p=self.p_gauss_mixtures, size=(n_samples,))
        mixture_centers = self.get_mixture_center(mixture_samples)
        samples = sp.stats.norm.rvs(loc=mixture_centers, scale=self.scale)
        return samples  # , self.logp_at_t0(samples)

    def sample_from_rho0_subset(self, n_samples, subset):
        grid = self.sample_quasiuniform_space(n_samples * 10, subset=subset)
        grid_density = np.exp(self.logp_at_t0(grid))
        grid_density /= np.sum(grid_density)
        samples = self.rng.choice(grid, size=(n_samples,), p=grid_density)
        return samples  # , self.logp_at_t0(samples)

    def sample_lagrangian_data(self, n_samples, t_subset="discrete"):
        assert t_subset in ["discrete", "all", "t0", "discrete_single_t", "discrete_often"]

        # data_x[:, 0] = np.abs(data_x[:, 0])
        data_x0 = self.rng.choice(self.lagrangian_samples_rho0, size=(n_samples,), replace=True)
        data_t = self.sample_uniform_time(n_samples=n_samples, t_subset=t_subset)
        data_x = self.x0_to_xt(data_x0, data_t)

        vel = np.stack(self.dxy_dt(*data_x.T, data_t), -1)
        vel = vel + 0.1 * self.rng.standard_normal(vel.shape) * 0.14  # ~10% std noise
        # logpdf = self.logp_at_t(data_x, time=data_t)
        logpdf = self.logp_at_t(data_x, time=data_t)
        logpdf = logpdf + 0.1 * self.rng.standard_normal(logpdf.shape) * 0.23  # ~10% std noise

        return data_x, data_t, logpdf, vel, data_x0 + np.random.randn(*data_x0.shape) * 0.05

    def sample_data_ivp(self, mb_size, to_tensor=False, device="cpu"):
        """
        initial condition (i.e. density at t=0) and velocity at t for t in [0, 1]
        """
        # data_x_rho = np.random.rand(mb_size, 2) * 3 - 1.5
        # data_t_rho = np.zeros(mb_size)
        #
        # data_x_vel = np.random.rand(mb_size, 2) * 3 - 1.5
        # data_t_vel = self.sample_time(mb_size)  # np.random.rand(self.mb_size)
        # vel = np.stack(self.dxy_dt(*data_x_vel.T, data_t_vel), -1)

        data_x_rho = self.sample_quasiuniform_space(n_samples=mb_size, subset="all", device="cpu").numpy()
        data_x_vel = self.sample_quasiuniform_space(n_samples=mb_size, subset="all", device="cpu").numpy()
        data_t_vel = self.sample_uniform_time(n_samples=mb_size, t_subset="all")
        data_t_rho = 0 * data_t_vel

        vel = np.stack(self.dxy_dt(*data_x_vel.T, data_t_vel), -1)
        # vel = vel + 0.1 * self.rng.standard_normal(vel.shape) * 0.14  # ~10% std noise
        logpdf = self.logp_at_t(data_x_rho, time=data_t_rho)

        # logpdf = self.logp_at_t(data_x_rho, time=data_t_rho)

        return_values = (data_x_rho, data_t_rho, logpdf, data_x_vel, data_t_vel, vel)

        if to_tensor:
            return_values = tuple(torch.tensor(val, dtype=torch.float32).reshape(mb_size, -1).to(device)
                                  for val in return_values)
        else:
            return_values = tuple(val.astype("float32") for val in
                                  return_values)

        return return_values


class MovingGaussiansRotOnly(MovingGaussians):
    def get_projection_matrix(self, t):
        """
        Returns position of given particle at time t,
        :param t:
        :return:
        """
        proj = self.projection_matrix_rotate(t)  # @ self.projection_matrix_shear(t)
        return proj


class MovingGaussians3d(DensityVelocityInterface):
    def __init__(self, num_timesteps=21, num_gaussians=4, scale=.3, scale_z=.6):
        # self.radius_scalar = 1. / 3
        self.radius_scalar = 2. / 3
        self.radiuses_base = np_to_tensor(
            np.linspace(0, 1., num_gaussians) + self.radius_scalar)  # torch.ones(num_gaussians) * self.radius_scalar
        # self.radiuses_base = 1 - np.linspace(0., .8, num_gaussians)

        self.num_timesteps = num_timesteps
        self.num_gaussians = num_gaussians
        self.dim = 3

        self.timesteps = torch.linspace(0, 1, self.num_timesteps)
        self.gauss_mixtures = torch.arange(num_gaussians)
        self.p_gauss_mixtures = torch.ones_like(self.gauss_mixtures) / len(self.gauss_mixtures)
        self.scale = [scale, scale, scale_z]

        # self.mixture_angles_radian = torch.linspace(0, 2 * np.pi, self.num_gaussians)
        self.mixture_angles_radian = np.linspace(0, 2 * np.pi, self.num_gaussians, endpoint=False)
        self.mixture_angles_radian = np_to_tensor(self.mixture_angles_radian)

        self.omega_x = .1
        self.omega_y = .1
        self.omega_xy = 0

        self.omega_rot = 4. / 2 * np.pi
        self.shift_dt = np_to_tensor(np.array([.6, -.6]))

        self.lim = 4 - 1e-3
        self.max_t_test = 1.25
        self.sampler = torch.quasirandom.SobolEngine(3, seed=1234)
        self.sampler_time = torch.quasirandom.SobolEngine(1, seed=1234)

    def score_density(self, model: DensityVelocityInterface,
                      scorer: Callable = r2_score, device="cpu",
                      n_space=1_000, n_times=20, t_max=1.2, split_size=1_000,
                      subset="test", verbose=False):
        assert subset in ["train", "val", "test", "all"]
        self.sampler.reset()
        self.sampler_time.reset()

        xy = self.sample_quasiuniform_space(n_space, device=device, subset=subset)
        t = self.sample_quasiuniform_time(n_times, t_max, device=device)
        xy_rep = xy.repeat(n_times, 1)
        t_rep = t.repeat_interleave(n_space, 0)
        # t_rep = t_rep[np.random.choice(t_rep.shape[0], t_rep.shape[0])]
        true_densities = []
        pred_densities = []

        for _xy, _t in zip(torch.split(xy_rep, split_size), torch.split(t_rep, split_size)):
            true_density = np.exp(tensor_to_np(self.log_density(x=_xy, t=_t)))
            pred_density = np.exp(tensor_to_np(model.log_density(x=_xy, t=_t)))

            true_densities.append(true_density)
            pred_densities.append(pred_density)
        true_density = np.concatenate(true_densities, 0)
        pred_density = np.concatenate(pred_densities, 0)
        return scorer(true_density, pred_density)

    # def score_velocity(self, model: DensityVelocityInterface,
    #                    scorer: Callable = lambda y_t, y: r2_score(y_t, y, multioutput="raw_values"),device="cpu",
    #                    n_samples=1_000, t_max=1.2):
    #     self.sampler.reset()
    #     xy = self.sample_quasiuniform(n_samples, t_max, device=device)
    #     true_velocity = tensor_to_np(self.velocity(x=xy, t=t))
    #     pred_velocity = tensor_to_np(model.velocity(x=xy, t=t))
    #     return scorer(true_velocity, pred_velocity)

    def sample_uniform_time(self, n_samples, t_subset="discrete"):
        # assert t_subset in ["discrete", "all", "t0", "discrete_single_t"]
        if t_subset == "discrete":
            data_t = self.sample_time(n_samples)  # np.random.rand(mb_size)
        elif t_subset == "discrete_single_t":
            data_t = np.broadcast_to(np.random.choice(self.timesteps, 1), n_samples)
        elif t_subset == "t0":
            data_t = np.zeros(n_samples)
        elif t_subset == "all":
            data_t = np.broadcast_to(np.random.rand(1) * self.max_t_test, n_samples)
        else:
            raise ValueError("Invalid Input")
        return data_t

    def sample_quasiuniform_space(self, n_samples, subset, device="cpu"):
        assert subset in ["train", "val", "test", "all"]
        raw_samples = self.sampler.draw(n_samples).to(device)
        # shuffle because quasi random samples are ordered
        idx = torch.randperm(raw_samples.shape[0])
        raw_samples = raw_samples[idx].view(raw_samples.size())

        if subset == "train":
            samples = self.uniform_to_train_space(raw_samples)
        elif subset == "val":
            samples = self.uniform_to_val_space(raw_samples)
        elif subset == "test":
            samples = self.uniform_to_test_space(raw_samples)
        else:
            samples = self.uniform_to_domain(raw_samples)
        return samples  # , t

    def uniform_to_domain(self, raw_samples):
        raw_samples[..., 0] = (raw_samples[..., 0] * 2 - 1) * self.lim
        raw_samples[..., 1] = (raw_samples[..., 1] * 2 - 1) * self.lim
        raw_samples[..., 2] = (raw_samples[..., 2] * 2 - 1) * self.lim
        return raw_samples

    def uniform_to_val_space(self, raw_samples):
        data_x_topleft = raw_samples[::2, ...] * self.lim  # [0, lim]
        data_x_topleft[..., 0] *= -0.2  # [-.2*lim, 0]
        data_x_topleft[..., -1] = data_x_topleft[..., -1] * 2 - self.lim  # [-lim, lim]

        data_x_bottomright = -raw_samples[1::2, ...] * self.lim  # [-lim, 0]
        data_x_bottomright[..., 0] *= -0.2
        data_x_bottomright[..., -1] = data_x_bottomright[..., -1] * 2 + self.lim  # [-lim, lim]

        samples = torch.concatenate([data_x_topleft, data_x_bottomright], 0)
        return samples

    def uniform_to_test_space(self, raw_samples):
        data_x_topleft = raw_samples[::2, ...] * self.lim
        data_x_topleft[..., 0] = -0.8 * data_x_topleft[..., 0] - 0.2 * self.lim
        data_x_topleft[..., -1] = data_x_topleft[..., -1] * 2 - self.lim  # [-lim, lim]

        data_x_bottomright = -raw_samples[1::2, ...] * self.lim
        data_x_bottomright[..., 0] = -0.8 * data_x_bottomright[..., 0] + 0.2 * self.lim
        data_x_bottomright[..., -1] = data_x_bottomright[..., -1] * 2 + self.lim  # [-lim, lim]

        samples = torch.concatenate([data_x_topleft, data_x_bottomright], 0)
        return samples

    def uniform_to_train_space(self, raw_samples):
        data_x_topright = raw_samples[::2, ...] * self.lim  # [0, lim]
        data_x_topright[..., -1] = data_x_topright[..., -1] * 2 - self.lim  # [-lim, lim]

        data_x_bottomleft = -raw_samples[1::2, ...] * self.lim
        data_x_bottomleft[..., -1] = data_x_bottomleft[..., -1] * 2 + self.lim

        samples = torch.concatenate([data_x_topright, data_x_bottomleft], 0)

        samples[..., -1] = samples[..., -1] * 0.2
        return samples

    def sample_quasiuniform_time(self, n_samples, t_max, device="cpu"):
        samples = self.sampler_time.draw(n_samples).to(device)
        t = samples * t_max
        return t

    def velocity(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:

        return torch.stack(self.dxyz_dt(x=x[..., 0].reshape(-1),
                                        y=x[..., 1].reshape(-1),
                                        z=x[..., 2].reshape(-1),
                                        time=t.reshape(-1)), -1)

    def log_density(self, x: Union[torch.Tensor, ArrayLike], t: Union[torch.Tensor, ArrayLike]) -> Union[
        torch.Tensor, ArrayLike]:
        return self.logp_at_t(position=x, time=t.reshape(-1))

    def xt_to_x0(self, x, t):
        x = np_to_tensor(x)
        t = np_to_tensor(t)
        x = 4 * torch.arctanh(x / 4)
        x /= (t.reshape(-1, 1) / 2 + 1)

        proj = self.get_projection_matrix(t).squeeze()
        # proj_inv = torch.linalg.inv(proj)
        # x = torch.arctanh(x / 2)
        # result = proj_inv @ (x - self.shift_dt * t)[..., None]
        x_shifted = (x)[..., None]
        result = torch.linalg.solve(proj, x_shifted).squeeze() - self.get_shift(t.reshape(-1, 1))
        return self.lim * torch.tanh(result / self.lim)

    def x0_to_xt(self, x0, t):
        x0 = np_to_tensor(x0)
        t = np_to_tensor(t)
        x_ = self.lim * torch.arctanh(x0 / self.lim)

        proj = self.get_projection_matrix(t).squeeze()
        result = (proj @ (x_ + self.get_shift(t.reshape(-1, 1)))[..., None]).squeeze()
        result *= (t.reshape(-1, 1) / 2 + 1)
        return self.lim * torch.tanh(result / self.lim)
        # return np.tanh(result) * 2

    def get_shift(self, t):
        return torch.sin(np.pi * t) * self.shift_dt.to(t)

    def get_projection_matrix(self, t):
        """
        Returns position of given particle at time t,
        :param t:
        :return:
        """
        proj = self.projection_matrix_rotate(t) @ self.projection_matrix_shear(t)
        return proj

    def projection_matrix_rotate(self, t):
        """
        Returns position of given particle at time t,
        :param t:
        :return:
        """
        w1, w2 = torch.cos(self.omega_rot * t), -torch.sin(self.omega_rot * t)
        w3, w4 = torch.sin(self.omega_rot * t), torch.cos(self.omega_rot * t)
        matrix = torch.stack([torch.stack([w1, w2], -1),
                              torch.stack([w3, w4], -1)], -2)

        return matrix

    def projection_matrix_shear(self, t):
        w1, w2, w3, w4 = (1 + t * self.omega_x), (t * self.omega_xy), torch.zeros_like(t), (1 + t * self.omega_y)
        matrix = torch.stack([torch.stack([w1, w2], -1), torch.stack([w3, w4], -1)], -2)
        return matrix

    def sample_time(self, batch_size):
        return np.random.choice(self.timesteps, batch_size)

    def get_mixture_center(self, mixture):
        angles = self.mixture_angles_radian[mixture]
        radiuses = self.radiuses_base[mixture]
        return np.concatenate([self.polar_to_cartesian(radiuses, angles), np.zeros(1)])

    @staticmethod
    def polar_to_cartesian(radiuses, angles):
        radiuses = np.array([radiuses]) if np.isscalar(radiuses) else radiuses
        angles = np.array([angles]) if np.isscalar(angles) else angles

        x = radiuses * np.cos(angles)
        y = radiuses * np.sin(angles)
        return np.stack([np.squeeze(x), np.squeeze(y)], -1)

    def logprob_mixture(self, mixture):
        return self.p_gauss_mixtures[mixture]

    def logprob_x_cond_on_mixture(self, positions, mixture):
        centers = self.get_mixture_center(mixture=mixture)
        log_probs = sp.stats.norm.logpdf(positions, centers, self.scale).sum(-1)
        return log_probs

    def logp_at_t0(self, position) -> numpy.typing.ArrayLike:
        # position = np.arctanh(position/4)*4
        assert position.shape[-1] == 3
        position_2d = position[..., :-1]
        mixturewise_logprobs = []
        for mixture in self.gauss_mixtures:
            mixturewise_logprobs.append(
                self.logprob_x_cond_on_mixture(position, mixture=mixture))
        mixturewise_logprobs = np.array(mixturewise_logprobs)
        mixture_logprobs = np.log(self.p_gauss_mixtures)[..., np.newaxis]
        logsumexp = sp.special.logsumexp(mixture_logprobs + mixturewise_logprobs, axis=0)
        return logsumexp

    def logp_at_t(self, position, time):
        # position_at_t0 = (projection_matrix_inv @ ((position - self.shift_dt * time.squeeze())[..., None])).squeeze()

        # 2d only
        position_xy = position[..., :-1]
        with torch.enable_grad():
            position_xy = np_to_tensor(position_xy).requires_grad_(True)
            position_xy_at_t0 = self.xt_to_x0(position_xy, time.squeeze())
            position_z_at_t0 = np_to_tensor(position[..., -1]).unsqueeze(-1)
            position_xyz_at_t = torch.concatenate([position_xy_at_t0, position_z_at_t0], -1)

            logp_at_t0 = self.logp_at_t0(position_xyz_at_t.detach().cpu().numpy()) + np.log(
                self.scale[-1] * np.sqrt(2 * np.pi))
            jac = batch_jacobian(position_xy_at_t0, position_xy)
            logabsdet = - torch.slogdet(jac)[1]
        # logabsdet = np.log(1+time*self.omega_x) + np.log(1+time*self.omega_y)
        # projection_matrix = self.get_projection_matrix(cast_to_tensor(time)).squeeze().numpy()
        # logabsdet_ref = np.linalg.slogdet(projection_matrix)[1]
        return logp_at_t0 - logabsdet.detach().cpu().numpy()

    def dxyz_dt(self, x, y, z, time=None):
        with torch.enable_grad():
            x = np_to_tensor(x)
            y = np_to_tensor(y)
            time = np_to_tensor(time).requires_grad_(True)
            xy = torch.stack([x, y], -1)

            xy_0 = self.xt_to_x0(xy, t=time).detach().requires_grad_(True)
            xy_t = self.x0_to_xt(xy_0, t=time)
            vel_x = torchutils.gradient(xy_t[..., 0], time)
            vel_y = torchutils.gradient(xy_t[..., 1], time)
        return vel_x.detach(), vel_y.detach(), torch.zeros_like(vel_x)

    def sample_data(self, n_samples, subset="train", t_subset="discrete"):
        assert subset in ["train", "val", "test", "all"]
        assert t_subset in ["discrete", "all", "t0", "discrete_single_t"]

        # data_x[:, 0] = np.abs(data_x[:, 0])
        data_x = self.sample_quasiuniform_space(n_samples=n_samples, subset=subset, device="cpu").numpy()
        data_t = self.sample_uniform_time(n_samples=n_samples, t_subset=t_subset)

        vel = np.stack(self.dxyz_dt(*data_x.T, data_t), -1)
        vel = vel + 0.1 * np.random.randn(*vel.shape) * 0.14  # ~10% std noise
        logpdf = self.logp_at_t(data_x, time=data_t)
        logpdf = logpdf + 0.1 * np.random.randn(*logpdf.shape) * 0.23  # ~10% std noise
        return data_x, data_t, logpdf, vel


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_gen = MovingGaussians3d()

    data_x = dict()
    data_t = dict()
    data_lnrho = dict()
    data_vel = dict()

    for subset in ["train", "val", "test", "all"]:
        data_x[subset], data_t[subset], data_lnrho[subset], data_vel[subset] = data_gen.sample_data(50_000,
                                                                                                    subset=subset,
                                                                                                    t_subset="discrete_single_t")
    # data_x_train, data_t, a, b = data_gen.sample_data(1_000, subset="train")
    # data_x_val, data_t, a, b = data_gen.sample_data(1_000, subset="val")
    # data_x_test, data_t, a, b = data_gen.sample_data(1_000, subset="test")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    t_unique = np.unique(data_t["all"])
    for subset in ["all"]:  # ["train", "val", "test"]:
        idx = (data_lnrho[subset] > -5)
        ax.scatter(*data_x[subset][idx].T, c=(data_lnrho[subset][idx]), marker=".", alpha=0.9)
    # ax.scatter(*data_x_val.T, marker=".")
    # ax.scatter(*data_x_test.T, marker=".")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(-data_gen.lim, data_gen.lim)
    ax.set_ylim(-data_gen.lim, data_gen.lim)
    ax.set_zlim(-data_gen.lim, data_gen.lim)

    plt.show()

    subsets = ["train", "val", "test", "all"]
    for subset in subsets:
        samples = torchutils.tensor_to_np(data_gen.sample_quasiuniform_space(n_samples=1000, subset=subset))[..., :-1]
        plt.scatter(*samples.T, label=subset, marker='+')
    plt.legend()
    plt.show()


def eval_model(data_gen, model, device="cuda", verbose=False, split_size=2_000):
    model.eval()
    consistency_loss = model.evaluate_consistency_loss(data_gen, ode_split_size=split_size)
    print(f'Consistency Loss: {consistency_loss:.2e}')
    val_score = data_gen.score_density(model, subset="val", device=device, n_space=10_000, n_times=20, verbose=verbose,
                                       split_size=split_size)
    test_score = data_gen.score_density(model, subset="test", device=device, n_space=10_000, n_times=100,
                                        verbose=verbose, split_size=split_size)

    print(f'final Val: {val_score:.2f}')
    print(f'final Test: {test_score:.2f}')
    model.train()
    return val_score, test_score, consistency_loss

