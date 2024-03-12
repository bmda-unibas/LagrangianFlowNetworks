import abc

import torch
import torch.nn as nn

from enflows.CNF.cnf import CompactTimeVariableCNF
from LFlow.ot_ode_layer import OTCompactTimeVariableODESolver



class ScalarGridLayer(nn.Module):
    def __init__(self, grid_len=30, dim=2, init_noise=None, padding_mode="border"):
        super().__init__()
        self.grid_len = grid_len
        self.dim = dim

        # xx = torch.linspace(-1, 1, grid_len, dtype=torch.get_default_dtype())
        # yy = torch.linspace(-1, 1, grid_len, dtype=torch.get_default_dtype())
        # X, Y = torch.meshgrid(xx, yy)
        # XY = torch.stack([X, Y], -1)
        self.padding_mode = padding_mode
        if self.dim == 2:
            noise = init_noise if init_noise is not None else 1e-4
            XY = torch.zeros((grid_len, grid_len, 1), dtype=torch.get_default_dtype())
            self._base_grid = torch.nn.Parameter(torch.randn_like(XY) * noise, requires_grad=True)
        else:
            noise = init_noise if init_noise is not None else 1e-2
            XYZ = torch.zeros((grid_len, grid_len, grid_len, 1), dtype=torch.get_default_dtype())
            self._base_grid = torch.nn.Parameter(torch.randn_like(XYZ) * noise, requires_grad=True)

    @property
    def base_grid(self):
        return self._base_grid

    @staticmethod
    def bilinear_interpolate_torch_gridsample(image, samples_grid, dim, padding_mode="border"):
        if dim == 2:
            # input image is: W x H x C
            image = image.permute(2, 0, 1)  # change to:      C x W x H
            image = image.unsqueeze(0)  # change to:  1 x C x W x H

            # samples_grid = samples_grid / (2 * 4)  # *2-1                       # normalize to between -1 and 1
            return torch.nn.functional.grid_sample(image, samples_grid.view(1, 1, -1, 2), align_corners=False,
                                                   padding_mode=padding_mode)
        elif dim == 3:
            # input image is: W x H x D x C
            image = image.permute(3, 0, 1, 2)  # change to:      C x W x H x D
            image = image.unsqueeze(0)  # change to:  1 x C x W x H x D
            # samples_grid = samples_grid / (2 * 4)  # *2-1                       # normalize to between -1 and 1
            return torch.nn.functional.grid_sample(image, samples_grid.view(1, 1, 1, -1, 3), align_corners=False,
                                                   padding_mode=padding_mode)

    def forward(self, x, validate_inputs=False):
        if validate_inputs:
            assert torch.all((x <= 1) or (x >= -1))
        tmp = self.bilinear_interpolate_torch_gridsample(self.base_grid, x, dim=self.dim,
                                                         padding_mode=self.padding_mode).squeeze()
        return tmp


class SemiLagrangianFlow(nn.Module):
    def __init__(self, odenet_flow, dim, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5,
                 divergence_fn="approximate", grid_len=100, init_noise=None):
        super().__init__()
        assert divergence_fn in ["brute_force", "approximate"]

        self.cnf_to_t0 = CompactTimeVariableCNF(odenet_flow, solver=solver,
                                                atol=atol,
                                                divergence_fn=divergence_fn,
                                                rtol=rtol)
        self.t0 = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.base_grid = ScalarGridLayer(grid_len, dim=dim, init_noise=init_noise)

    @abc.abstractmethod
    def transform_to_basegrid_range(self, x):
        pass

    def log_density(self, x, t, tol=None, method=None):
        divergence_accum_tbase, z_base = self.x_to_z(x, t, tol=tol, method=method)
        # z_unit_cube, logabsdet = self.tanh_to_unit_cube(z_base)
        logp_x = (self.base_grid(self.transform_to_basegrid_range(z_base))).squeeze() - divergence_accum_tbase.view(
            -1)  # + logabsdet.view(-1)
        return logp_x

    def x_to_z(self, x, t, tol=None, method=None):
        divergence_accum_t = torch.zeros(x.shape[0], 1).type(torch.float32).to(x)
        t_zero = torch.zeros_like(t)
        z_t0 = torch.zeros_like(x)
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(x)
        divergence_accum_t0 = torch.zeros_like(divergence_accum_t)
        idx = (t > 0).squeeze()
        if idx.sum() > 0:
            tmp_z_t0, tmp_divergence_accum = self.cnf_to_t0.integrate(t0=t[idx], t1=t_zero[idx], z=x[idx],
                                                                      logpz=divergence_accum_t[idx])
            z_t0[idx] = tmp_z_t0
            divergence_accum_t0[idx] = tmp_divergence_accum
        z_t0[~idx] = x[~idx]
        return divergence_accum_t0, z_t0

    def z_to_x(self, z_b, t):
        raise NotImplementedError("..")

    def velocity(self, x, t):
        return self.cnf_to_t0.diffeq(t, x)

    def set_rtol(self, rtol):
        self.cnf_to_t0.rtol = rtol
        self.cnf_to_t0.test_rtol = rtol
        self.cnf_to_base.test_rtol = rtol
        self.cnf_to_base.rtol = rtol

    def set_atol(self, atol):
        self.cnf_to_t0.test_atol = atol
        self.cnf_to_base.test_atol = atol
        self.cnf_to_t0.atol = atol
        self.cnf_to_base.atol = atol

    def set_solver(self, solver):
        self.cnf_to_t0.solver = solver
        self.cnf_to_base.solver = solver


class OT_SLDA(nn.Module):
    def __init__(self, odenet_flow, dim, T=1.0, solver='dopri5', atol=1e-5, rtol=1e-5,
                 divergence_fn="approximate", init_log_scale_norm=0, loc=0, scale=1, flexible_norm=True):
        super().__init__()
        assert divergence_fn in ["brute_force", "approximate"]

        self.cnf_to_t0 = OTCompactTimeVariableODESolver(odenet_flow, solver=solver, atol=atol,
                                                        divergence_fn=divergence_fn,
                                                        rtol=rtol)

        self.base_grid = ScalarGridLayer(100, dim=dim, init_noise=1e-6, padding_mode="zeros")

        # self.trainable_log_scale = torch.nn.Parameter(
        #     torch.tensor(init_log_scale_norm, dtype=torch.float32, requires_grad=True))
        self.t0 = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # self.act_norm = ActNorm(dim)

    def transform_to_basegrid_range(self, x):
        return x / 5

    def log_density(self, x, t, get_penalty=False):
        divergence_accum_tbase, z_base, penalty = self.x_to_z(x, t)
        # z_unit_cube, logabsdet = self.tanh_to_unit_cube(z_base)
        logp_x = (self.base_grid(self.transform_to_basegrid_range(z_base))).squeeze() - divergence_accum_tbase.view(
            -1)  # + logabsdet.view(-1)
        if get_penalty:
            return logp_x, penalty
        else:
            return logp_x

    def x_to_z(self, x, t):
        divergence_accum_t = torch.zeros(x.shape[0], 1).type(torch.float32).to(x)
        t_zero = torch.zeros_like(t)
        z_t0 = torch.zeros_like(x)

        if not torch.is_tensor(t):
            t = torch.tensor(t).to(x)
        divergence_accum_t0 = torch.zeros_like(divergence_accum_t)
        penalty = torch.zeros_like(divergence_accum_t)
        idx = (t > 0).squeeze()
        if idx.sum() > 0:
            tmp_z_t0, tmp_divergence_accum, tmp_penalty = self.cnf_to_t0.integrate(t0=t[idx], t1=t_zero[idx], z=x[idx],
                                                                                   logpz=divergence_accum_t[idx])
            z_t0[idx] = tmp_z_t0
            divergence_accum_t0[idx] = tmp_divergence_accum
            penalty[idx] = tmp_penalty
        z_t0[~idx] = x[~idx]
        #

        return divergence_accum_t0, z_t0, penalty

    def z_to_x(self, z_b, t):
        divergence_accum_t = torch.zeros(z_b.shape[0], 1).type(torch.float32).to(z_b)
        t_zero = torch.zeros_like(t)
        z_t0 = torch.zeros_like(z_b)
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(z_b)

        divergence_accum_t0 = torch.zeros_like(divergence_accum_t)

        z_t, divergence_accum_t, penalty0 = self.cnf_to_t0.integrate(t0=t_zero, t1=t, z=z_t0,
                                                                     logpz=divergence_accum_t0)
        return divergence_accum_t, z_t, penalty0

    def velocity(self, x, t):
        return self.cnf_to_t0.diffeq(t, x)


