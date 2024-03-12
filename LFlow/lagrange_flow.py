from typing import Callable
import torch
from torch.nn import functional
from enflows.flows.base import Flow
from enflows.utils import torchutils
from enflows.nn.nets import ResidualNet


def transformed_mse(input, target, transform=functional.softplus, *args, **kwargs):
    return functional.mse_loss(transform(input), transform(target), *args, **kwargs)

def jacobian_scalar_input(input, output):
    return torch.stack([torchutils.gradient(output[..., i], input) for i in range(output.shape[-1])], -1)


class LagrangeFlow(Flow):
    def __init__(self, transform, distribution, embedding_net, init_log_scale_norm: float = 0.,
                 flexible_norm=False):
        assert embedding_net is not None, "embedding_net must be provided, as this is a conditional flow."
        super().__init__(transform=transform, distribution=distribution, embedding_net=embedding_net)

        self.init_log_scale_norm = init_log_scale_norm
        self.features = self._distribution._shape[0]
        self.basis_vectors = torch.nn.Parameter(torch.eye(self.features), requires_grad=False)
        self.const = torch.nn.Parameter(torch.ones(1,1))
        self.flexible_norm = flexible_norm
        if flexible_norm:
            self.norm_const_net = ResidualNet(in_features=1, out_features=1, num_blocks=1, hidden_features=16)
        else:
            self._trainable_log_scale = torch.nn.Parameter(
                torch.tensor(init_log_scale_norm, dtype=torch.float32, requires_grad=True))

        self.max_time = 1

    @property
    def trainable_log_scale(self):
        if self.flexible_norm:
            return self.norm_const_net(self.const).squeeze() + self.init_log_scale_norm
        else:
            return self._trainable_log_scale


    def log_density(self, x, t):
        return self.log_prob(x, t) + self.trainable_log_scale

    def velocity(self, x, t):
        """Calculate velocity with respect to context.

        Args:
            x: Tensor, input variables.
            t: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size, input_dim], the change of the position of a particle with respect to the  context.
        """
        x = torch.as_tensor(x)  # .clone()
        t = torch.as_tensor(t)  # .clone()
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                "Number of input items must be equal to number of context items."
            )
        return self._velocity_jacobian_solve(x, t)
        # return self._velocity_finite_differences(x, t)

    def log_density_and_velocity(self, x, t):
        """Calculate velocity with respect to context.

        Args:
            x: Tensor, input variables.
            t: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size, input_dim], the change of the position of a particle with respect to the  context.
        """
        x = torch.as_tensor(x)  # .clone()
        t = torch.as_tensor(t)  # .clone()
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                "Number of input items must be equal to number of context items."
            )
        return self._log_density_and_velocity(x, t)

    def _dz_dt(self, x, t):
        _t = t.requires_grad_(True)
        embedded_context = self._embedding_net(_t)
        z, logabsdet = self._transform(x, context=embedded_context)
        dz_dt_jac = torch.stack([torchutils.gradient(z[..., i], _t) for i in range(z.shape[-1])], -2)
        return dz_dt_jac

    def x_to_z(self, x, t):
        embedded_context = self._embedding_net(t)
        z, logabsdet = self._transform(x, context=embedded_context)
        return z

    def __x_to_z_sum(self, x, t):
        embedded_context = self._embedding_net(t)
        z, logabsdet = self._transform(x, context=embedded_context)
        return z.sum(0)

    def z_to_x(self, x, t):
        embedded_context = self._embedding_net(t)
        z, logabsdet = self._transform.inverse(x, context=embedded_context)
        return z

    def _velocity_jacobian_solve(self, x: torch.Tensor, t: torch.Tensor, eps=1e-6):
            with torch.enable_grad():
                dz_dx_jac, dz_dt_jac = (jac.movedim(1, 0) for jac in
                                        torch.autograd.functional.jacobian(self.__x_to_z_sum, (x, t), create_graph=True))

                # eye = torch.eye(dz_dx_jac.shape[-1])[None, ...].to(dz_dx_jac.device)  # .repeat(dz_dx_jac.shape[0], 0)
                dxhat_dt = -torch.linalg.solve(dz_dx_jac + eps * self.basis_vectors.unsqueeze(0), dz_dt_jac).squeeze()
                return dxhat_dt

    def _velocity_finite_differences(self, x: torch.Tensor, t: torch.Tensor, eps=1e-3,
                                     type="forward"):

        embedded_context = self._embedding_net(t)
        z, _ = self._transform(x, context=embedded_context)
        z = z.detach()
        if type == "central":
            eps /= 2
            x_tplusdt, _ = self._transform.inverse(z,
                                                   context=self._embedding_net(t.detach() + eps))
            x_tmindt, _ = self._transform.inverse(z,
                                                  context=self._embedding_net(t.detach() - eps))
            dxhat_dt = (x_tplusdt - x_tmindt) / (2 * eps)
        elif type == "forward":
            x_tplusdt, _ = self._transform.inverse(z,
                                                   context=self._embedding_net(t.detach() + eps))
            dxhat_dt = (x_tplusdt - x) / eps
        elif type == "backward":
            x_tmindt, _ = self._transform.inverse(z,
                                                  context=self._embedding_net(t.detach() - eps))
            dxhat_dt = (x - x_tmindt) / eps
        else:
            raise ValueError("Unknown type of finite difference.")

        return dxhat_dt

    def _log_density_and_velocity(self, x, t):
        with torch.enable_grad():
            _t = t.requires_grad_(True)
            x = x.requires_grad_(True)

            embedded_context = self._embedding_net(_t)
            z, logabsdet = self._transform(x, context=embedded_context)
            if self._context_used_in_base:
                log_prob = self._distribution.log_prob(z, context=embedded_context)
            else:
                log_prob = self._distribution.log_prob(z)

            dz_dx_jac = torch.stack([torchutils.gradient(z[..., i], x) for i in range(z.shape[-1])], -2)
            dz_dt_jac = torch.stack([torchutils.gradient(z[..., i], t) for i in range(z.shape[-1])], -2)

            # eye = torch.eye(dz_dx_jac.shape[-1])[None, ...].to(dz_dx_jac.device)  # .repeat(dz_dx_jac.shape[0], 0)
            dxhat_dt = -torch.linalg.solve(dz_dx_jac + 1e-8 * self.basis_vectors, dz_dt_jac).squeeze()
            # dxhat_dt = -torch.linalg.solve(dz_dx_jac, dz_dt_jac).squeeze()

            return log_prob + logabsdet + self.trainable_log_scale, dxhat_dt

    def sample_and_velocity(self, samples_per_context, t):
        """Generates samples from the flow, together with their velocities.

        For flows, this is more efficient that calling `sample` and `velocity` separately.
        """
        with torch.enable_grad():
            t = t.detach().requires_grad_(True)

            embedded_context = self._embedding_net(t)
            if self._context_used_in_base:
                z, log_prob = self._distribution.sample_and_log_prob(
                    samples_per_context, context=embedded_context
                )
            else:
                z, log_prob = self._distribution.sample_and_log_prob(
                    samples_per_context
                )

            # Merge the context dimension with sample dimension in order to apply the transform.
            z = torchutils.merge_leading_dims(z, num_dims=2)
            embedded_context = torchutils.repeat_rows(embedded_context, num_reps=samples_per_context)

            # noise = noise.squeeze().detach()

            x_hat, logabsdet = self._transform.inverse(z, context=embedded_context)
            dx_dt = jacobian_scalar_input(output=x_hat, input=t).squeeze()
            velocity = dx_dt
            # dx_dt_ = self.velocity(samples.detach(), context.detach())
            # diff = (dx_dt_ - dx_dt).abs()
            t.requires_grad_(False)

            return x_hat, velocity

    def sample_and_log_density(self, samples_per_context, t):
        """Generates samples from the flow, together with their velocities.

        For flows, this is more efficient that calling `sample` and `velocity` separately.
        """

        sample, log_prob = self.sample_and_log_prob(num_samples=samples_per_context, context=t)
        return sample, log_prob + self.trainable_log_scale

    def global_density_penalty(self, factor=1., transform: Callable = functional.softplus):
        if transform is None:
            return factor * self.trainable_log_scale
        else:
            return factor * transform(self.trainable_log_scale)

    def transport_cost_penalty(self, n_z_samples=64, n_t_samples=20, forward_mode=False):
        # z_bg = self._distribution.sample(num_samples)

        # z_bg = torch.randn(num_samples, 2, device=self.trainable_log_scale.device)
        self.eval()
        # t_bg = torch.rand(num_samples, 1, device=self.trainable_log_scale.device)

        if forward_mode:
            with torch.no_grad():
                t_bg = torch.rand(n_t_samples, 1, device=self.trainable_log_scale.device) * self.max_time
                # z_bg = self._distribution.sample(n_z_samples)
                z_bg = torch.randn((n_z_samples, self.features), device=self.trainable_log_scale.device)
                z_bg_rep = z_bg.repeat_interleave(t_bg.shape[0], 0)
                t_bg_tile = t_bg.tile(z_bg.shape[0], 1)

            x_bg, _ = self._transform.inverse(z_bg_rep, context=self._embedding_net(t_bg_tile))
            vel_bg = self.velocity(x_bg.detach(), t=t_bg_tile)
        else:
            t_bg = torch.rand(n_t_samples * n_z_samples, 1, device=self.trainable_log_scale.device) * self.max_time
            x_bg, vel_bg = self.sample_and_velocity(1, t_bg)
        self.train()

        # x_bg, _ = self._transform.inverse(z_bg, context=self._embedding_net(t_bg))
        # vel = self.velocity(x_bg.detach(), context=t_bg)

        speed_penalty = (vel_bg ** 2).sum(-1).mean()
        return speed_penalty



class ConditionedLagrangeFlow(Flow):
    def __init__(self, transform, distribution, embedding_net, log_scale_network:torch.nn.Module, log_scale_init = 18,
                 max_night = 4):
        assert embedding_net is not None, "embedding_net must be provided, as this is a conditional flow."
        super().__init__(transform=transform, distribution=distribution, embedding_net=embedding_net)

        self.log_scale_network = log_scale_network
        self.log_scale_factor = log_scale_init

        self.features = self._distribution._shape[0]
        self.basis_vectors = torch.nn.Parameter(torch.eye(self.features), requires_grad=False)
        self.max_night = max_night

    def log_scale_constant(self, c):
        return self.log_scale_factor + self.log_scale_network(c).squeeze()

    def log_density(self, x, t, c):

        x = torch.as_tensor(x)  # .clone()
        t = torch.as_tensor(t)  # .clone()
        c = torch.as_tensor(c)  # .clone()

        context = self.concat_tc(t, c)
        return self.log_prob(x, context).squeeze() + self.log_scale_constant(c)

    def concat_tc(self, t, c):
        return torch.concatenate((t, c/self.max_night), axis=-1)


    def velocity(self, x, t, c):
        """Calculate velocity with respect to context.

        Args:
            x: Tensor, input variables.
            t: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size, input_dim], the change of the position of a particle with respect to the  context.
        """
        x = torch.as_tensor(x)  # .clone()
        t = torch.as_tensor(t)  # .clone()
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                "Number of input items must be equal to number of context items."
            )
        return self._velocity_jacobian_solve(x, t, c)
        # return self._velocity_finite_differences(x, t)

    def log_density_and_velocity(self, x, t, c):
        """Calculate velocity with respect to context.

        Args:
            x: Tensor, input variables.
            t: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.

        Returns:
            A Tensor of shape [input_size, input_dim], the change of the position of a particle with respect to the  context.
        """
        x = torch.as_tensor(x)  # .clone()
        t = torch.as_tensor(t)  # .clone()
        c = torch.as_tensor(c)  # .clone()
        if x.shape[0] != t.shape[0]:
            raise ValueError(
                "Number of input items must be equal to number of context items."
            )
        return self._log_density_and_velocity(x, t, c)

    def _dz_dt(self, x, t, c):
        _t = t.requires_grad_(True)
        context = self.concat_tc(_t, c)
        embedded_context = self._embedding_net(context)
        z, logabsdet = self._transform(x, context=embedded_context)
        dz_dt_jac = torch.stack([torchutils.gradient(z[..., i], _t) for i in range(z.shape[-1])], -2)
        return dz_dt_jac

    def x_to_z(self, x, t, c):
        context = self.concat_tc(t, c)
        embedded_context = self._embedding_net(context)
        z, logabsdet = self._transform(x, context=embedded_context)
        return z

    def __x_to_z_sum(self, x, t, c):
        context = self.concat_tc(t, c)
        embedded_context = self._embedding_net(context)
        z, logabsdet = self._transform(x, context=embedded_context)
        return z.sum(0)

    def z_to_x(self, x, t, c):
        context = self.concat_tc(t, c)
        embedded_context = self._embedding_net(context)
        z, logabsdet = self._transform.inverse(x, context=embedded_context)
        return z

    def _velocity_jacobian_solve(self, x: torch.Tensor, t: torch.Tensor, c:torch.Tensor, eps=1e-6):
            with torch.enable_grad():
                # tmp = torch.autograd.functional.jacobian(self.__x_to_z_sum, (x, t, c), create_graph=True)
                # dz_dx_jac, dz_dt_jac, dz_dc_jac = [jac.movedim() for jac in tmp]
                dz_dx_jac, dz_dt_jac, dz_dc_jac = (jac.movedim(1, 0) for jac in
                                        torch.autograd.functional.jacobian(self.__x_to_z_sum, (x, t, c), create_graph=True))
                # eye = torch.eye(dz_dx_jac.shape[-1])[None, ...].to(dz_dx_jac.device)  # .repeat(dz_dx_jac.shape[0], 0)
                dxhat_dt = -torch.linalg.solve(dz_dx_jac + eps * self.basis_vectors.unsqueeze(0), dz_dt_jac).squeeze()
                return dxhat_dt


    def _log_density_and_velocity(self, x, t, c):
        with torch.enable_grad():
            _t = t.requires_grad_(True)
            x = x.requires_grad_(True)
            context = self.concat_tc(_t, c)
            embedded_context = self._embedding_net(context)
            z, logabsdet = self._transform(x, context=embedded_context)
            if self._context_used_in_base:
                log_prob = self._distribution.log_prob(z, context=embedded_context)
            else:
                log_prob = self._distribution.log_prob(z)

            dz_dx_jac = torch.stack([torchutils.gradient(z[..., i], x) for i in range(z.shape[-1])], -2)
            dz_dt_jac = torch.stack([torchutils.gradient(z[..., i], t) for i in range(z.shape[-1])], -2)

            # eye = torch.eye(dz_dx_jac.shape[-1])[None, ...].to(dz_dx_jac.device)  # .repeat(dz_dx_jac.shape[0], 0)
            dxhat_dt = -torch.linalg.solve(dz_dx_jac + 1e-8 * self.basis_vectors, dz_dt_jac).squeeze()
            # dxhat_dt = -torch.linalg.solve(dz_dx_jac, dz_dt_jac).squeeze()

            return log_prob + logabsdet + self.log_scale_constant(c), dxhat_dt

    def global_density_penalty(self, c, factor=1., transform: Callable = functional.softplus):
        if transform is None:
            return factor * self.log_scale_network(c).mean()
        else:
            return factor * transform(self.log_scale_constant(c)).mean()



