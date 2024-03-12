
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from enflows.CNF.cnf import divergence_approx, divergence_bf, sample_gaussian_like, sample_rademacher_like, _flip


class OTCompactTimeVariableODESolver(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(self, dynamics_network, solver='dopri5', atol=1e-5, rtol=1e-5,
                 divergence_fn="approximate"):
        super(OTCompactTimeVariableODESolver, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        nreg = 0

        self.diffeq = dynamics_network
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.rademacher = True

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self.register_buffer("_num_evals", torch.tensor(0.))
        self.before_odeint()

        self.odeint_kwargs = dict(
            train=dict(
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
                # adjoint_options={"norm": "seminorm"}
                ),
            test=dict(
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                # adjoint_options={"norm": "seminorm"}
                )
        )

    def integrate(self, t0, t1, z, logpz=None):

        _logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        _penalty = torch.zeros(z.shape[0], 1).to(z)
        initial_state = (t0, t1, z, _logpz, _penalty)

        integration_times = torch.tensor([self.start_time, self.end_time]).to(t0)

        # Refresh the odefunc statistics.
        self.before_odeint(e=self.sample_e_like(z))

        self.get_odeint_kwargs()
        state_t = odeint(
            func=self,
            y0=initial_state,
            t=integration_times,
            **self.get_odeint_kwargs()
        )
        _, _,  z_t, logpz_t, penalty = tuple(s[-1] for s in state_t)

        return z_t, logpz_t, penalty

    def forward(self, s, states):
        assert len(states) >= 2
        t0, t1, y, _, _ = states
        ratio = (t1 - t0) / (self.end_time - self.start_time)

        # increment num evals
        self._num_evals += 1

        # Sample and fix the noise.
        if self._e is None:
            self._e = self.sample_e_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t = (s - self.start_time) * ratio + t0
            dy = self.diffeq(t, y)
            dy = dy * ratio.reshape(-1, *([1] * (y.ndim - 1)))

            divergence = self.calculate_divergence(y, dy)

        dpenalty = 0.5*torch.linalg.norm(dy, dim=-1, keepdim=True)
        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), dy, -divergence, dpenalty])

    def sample_e_like(self, y):
        if self.rademacher:
            return sample_rademacher_like(y)
        else:
            return sample_gaussian_like(y)

    def calculate_divergence(self, y, dy):
        # Hack for 2D data to use brute force divergence computation.
        if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
            divergence = divergence_bf(dy, y).view(-1, 1)
        else:
            if self.training:
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
            else:
                divergence = divergence_bf(dy, y, e=self._e).view(-1, 1)
        return divergence

    def get_odeint_kwargs(self):
        if self.training:
            return self.odeint_kwargs["train"]
        else:
            return self.odeint_kwargs["test"]

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()




