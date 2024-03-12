"""
A very basic application of the LFlows.

Interpolation of the two half moons of the two-moons data set.
Here, we train with a maximum likelihood objective and also visualize the velocity.

"""
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from enflows.transforms.autoregressive import MaskedDeepSigmoidTransform, MaskedSumOfSigmoidsTransform
from torch import nn
from torch import optim

# from enflows.flows import Flow
from LFlow import LagrangeFlow as Flow
from enflows.distributions.normal import DiagonalNormal
from enflows.transforms.conditional import *
from enflows.transforms import *
from enflows.nn.nets import *
from enflows.transforms.permutations import ReversePermutation
from enflows.nn.nets import ResidualNet
from enflows.utils.torchutils import tensor_to_np, np_to_tensor
from data import GaussianMM, load_2dtarget
import logging
import argparse
from pprint import pformat
import time
import os
import pickle
import random

parser = argparse.ArgumentParser()
parser.add_argument('--target0', type=str, default="circles")
parser.add_argument('--target1', type=str, default="pinwheel")
parser.add_argument('--weight_denom', type=float, default=700)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--to_console', action="store_true")
args = parser.parse_args()
num_iter = 5_000

dirname = f"results/{args.target0}_{args.target1}_{args.weight_denom}_{args.seed}"
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

print(dirname)
os.makedirs(dirname, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(message)s',
                    # datefmt='%Y-%m-%d,%H:%M:%S',
                    filename=f"{dirname}/log.txt" if not args.to_console else None,
                    filemode='w' if not args.to_console else None,
                    )

logging.info(pformat(vars(args)))

log = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
num_layers = 10
base_dist = DiagonalNormal(shape=[2])
context_features = 10

target0 = load_2dtarget(args.target0).to(device)
target1 = load_2dtarget(args.target1).to(device)
x0_fixed = target0.sample(10_000).float().to(device)
x1_fixed = target1.sample(10_000).float().to(device)

lim = 2
mb_size = 256 * 8


def plot_model(flow):
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    ax = ax.flatten()
    xline = torch.linspace(-lim, lim, 100, device=device)
    yline = torch.linspace(-lim, lim, 100, device=device)
    xgrid, ygrid = torch.meshgrid(xline, yline, indexing="xy")
    xgrid_vel, ygrid_vel = torch.meshgrid(xline[::5], yline[::5], indexing="xy")
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    xyinput_vel = torch.cat([xgrid_vel.reshape(-1, 1), ygrid_vel.reshape(-1, 1)], dim=1)
    flow.eval()
    with torch.no_grad():
        ones = torch.ones(10000, 1, device=device)
        for k, factor in enumerate([0, 0.25, 0.5, 0.75, 1.]):
            zgrid = flow.log_prob(xyinput, factor * ones).exp().reshape(100, 100)
            ax[k].imshow(zgrid.cpu().detach().numpy(), extent=(-2, 2, -2, 2), origin='lower',
                         cmap="viridis")

            ax[k].get_xaxis().set_ticks([])
            ax[k].get_yaxis().set_ticks([])
            # ax[k].contourf(xgrid.cpu().detach().numpy(), ygrid.cpu().detach().numpy(), zgrid.cpu().detach().numpy(),
            #                cmap="Blues")
            # ax[i].quiver(xgrid_vel.numpy(), ygrid_vel.numpy(), vel[:, 0], vel[:, 1], angles='xy', scale_units='xy')
            # ax[i].axis('equal')
            # ax[k].set_ylim(-1.5, 1.5)


def build_model():

    densenet_builder = LipschitzDenseNetBuilder(input_channels=2,
                                                densenet_depth=5,
                                                densenet_growth=32,
                                                activation_function=CSin(15),
                                                lip_coeff=.97,
                                                context_features=context_features
                                                )
    transforms = []
    for i in range(num_layers):
        transforms.append(ActNorm(2))
        transforms.append(iResBlock(densenet_builder.build_network(),
                                    brute_force=True))
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=128,
                                num_blocks=1,
                                activation=torch.nn.functional.silu)
    flow = Flow(transform, base_dist, init_log_scale_norm=0, embedding_net=embedding_net,
                flexible_norm=False).to(device)
    flow.trainable_log_scale.requires_grad_(False)
    return flow


def gen_samples(mb_size):
    # Sample from target distributions.
    x0 = target0.sample(mb_size)
    x1 = target1.sample(mb_size)

    # Include random samples.
    xu = (
            torch.rand(mb_size, 2).to(x0) * 2 * lim - lim
    )
    x0 = torch.cat([x0, xu], dim=0)
    x1 = torch.cat([x1, xu], dim=0)

    y0 = torch.zeros(x0.shape[0], 1)
    y1 = torch.ones(x1.shape[0], 1)
    y = torch.cat([y0, y1], dim=0)

    p0 = target0.logprob(x0)
    p1 = target1.logprob(x1)
    return_vals = torch.cat([x0, x1], 0), torch.cat([p0, p1], 0).reshape(-1), y.reshape(-1, 1)

    return [val.float().to(device) for val in return_vals]


def gen_ot_samples(mb_size):
    x0 = target0.sample(mb_size)
    x1 = target1.sample(mb_size)
    xu = (
            torch.rand(mb_size, 2).to(x0) * 2 * lim - lim
    )

    p0 = torch.exp(target0.logprob(x0))
    p1 = torch.exp(target1.logprob(x1))
    pu = torch.tensor(1 / (2 * lim) ** 2).expand(xu.shape[0]).to(p0)

    x = torch.cat([x0, x1, xu], dim=0)
    y = torch.rand(x.shape[0], 1).to(x)
    ps = torch.cat([p0, p1, pu], dim=0)

    return_vals = x, ps, y.reshape(-1, 1)

    return [val.float().to(device) for val in return_vals]


@torch.no_grad()
def xfixed_at_times(flow: Flow, steps=5, sample_from_z=False):
    # Sample from p0
    # x0 = x0_fixed

    if sample_from_z:
        z = torch.randn((5_000, 2), dtype=torch.float32).to(device)
        ones = z.new_ones((z.shape[0], 1))
    else:
        x0 = target0.sample(5_000).float().to(device)
        ones = x0.new_ones((x0.shape[0], 1))
        z = flow.x_to_z(x0, 0. * ones)

    x_t_list = []
    for t_factor in torch.linspace(0., 1., steps, dtype=torch.float32).to(z):
        time = ones * t_factor
        x_t = flow.z_to_x(z, time)
        x_t_list.append(x_t)
    return x_t_list


@torch.no_grad()
def calc_w2(flow: Flow, n_iter=1, return_all=False, sample_from_z=False):
    costs = []
    for i in range(n_iter):
        x_t_list = xfixed_at_times(flow, sample_from_z=sample_from_z)

        dist = x_t_list[0] - x_t_list[-1]
        cost = torch.sum(dist * dist, dim=-1).mean()
        costs.append(cost.item())
    if return_all:
        return np.mean(costs), np.std(costs), costs
    else:
        return np.mean(costs), np.std(costs)


@torch.no_grad()
def evaluate(flow: Flow):
    w2_mean, w2_std = calc_w2(flow, n_iter=3)
    log.info(f"estimated W2 {w2_mean=:.3f} {w2_std=:.4f}")

    w2_mean, w2_std = calc_w2(flow, n_iter=3, sample_from_z=True)
    log.info(f"z sample estimated W2 {w2_mean=:.3f} {w2_std=:.4f}")

    x0 = torch.rand_like(x0_fixed) * 2 * lim - lim
    x1 = torch.rand_like(x1_fixed) * 2 * lim - lim
    logp0 = target0.logprob(x0)
    logp1 = target1.logprob(x1)

    y0 = torch.zeros(x0.shape[0], 1).to(x0)
    y1 = torch.ones(x1.shape[0], 1).to(x1)

    logrho0 = flow.log_prob(x0, y0)
    logrho1 = flow.log_prob(x1, y1)

    mse0 = ((logp0.exp() - logrho0.exp()) ** 2).mean().item()
    mse1 = ((logp1.exp() - logrho1.exp()) ** 2).mean().item()
    return mse0, mse1


def eval_and_save_results(flow, dirname):
    w2_mean, w2_std, costs = calc_w2(flow, n_iter=50, return_all=True)

    w2_mean_z, w2_std_z, costs_z = calc_w2(flow, n_iter=50, return_all=True, sample_from_z=True)

    mse0, mse1 = evaluate(flow)

    results = dict(w2_mean=w2_mean,
                   w2_std=w2_std,
                   costs=costs,
                   w2_mean_z=w2_mean_z,
                   w2_std_z=w2_std_z,
                   costs_z=costs_z,
                   mse0=mse0,
                   mse1=mse1,
                   **vars(args))

    with open(f"{dirname}/results.txt", "w") as fout:
        fout.write(pformat(results))

    with open(f"{dirname}/results.pkl", "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    flow = build_model()

    optimizer = optim.Adam(flow.parameters(), lr=2e-3, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=1e-6)

    start_time = time.time()
    testing_time = 0.
    for i in range(num_iter):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        x, logrho_target, condition = gen_samples(mb_size=mb_size)

        log_rho_pred = flow.log_prob(x, condition).squeeze()

        rho_loss = (torch.exp(log_rho_pred).clamp(min=1e-8) -
                    torch.exp(logrho_target.squeeze()).clamp(min=1e-8)).square().mean()

        x_ot, logrho_ot, condition_ot = gen_ot_samples(mb_size=mb_size)
        log_rho_pred_samples, vel_pred_samples = flow.log_density_and_velocity(x_ot, condition_ot)
        sample_prob = logrho_ot.exp()
        ot_loss = lim ** 2 * (torch.sum(vel_pred_samples * vel_pred_samples,
                                        dim=-1) * log_rho_pred_samples.exp() / sample_prob).mean()
        # ot_loss = flow.transport_cost_penalty(n_t_samples=32, n_z_samples=256, forward_mode=True)
        loss = rho_loss + ot_loss * (1. / args.weight_denom)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), .1)

        # print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f}MB")

        optimizer.step()
        # scheduler.step()

        wall_time = time.time() - start_time - testing_time
        if i % 5 == 0:
            log.info(f'i: {i}; Loss: {loss.item():.2e}  ot_loss: {ot_loss.item():.2e} {wall_time=:.2f}')
            # print(f'i: {i}; Loss: {loss.item():.2e} ')

        if (i + 1) % 100 == 0:
            flow.eval()
            testing_time_start = time.time()
            # log.handlers[0].flush()
            mse0, mse1 = evaluate(flow)
            log.info(f"MSE (t=0): {mse0:.2e} | MSE (t=1): {mse1:.2e}")

            plot_model(flow)

            flow.train()
            # ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
            # plt.title('iteration {}'.format(i + 1))
            plt.tight_layout()
            plt.savefig(f"{dirname}/densities_{i}.png")
            testing_time += time.time() - testing_time_start
            flow.train()

    flow.eval()
    eval_and_save_results(flow, dirname)


if __name__ == "__main__":
    main()
