"""
Util stuff like plots
"""
import torch

import tqdm
import random

from datasets.moving_gaussians import MovingGaussians, eval_model
from experiments.gaussians.util.plot_util import *
from experiments.gaussians.density_velocity_interface import DensityVelocityInterface
import matplotlib.colors as colors
from enflows.utils.torchutils import tensor_to_np
import os
import time

from pprint import pformat

random.seed(0)
torch.manual_seed(1234)
np.random.seed(0)

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


cmap = plt.get_cmap("Blues")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_loss_dict(iteration, loss_dict, time_diff, extra_text=""):
    loss_text = f"iter:{iteration:04d} || "
    for key, val in loss_dict.items():
        loss_text += f"{key}: {val:.2E} | "
    loss_text += f"| timedelta: {time_diff} |"
    print(extra_text + loss_text)


def write_results(hparams, study_name, seed):
    dirs = "./repeated_runs/"
    os.makedirs(dirs, exist_ok=True)
    filename = f"{dirs}{study_name}_{seed}.txt"
    print(f"Saved model to '{filename}'")
    with open(filename, "w") as fout:
        fout.write(pformat(hparams))


def plot_density(flow: DensityVelocityInterface,  i=None, title: str = "", base_path="./results",
                 num_timesteps=int(5 ** 2), ax_shape=None,
                 lim=1.5, device=device, include_vel=True, resolution=200, log_scale=True, condition=None,
                 split_size=10 * 1_000, vel_scale=300, paper=True, quiv_skip=10, vmin=None, vmax=None, flux=False,
                 clip=False):
    # cmap = plt.get_cmap("inferno")
    if condition is None:
        condition = np.arange(0, 1.25, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)


    if paper:
        condition = np.array([0., 0.5, 1.2])
        ax_shape = (1, 3)
        fig, all_axs = plt.subplots(*ax_shape, figsize=(int(ax_shape[1]) * 3, int(ax_shape[0]) * 3))
    else:
        ax_shape = (int(np.sqrt(len(condition))), int(np.sqrt(len(condition)))) if ax_shape is None else ax_shape
        fig, all_axs = plt.subplots(*ax_shape, figsize=(int(ax_shape[1]) * 3, int(ax_shape[0]) * 3))

    try:
        axs1 = all_axs.flatten()
    except AttributeError:
        axs1 = [all_axs]
    xline = torch.linspace(-lim, lim, steps=resolution).to(device)
    yline = torch.linspace(-lim, lim, steps=resolution).to(device)

    xgrid, ygrid = torch.meshgrid(xline, yline, indexing="xy")

    xyinput = torch.stack([xgrid, ygrid], -1).reshape(-1, 2)

    # xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    zgrids = []
    vels = []
    for cond in condition: #tqdm.tqdm(condition):
        with torch.no_grad():
            log_prob_pred = flow.log_density(xyinput,
                                             torch.tensor(cond).to(device) * torch.ones(xyinput.shape[0], 1).to(device)
                                             )
            prob_pred = np.exp(tensor_to_np(log_prob_pred)).reshape((xline.shape[0], yline.shape[0]))
            zgrids.append(prob_pred)
            # zgrids.append(
            #     log_prob_pred.exp().reshape(xline.shape[0], yline.shape[0]).cpu().numpy())
        xyinput_mbs = torch.split(xyinput.clone(), split_size)
        cond_mbs = torch.split((cond * torch.ones(xyinput.shape[0], 1).to(device)).clone(), split_size)
        if include_vel:
            vels_mbs = []
            for xyinput_mb, cond_mb in zip(xyinput_mbs, cond_mbs):
                # vels_mbs.append(flow.velocity(x=xyinput_mb, t=cond_mb[0]))
                if flux:
                    density_mb = np.exp(tensor_to_np(flow.log_density(xyinput_mb,
                                                     cond_mb)))
                    vel_mb = tensor_to_np(flow.velocity(x=xyinput_mb, t=cond_mb))
                    flux_mb = vel_mb * density_mb.reshape(-1,1)
                    # mask = (density_mb < 1e-3).squeeze()
                    # vel_mb[mask] = 0*vel_mb[mask]
                    vel_scale*=0.86
                    # flux_mb = flux_mb  / np.sqrt(np.sum(np.square(flux_mb), -1)+1e-5)[..., np.newaxis]
                    flux_mb[density_mb.squeeze()<3e-2] = 0.
                    vels_mbs.append(flux_mb)
                    # vels_mbs.append(flux_mb)
                else:
                    vel_mb = tensor_to_np(flow.velocity(x=xyinput_mb, t=cond_mb))
                    if clip:
                        density_mb = np.exp(tensor_to_np(flow.log_density(xyinput_mb,
                                                                          cond_mb)))
                        vel_mb[density_mb.squeeze()<3e-2] = 0.

                    vels_mbs.append(vel_mb)
            vels.append(np.concatenate(vels_mbs, 0))
        else:
            vels.append(None)

    vmin = np.min(np.array(zgrids)) if vmin is None else vmin
    vmax = np.max(np.array(zgrids)) if vmax is None else vmax

    for ax, zgrid, cond, vel in zip(axs1, zgrids, condition, vels):
        # ax.contourf(xgrid.cpu().numpy(),
        #             ygrid.cpu().numpy(),
        #             zgrid.cpu().numpy(), alpha=0.6)
        if log_scale:
            norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)
        im = ax.imshow(zgrid, alpha=0.9,
                       extent=[-lim, lim, -lim, lim], origin='lower', cmap=cmap,
                       # vmin=vmin, vmax=vmax
                       norm=norm
                       )
        # fig.colorbar(im, ax=ax)
        # ax.scatter(*xyinput.detach().numpy().T, c=zgrid.reshape(-1), alpha = 0.3, marker='+')

        ax.axvline(0., color="gray", ls='--')
        ax.axhline(0., color="gray", ls='--')

        if vel is not None:
            # stream_skip(*xyinput.cpu().detach().numpy().T, *vel.T, grid_shape=xgrid.shape, ax=ax, skip=6,
            #             color="gray")
            quiver_skip(*xyinput.cpu().detach().numpy().T, *vel.T, grid_shape=xgrid.shape, ax=ax, skip=quiv_skip,
                        scale=vel_scale,
                        color="gray")

        ax.set_aspect("equal")
        ax.set_title(f"t={cond:.2f}")
        ax.axis('off')

    if i is not None and (not paper):
        plt.suptitle('iteration {}'.format(i + 1))
    plt.tight_layout()
    # plt.show()
    if title != "":
        title = "_" + title
    os.makedirs(base_path, exist_ok=True)
    if i is not None:
        plt.savefig(f"{base_path}/plot_{i + 1}{title}.png")
    else:
        plt.savefig(f"{base_path}/plot_{title}.png")

    plt.close()


def plot_density_3d(flow: DensityVelocityInterface, i=None, title: str = "", base_path="./results",
                    num_timesteps=int(5 ** 2), ax_shape=None,
                    lim=1.5, device=device, include_vel=True, resolution=100, log_scale=True, condition=None,
                    vmin=0, vmax=0.46, split_size=10_000, vel_scale=30, paper=True):
    os.makedirs(base_path, exist_ok=True)
    # condition = np.arange(0, .5, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)
    # condition = np.arange(-1, .5, .5)  # np.linspace(0, 1.25, num_timesteps)
    # assert np.isclose(int(np.sqrt(num_timesteps)) ** 2, num_timesteps)
    # cmap = plt.get_cmap("inferno")
    skip_amount_quiver = 12
    if paper:
        condition = np.array([0., 0.5, 1.2])
        ax_shape = (1, 3)
    else:
        if condition is None:
            condition = np.arange(0, 1.25, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)
        ax_shape = (int(np.sqrt(len(condition))), int(np.sqrt(len(condition)))) if ax_shape is None else ax_shape
        # assert np.isclxose(num_timesteps, np.prod(ax_shape))
    fig, all_axs = plt.subplots(*ax_shape, figsize=(int(ax_shape[1]) * 3, int(ax_shape[0]) * 3))
    axs1 = all_axs.flatten()
    xline = torch.linspace(-lim, lim, steps=resolution).to(device)
    yline = torch.linspace(-lim, lim, steps=resolution).to(device)

    xgrid, ygrid = torch.meshgrid(xline, yline, indexing="xy")
    zgrid = torch.zeros_like(xgrid)

    xyzinput = torch.stack([xgrid, ygrid, zgrid], -1).reshape(-1, 3)

    # xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    zgrids = []
    vels = []

    if log_scale:
        norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for cond in tqdm.tqdm(condition):

        xyinput_mbs = torch.split(xyzinput.clone(), split_size)
        cond_mbs = torch.split((cond * torch.ones(xyzinput.shape[0], 1).to(device)).clone(), split_size)

        with torch.no_grad():
            # log_prob_pred = flow.log_density(xyzinput,
            #                                  torch.tensor(cond).to(device) * torch.ones(xyzinput.shape[0], 1).to(device)
            #                                  )
            # prob_pred = np.exp(tensor_to_np(log_prob_pred)).reshape((xline.shape[0], yline.shape[0]))

            probs_mb = []
            for xyinput_mb, cond_mb in zip(xyinput_mbs, cond_mbs):
                # vels_mbs.append(flow.velocity(x=xyinput_mb, t=cond_mb[0]))

                log_prob_pred = flow.log_density(xyinput_mb,
                                                 torch.tensor(cond_mb).to(device) * torch.ones(xyinput_mb.shape[0], 1).to(
                                                     device)
                                                 )
                probs_mb.append(np.exp(tensor_to_np(log_prob_pred)))

            prob_pred = np.concatenate(probs_mb, 0).reshape((xline.shape[0], yline.shape[0]))
            zgrids.append(prob_pred)

        if include_vel:
            vels_mbs = []
            for xyinput_mb, cond_mb in zip(xyinput_mbs, cond_mbs):
                # vels_mbs.append(flow.velocity(x=xyinput_mb, t=cond_mb[0]))

                # density_mb = flow.log_density(xyinput_mb,
                #                                  torch.tensor(cond_mb).to(device) * torch.ones(xyinput_mb.shape[0], 1).to(
                #                                      device)
                #                                  ).exp()
                # flux_mb = density_mb.unsqueeze(-1)*flow.velocity(x=xyinput_mb, t=cond_mb)
                # flux_mb = flux_mb / (flux_mb**2).sum(-1).unsqueeze(-1).sqrt()
                # vels_mbs.append(tensor_to_np(flux_mb))
                vels_mbs.append(tensor_to_np(flow.velocity(x=xyinput_mb, t=cond_mb)))

            vels.append(np.concatenate(vels_mbs, 0))
        else:
            vels.append(None)

    vmin = np.min(np.array(zgrids)) if vmin is None else vmin
    vmax = np.max(np.array(zgrids)) if vmax is None else vmax
    for ax, zgrid, cond, vel in zip(axs1, zgrids, condition, vels):
        # ax.contourf(xgrid.cpu().numpy(),
        #             ygrid.cpu().numpy(),
        #             zgrid,
        #             cmap=cmap,
        #             norm=norm)
        im = ax.imshow(zgrid, alpha=0.9,
                       extent=[-lim, lim, -lim, lim], origin='lower', cmap=cmap,
                       # vmin=vmin, vmax=vmax,
                       norm=norm
                       )
        # fig.colorbar(im, ax=ax)
        ax.axvline(0., color="gray", ls='--')
        ax.axhline(0., color="gray", ls='--')
        # ax.scatter(*xyinput.detach().numpy().T, c=zgrid.reshape(-1), alpha = 0.3, marker='+')
        if vel is not None:
            # stream_skip(*xyinput.cpu().detach().numpy().T, *vel.T, grid_shape=xgrid.shape, ax=ax, skip=6,
            #             color="gray")
            quiver_skip(*xyzinput[..., :-1].cpu().detach().numpy().T, *vel[..., :-1].T,
                        grid_shape=xgrid.shape, ax=ax, skip=skip_amount_quiver, scale=vel_scale,
                        color="gray", angles='xy', scale_units='xy')

        ax.set_aspect("equal")
        ax.set_title(f"t={cond:.2f}")
        ax.axis('off')

    if i is not None and (not paper):
        plt.suptitle('iteration {}'.format(i + 1))
    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    # plt.show()
    if title != "":
        title = "_" + title
    if i is not None:
        plt.savefig(f"{base_path}/plot_{i + 1}{title}.pdf")
    else:
        plt.savefig(f"{base_path}/plot_{title}.pdf")
    print(f"Saved figure to '{base_path}/plot_{title}.pdf'")
    plt.close()

def plot_density_3d_divfree(flow: DensityVelocityInterface, i=None, title: str = "", base_path="./results",
                    num_timesteps=int(5 ** 2), ax_shape=None,
                    lim=1.5, device=device, include_vel=True, resolution=100, log_scale=True, condition=None,
                    vmin=None, vmax=None, split_size=10_000, vel_scale=300, paper=True):
    # condition = np.arange(0, .5, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)

    # condition = np.arange(-1, .5, .5)  # np.linspace(0, 1.25, num_timesteps)
    # assert np.isclose(int(np.sqrt(num_timesteps)) ** 2, num_timesteps)
    # cmap = plt.get_cmap("inferno")

    if paper:
        condition = np.array([0., 0.5, 1.2])
        ax_shape = (1, 3)
    else:
        if condition is None:
            condition = np.arange(0, 1.25, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)
        ax_shape = (int(np.sqrt(len(condition))), int(np.sqrt(len(condition)))) if ax_shape is None else ax_shape
        # assert np.isclxose(num_timesteps, np.prod(ax_shape))
    fig, all_axs = plt.subplots(*ax_shape, figsize=(int(ax_shape[1]) * 3, int(ax_shape[0]) * 3))
    axs1 = all_axs.flatten()
    xline = torch.linspace(-lim, lim, steps=resolution).to(device)
    yline = torch.linspace(-lim, lim, steps=resolution).to(device)

    xgrid, ygrid = torch.meshgrid(xline, yline, indexing="xy")
    zgrid = torch.zeros_like(xgrid)

    xyzinput = torch.stack([xgrid, ygrid, zgrid], -1).reshape(-1, 3)

    # xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

    zgrids = []
    vels = []

    if log_scale:
        norm = colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for cond in tqdm.tqdm(condition):

        xyinput_mbs = torch.split(xyzinput.clone(), split_size)
        cond_mbs = torch.split((cond * torch.ones(xyzinput.shape[0], 1).to(device)).clone(), split_size)

        with torch.no_grad():
            # log_prob_pred = flow.log_density(xyzinput,
            #                                  torch.tensor(cond).to(device) * torch.ones(xyzinput.shape[0], 1).to(device)
            #                                  )
            # prob_pred = np.exp(tensor_to_np(log_prob_pred)).reshape((xline.shape[0], yline.shape[0]))

            probs_mb = []
            for xyinput_mb, cond_mb in zip(xyinput_mbs, cond_mbs):
                # vels_mbs.append(flow.velocity(x=xyinput_mb, t=cond_mb[0]))

                log_prob_pred = flow.log_density(xyinput_mb,
                                                 torch.tensor(cond_mb).to(device) * torch.ones(xyinput_mb.shape[0], 1).to(
                                                     device)
                                                 )
                probs_mb.append(np.exp(tensor_to_np(log_prob_pred)))

            prob_pred = np.concatenate(probs_mb, 0).reshape((xline.shape[0], yline.shape[0]))

            zgrids.append(prob_pred)

        if include_vel:
            vels_mbs = []
            for xyinput_mb, cond_mb in zip(xyinput_mbs, cond_mbs):
                # vels_mbs.append(flow.velocity(x=xyinput_mb, t=cond_mb[0]))
                density_mb = flow.log_density(xyinput_mb,
                                                 torch.tensor(cond_mb).to(device) * torch.ones(xyinput_mb.shape[0], 1).to(
                                                     device)
                                                 ).exp()

                flux_mb = flow.flux(x=xyinput_mb, t=cond_mb)
                flux_mb = flux_mb / (flux_mb**2).sum(-1).unsqueeze(-1).sqrt()
                # mask = (density_mb < 1e-3).squeeze()
                # vel_mb[mask] = 0*vel_mb[mask]
                vels_mbs.append(tensor_to_np(flux_mb))
            vels.append(np.concatenate(vels_mbs, 0))
        else:
            vels.append(None)

    vmin = np.min(np.array(zgrids)) if vmin is None else vmin
    vmax = np.max(np.array(zgrids)) if vmax is None else vmax
    for ax, zgrid, cond, vel in zip(axs1, zgrids, condition, vels):
        # ax.contourf(xgrid.cpu().numpy(),
        #             ygrid.cpu().numpy(),
        #             zgrid.cpu().numpy(), alpha=0.6)
        im = ax.imshow(zgrid, alpha=0.9,
                       extent=[-lim, lim, -lim, lim], origin='lower', cmap=cmap,
                       # vmin=vmin, vmax=vmax
                       norm=norm
                       )
        # fig.colorbar(im, ax=ax)
        ax.axvline(0., color="gray", ls='--')
        ax.axhline(0., color="gray", ls='--')
        # ax.scatter(*xyinput.detach().numpy().T, c=zgrid.reshape(-1), alpha = 0.3, marker='+')
        if vel is not None:
            # stream_skip(*xyinput.cpu().detach().numpy().T, *vel.T, grid_shape=xgrid.shape, ax=ax, skip=6,
            #             color="gray")
            quiver_skip(*xyzinput[..., :-1].cpu().detach().numpy().T, *vel[..., :-1].T,
                        grid_shape=xgrid.shape, ax=ax, skip=12, scale=vel_scale,
                        color="gray")

        ax.set_aspect("equal")
        ax.set_title(f"t={cond:.2f}")
        ax.axis('off')

    if i is not None and (not paper):
        plt.suptitle('iteration {}'.format(i + 1))

    plt.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.1, 0.05, 0.8])
    fig.colorbar(im, cax=cbar_ax)

    # plt.show()
    if title != "":
        title = "_" + title
    if i is not None:
        plt.savefig(f"{base_path}/plot_{i + 1}{title}.png")
    else:
        plt.savefig(f"{base_path}/plot_{title}.png")

    plt.close()



def plot_density_groundtruth(data_generator: MovingGaussians, iteration: int, title: str = "",
                             base_path="./results", split_quarters=False, num_timesteps=int(5 ** 2), ax_shape=None,
                             lim=1.5):
    num_timesteps = int(5 ** 2)
    condition = np.arange(0, 1.25, .25 / 5)  # np.linspace(0, 1.25, num_timesteps)
    # cmap = plt.get_cmap("inferno")

    # assert np.isclose(int(np.sqrt(num_timesteps)) ** 2, num_timesteps)
    ax_shape = (int(np.sqrt(num_timesteps)), int(np.sqrt(num_timesteps))) if ax_shape is None else ax_shape

    # fig, all_axs = plt.subplots(int(np.sqrt(num_timesteps)), int(np.sqrt(num_timesteps)), figsize=(13, 13))
    fig, all_axs = plt.subplots(*ax_shape, figsize=(int(ax_shape[1]) * 3, int(ax_shape[0]) * 3))

    axs1 = all_axs.flatten()
    xline = torch.linspace(-lim, lim, steps=100).to(device)
    yline = torch.linspace(-lim, lim, steps=100).to(device)

    xgrid, ygrid = torch.meshgrid(xline, yline, indexing="xy")

    xyinput = torch.stack([xgrid, ygrid], -1).reshape(-1, 2)

    zgrids = []
    fluxes = []
    for cond in condition:
        log_prob_pred = data_generator.logp_at_t(xyinput.cpu().numpy().astype("float64"),
                                                 time=cond * np.ones(xyinput.shape[0]))
        vel_x, vel_y = data_generator.dxy_dt(*xyinput.cpu().numpy().T, 0.1 * np.ones(xyinput.shape[0]))
        flux = np.stack([vel_x, vel_y], -1)

        zgrids.append(np.exp(log_prob_pred).reshape(xline.shape[0], yline.shape[0]))
        # flux = flux * zgrids[-1].flatten()[..., None]
        fluxes.append(flux)
    vmin = np.min(np.array(zgrids))
    vmax = np.max(np.array(zgrids))

    for ax, zgrid, cond, flux in zip(axs1, zgrids, condition, fluxes):
        # ax.contourf(xgrid.cpu().numpy(),
        #             ygrid.cpu().numpy(),
        #             zgrid.cpu().numpy(), alpha=0.6)
        im = ax.imshow(zgrid, alpha=0.9,
                       extent=[-lim, lim, -lim, lim], origin='lower', cmap=cmap,
                       vmin=vmin, vmax=vmax)

        ls = "-"
        alpha = 0.5
        if split_quarters:
            # lower left
            delta = 2e-2
            ax.axvline(-delta, ymax=0.5 - 2e-2, color="orangered", ls=ls, alpha=alpha)
            ax.axhline(-delta, xmax=0.5 - 2e-2, color="orangered", ls=ls, alpha=alpha)
            ax.axvline(delta, ymin=0.5 + 2e-2, color="orangered", ls=ls, alpha=alpha)
            ax.axhline(delta, xmin=0.5 + 2e-2, color="orangered", ls=ls, alpha=alpha)

            ax.axvline(+delta, ymax=0.5 - 2e-2, color="lime", ls=ls, alpha=alpha)
            ax.axvline(-delta, ymin=0.5 + 2e-2, color="lime", ls=ls, alpha=alpha)
            ax.axhline(+delta, xmax=0.5 - 2e-2, color="lime", ls=ls, alpha=alpha)
            ax.axhline(-delta, xmin=0.5 + 2e-2, color="lime", ls=ls, alpha=alpha)

            # ax.axhline(3e-2, xmax=0.5 - 2e-2, color="green", ls=ls)

            # # lower left
            # ax.axvline(-3e-2, ymax=0.5 - 2e-2, color="red", ls=ls)
            # ax.axhline(-3e-2, xmax=0.5 - 2e-2, color="red", ls=ls)

            # ax.axvline(0., color="gray", ls=ls)
            # ax.axhline(0., color="gray", ls=ls)
        else:
            ax.axvline(0., color="gray", ls=ls)
            ax.axhline(0., color="gray", ls=ls)
        # ax.scatter(*xyinput.detach().numpy().T, c=zgrid.reshape(-1), alpha = 0.3, marker='+')
        if flux is not None:
            stream_skip(*xyinput.cpu().detach().numpy().T, *flux.T, grid_shape=xgrid.shape, ax=ax, skip=6,
                        color="gray")

        ax.set_aspect("equal")
        ax.set_title(f"{cond:.2f}")
        ax.axis('off')

    plt.suptitle('iteration {}'.format(iteration + 1))
    plt.tight_layout()

    if title != "":
        title = "_" + title
    plt.savefig(f"{base_path}/plot_groundtruth{title}.png")
    plt.close()

    sums = [density.sum() for density in zgrids]

    plt.plot(condition, sums)
    plt.savefig(f"{base_path}/plot_sum_groundtruth{title}.png")


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.time_counter = 0.
        self.end_time = None
        self.time_list = []

    def reset(self):
        self.start_time = time.time()
        self.time_counter = 0.

    def reset_list(self):
        self.time_list = []

    def save_to_list(self):
        self.time_list.append(self.time_counter)

    def stop(self):
        self.end_time = time.time()
        self.time_counter += self.end_time - self.start_time

    def cont(self):
        self.start_time = time.time()

@torch.no_grad()
def eval_and_log_model(hparams, model, seed, study_name, data_gen, split_size=5_000):
    val_score, test_score, consistency_loss = eval_model(data_gen=data_gen, model=model, device=device,
                                                         verbose=True,
                                                         split_size=split_size)
    hparams["val_score"] = val_score
    hparams["test_score"] = test_score
    hparams["consistency_loss"] = consistency_loss
    write_results(hparams, study_name, seed)
    return hparams