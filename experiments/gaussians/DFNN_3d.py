import argparse

import os
import random
import time
from datetime import timedelta
import torch
from torch import optim
import numpy as np
import pandas as pd

from datasets.moving_gaussians import MovingGaussians3d as MovingGaussians, eval_model
from experiments.gaussians.util.exp_gaussians_util import plot_density_3d_divfree as plot_density, \
    print_loss_dict, device, count_parameters, eval_and_log_model

from experiments.gaussians.density_velocity_interface import DensityVelocityInterface
from experiments.gaussians.util.divergence_free.div_free import build_divfree_vector_field
from functorch import vmap

import optuna
import logging
import sys
from collections import OrderedDict
from enflows.utils.torchutils import np_to_tensor
from pprint import pprint, pformat
import matplotlib.pyplot as plt
from experiments.gaussians.util.divergence_free.model import NeuralConservationLaw

parser = argparse.ArgumentParser()
parser.add_argument('-no_vel', '--no_velocity', dest="velocity", action='store_false')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--test', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--optuna', dest="no_optuna", action='store_false')
parser.add_argument('--no-plots', action='store_true')

args = parser.parse_args()

INCLUDE_PLOTS = not args.no_plots


class DivFreeNetwork(torch.nn.Module, DensityVelocityInterface):
    def __init__(self, space_dim,
                 hidden_features,
                 num_layers, data_gen: MovingGaussians,
                 n_mixtures=64, load = False):
        super().__init__()
        self.space_dim = space_dim
        self.data_gen = data_gen
        self.in_features = self.out_features = space_dim + 1
        self._t_idx = 0
        self._x_idx = [i for i in range(self.in_features) if i != self._t_idx]

        self.network = NeuralConservationLaw(self.in_features, d_model=hidden_features, num_hidden_layers=num_layers,
                                             n_mixtures=n_mixtures).to(device)
        self.u_fn, self.params, self.A_fn = build_divfree_vector_field(self.network)
        self.u_fn_vmapped = vmap(self.u_fn, in_dims=(None, 0))

        if load:
            self.params = self.load()

    def get_antisymmetric_matrix(self, tx_samples):
        return vmap(self.A_fn, in_dims=(None, 0))(self.params, tx_samples)

    def get_divergence_free_vector(self, x):
        return self.u_fn_vmapped(self.params, x)

    def forward(self, x, mods=None):
        div_free_vec = self.get_divergence_free_vector(x)
        rho = div_free_vec[..., 0].unsqueeze(-1)
        flux = div_free_vec[..., 1:]
        return torch.concatenate([rho, flux], -1)
        # return self.network.forward(_x, mods)

    def density(self, x, t):
        return self.forward(torch.concatenate([t, x], -1))[..., [0]]

    def log_density(self, x, t):
        return self._log_density(torch.concatenate([t, x], -1))

    def _log_density(self, inputs):
        return (self.forward(inputs)[..., [0]]).log()

    def density_and_flux(self, x, t):
        return self._density_and_flux(torch.concatenate([t, x], -1))

    def _density_and_flux(self, inputs):
        return self.forward(inputs)

    def velocity(self, x, t):
        return self._velocity(torch.concatenate([t, x], -1))

    def _velocity(self, inputs):
        return self.forward(inputs)[..., 1:] / (self.forward(inputs)[..., [0]] + 1e-6)

    def flux(self, x, t):
        return self._flux(torch.concatenate([t, x], -1))

    def _flux(self, inputs):
        return self.forward(inputs)[..., 1:]

    def sample_input_domain(self, n_samples: int):
        data_x, data_t, _, __ = self.data_gen.sample_data(n_samples, t_subset="all", subset="all")
        tx_samples = np_to_tensor(np.concatenate([data_t[..., None], data_x], -1), device=device, dtype=torch.float32)
        return tx_samples.detach()

    def boundary_loss(self, n_samples=4096):
        tx_samples = self.sample_input_domain(n_samples)
        tx_samples_i = [tx_samples[i::6, ...] for i in range(6)]

        tx_samples_i[0][:, 1] = -4
        tx_samples_i[1][:, 1] = 4

        tx_samples_i[2][:, 2] = -4
        tx_samples_i[3][:, 2] = 4

        tx_samples_i[4][:, 3] = -4
        tx_samples_i[5][:, 3] = 4
        tx_samples_all = torch.concatenate(tx_samples_i, 0)

        all_outputs = self.forward(tx_samples_all)

        mask = all_outputs[..., [0]] > 1e-5
        loss = (mask * all_outputs.abs()).mean()  # .sqrt()
        # loss = all_outputs.abs().mean()  # .sqrt()
        return loss

    def zero_prior(self, n_samples=4096):
        tx_samples = self.sample_input_domain(n_samples)
        # density = self._log_density(tx_samples).exp()
        density = self.forward(tx_samples)[..., [0]]
        # log_rho = self._log_density(tx_samples_all)
        # vel = self._velocity(tx_samples_all)
        # loss = (vel.squeeze() * log_rho.exp().squeeze().unsqueeze(-1)).mean()
        loss = density.mean().sqrt()
        return loss

    def save(self, path):
        print(f"saving model to '{path}'")
        # torch.save(self.network, path)
        torch.save(self.params, path)

    def load(self) -> NeuralConservationLaw:
        print(f"Loading model from '{model_path}'")
        return torch.load(model_path)


def loss_fun(div_free_network: DivFreeNetwork, hparams: dict, data_list=None,
             eval=False):
    # if data_list is not None:
    data_x, data_t, target_logprob, target_vel = [data for data in data_list]

    if SETTINGS["velocity_data"]:
        tmp = div_free_network.density_and_flux(x=data_x, t=data_t)  # , div_free_network.flux(x=data_x, t=data_t)
        pred_log_density, pred_flux = tmp[..., [0]].log(), tmp[..., 1:]
    else:
        pred_log_density = div_free_network.log_density(x=data_x, t=data_t)

    loss_mse_rho = torch.nn.functional.mse_loss(target_logprob.squeeze().exp(),
                                                pred_log_density.squeeze().exp()
                                                ) ** 0.5

    if SETTINGS["velocity_data"]:
        vel_weight = hparams["vel_weight"]

        loss_mse_flux = torch.nn.functional.mse_loss(target_vel.squeeze() * target_logprob.exp().reshape(-1, 1),
                                                     pred_flux.squeeze()) ** 0.5
    else:
        vel_weight = 0.
        loss_mse_flux = data_x.new_zeros(tuple())

    loss_prior = hparams["prior_weight"] * div_free_network.zero_prior(
        2 * 1024)  # * div_free_network.boundary_loss(1024)
    loss_bv = hparams["boundary_weight"] * div_free_network.boundary_loss(
        2 * 1024)  # * div_free_network.boundary_loss(1024)
    loss = loss_mse_rho + vel_weight * loss_mse_flux + loss_bv + loss_prior

    loss_dict = {"total": loss.detach().cpu().numpy().item(),
                 "mse_density": loss_mse_rho.detach().cpu().numpy().item(),
                 "mse_velocity": loss_mse_flux.detach().cpu().numpy().item(),
                 }

    return loss, loss_dict


def train_loop(network: DivFreeNetwork, num_iter, optimizer, scheduler, data_gen: MovingGaussians,
               trial: optuna.trial.Trial,
               hparams: dict):
    start_time = time.monotonic()

    try:
        for iteration in range(num_iter):
            optimizer.zero_grad()
            train_data_list = data_gen.sample_data(n_samples=hparams["mb_size"], subset="train", t_subset="discrete")
            train_data_list = [np_to_tensor(val, device=device).reshape(hparams["mb_size"], -1)
                               for val in train_data_list]
            loss, loss_dict = loss_fun(network, data_list=train_data_list, hparams=hparams)

            loss.backward()
            optimizer.step()
            scheduler.step()

            # mlflow.log_metrics(loss_dict, step=iteration)
            if (iteration + 1) % 50 == 0:
                if INCLUDE_PLOTS:
                    plot_density(network, iteration, title=study_name,
                                 base_path=dirname, lim=data_gen.lim, include_vel=True,
                                 split_size=1_000, vmin=0, vmax=0.46, vel_scale=30)
            if (iteration + 1) % 10 == 0:
                end_time = time.monotonic()
                time_diff = timedelta(seconds=end_time - start_time)
                start_time = time.monotonic()
                print_loss_dict(iteration, loss_dict, time_diff)

                if np.isnan(loss_dict["total"]):
                    return np.nan

            if (iteration + 1) % 100 == 0:
                try:
                    network.eval()
                    val_r2_density = data_gen.score_density(network, device=device, subset="val")
                    network.train()

                    print(f'Val: {data_gen.score_density(network, subset="val", device=device, n_times=40):.2%}')
                    print(f'Test: {data_gen.score_density(network, subset="test", device=device, n_times=40):.2%}')
                    print(f"{val_r2_density:.2%}")
                except ValueError:
                    return np.nan

    except torch.cuda.OutOfMemoryError:
        print("OUT OF MEMORY !!!")
        raise optuna.exceptions.TrialPruned()
    except KeyboardInterrupt:
        print("interrupted training..")


    network.eval()
    network.save(model_path)
    print(f'Val: {data_gen.score_density(network, subset="val", device=device):.2f}')
    print(f'Test: {data_gen.score_density(network, subset="test", device=device):.2f}')

    plot_density(network, iteration, title=study_name, base_path=dirname, lim=data_gen.lim,
                 include_vel=True, split_size=1_000, vmin=0, vmax=0.46, log_scale=True, vel_scale=30)

    return data_gen.score_density(network, device=device, subset="val")


def objective(trial: optuna.trial.Trial, save=False):
    hparams = dict(num_layers=trial.suggest_int("num_layers", low=1, high=5),
                   hidden_features=2 ** trial.suggest_int("layerwise_hidden_features", 6, 8),
                   vel_weight=trial.suggest_float("vel_weight", low=1e-4, high=1e-1, log=True),  # 0.005
                   n_mixtures=2 ** trial.suggest_int("n_mixtures", 4, 8),
                   num_iter=2_000 if args.test else 1_000,
                   lr=trial.suggest_float("lr", low=1e-5, high=1e-2, log=True),
                   optimizer="Adam",
                   mb_size=2 * 1024,
                   # boundary_weight=1e-2,#trial.suggest_float("boundary_weight", low=1e-7, high=1e-1),
                   boundary_weight=trial.suggest_float("boundary_weight", low=1e-3, high=1e-1, log=True),
                   prior_weight=trial.suggest_float("prior_weight", low=1e-6, high=1, log=True),
                   SEED=args.seed)
    return _objective(hparams, trial, save=save)


def _objective(hparams: dict, trial: optuna.trial.Trial = None):
    set_seeds(SEED=hparams["SEED"])
    pprint(hparams)

    data_gen = MovingGaussians(num_timesteps=21)

    # mlflow.log_params(hparams)
    # if INCLUDE_PLOTS:
    #     plot_density(data_gen, title="discrete_timesteps", lim=data_gen.lim)

    network = DivFreeNetwork(space_dim=3,
                             hidden_features=hparams["hidden_features"],
                             num_layers=hparams["num_layers"],
                             data_gen=data_gen,
                             n_mixtures=hparams["n_mixtures"],
                             load=args.load,
                             ).to(device)
    print(f"---------Number of Parameters: {count_parameters(network)}")
    if not args.load:
        optimizer = getattr(optim, hparams["optimizer"])(network.params, lr=hparams["lr"])
        # optimizer = getattr(optim, hparams["optimizer"])(network.parameters(), lr=hparams["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2_000, eta_min=1e-7, last_epoch=-1,
                                                               verbose=False)
        try:
            val_r2_density = train_loop(network=network, optimizer=optimizer, scheduler=scheduler,
                                        data_gen=data_gen, num_iter=hparams["num_iter"],
                                        trial=trial, hparams=hparams)
        except KeyboardInterrupt:
            val_r2_density = -1
            print("Interrupted Training")

        if np.isnan(val_r2_density):
            return -2
    else:
        plot_density(network, 10_000, title=study_name, base_path=dirname, lim=data_gen.lim,
                     include_vel=True, split_size=1_000, vmin=0, vmax=0.46, log_scale=True, vel_scale=30)

    hparams_ = eval_and_log_model(hparams, network, args.seed, study_name, data_gen, split_size=512)

    return max(hparams_["val_score"], -1)


def set_seeds(SEED):
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    random.seed(SEED)



def load_and_eval_model():
    model: DivFreeNetwork = torch.load(model_path).to(device)
    print(f"Number of Parameters: {count_parameters(model)}")

    data_gen = MovingGaussians(num_timesteps=21)
    model.eval()

    model.load(model_path)
    eval_model(data_gen, model)


if __name__ == "__main__":
    SETTINGS = OrderedDict(
        velocity_data=args.velocity
    )

    set_str = [int(setting) for _, setting in SETTINGS.items()]
    study_name = f"DFNN_3d"  # Unique identifier of the study.
    model_path = "./saved_models/" + study_name + f"_{args.seed}.pt"

    dirname = f"./results/{study_name}"
    os.makedirs(dirname, exist_ok=True)

    storage_name = "sqlite:///optuna/{}.db".format(study_name)

    if args.no_optuna:
        hparams_no_optuna = {'SEED': args.seed,
                             'boundary_weight': 0.06424603275675751,
                             'hidden_features': 128,
                             'lr': 0.0036458038237092262,
                             'mb_size': 2 * 1024,
                             'n_mixtures': 16,
                             'num_iter': 2000,
                             'num_layers': 3,
                             'optimizer': 'Adam',
                             'prior_weight': 0.000263896650559716,
                             'vel_weight': 0.053455868421330335}

        _objective(hparams=hparams_no_optuna, trial=None)
    else:
        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        # objective = mlflc.track_in_mlflow()(objective)
        if not args.load:

            try:
                study = optuna.create_study(study_name=study_name, storage=storage_name,
                                            load_if_exists=not args.overwrite,
                                            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=150,
                                            #                                    interval_steps=1, ),
                                            direction="maximize")
            except optuna.exceptions.DuplicatedStudyError:
                optuna.delete_study(study_name=study_name, storage=storage_name)
                study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=False,
                                            # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=150,
                                            #                                    interval_steps=1, ),
                                            direction="maximize")

            if args.test:
                # print(study.best_trial.params)

                # hparams_default = study.best_trial.params
                # hparams_default["vel_weight"] = 1e-2
                # trial = optuna.trial.FixedTrial(hparams_default)
                # objective(study.best_trial, save=True)

                objective(study.best_trial, save=True)

            else:
                hparams_default = {
                    "num_layers": 2,
                    "hidden_features": int(np.log2(128)),
                    "lr": 0.0015847017006431641,
                    "vel_weight": 0.003315787578863481,  # .1,
                    "n_mixtures": int(np.log2(32)),
                    "boundary_weight": 1e-2,
                    "prior_weight": 0.047520173847855604,
                }

                study.enqueue_trial(hparams_default, skip_if_exists=True)
                study.optimize(objective, n_trials=100,
                               gc_after_trial=True)
