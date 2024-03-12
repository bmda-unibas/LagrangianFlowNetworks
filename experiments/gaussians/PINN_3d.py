import argparse

import os
import random
import time
from datetime import timedelta
import torch
from torch import optim
import numpy as np

from datasets.moving_gaussians import MovingGaussians3d
from experiments.gaussians.util.exp_gaussians_util import plot_density_3d, \
    print_loss_dict, device, eval_and_log_model, eval_model

from LFlow import transformed_mse
from enflows.utils.torchutils import gradient, divergence
from experiments.gaussians.density_velocity_interface import DensityVelocityInterface

import optuna
import logging
import sys
from collections import OrderedDict
from enflows.utils.torchutils import np_to_tensor
from siren_pytorch import SirenNet
from pprint import pprint, pformat

parser = argparse.ArgumentParser()
# parser.add_argument('-pde', '--pde_loss', action='store_true')
# parser.add_argument('-vel', '--velocity', action='store_true')
parser.add_argument('--no-vel', dest="velocity", action='store_false')
parser.add_argument('--no-pde', dest="pde_loss", action='store_false')

parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--n_samples_test', type=int, default=13)
parser.add_argument('--test', action='store_true')
parser.add_argument('--load', action='store_true')
parser.add_argument('--overwrite', action='store_true')
# parser.add_argument('--no-optuna', action='store_true')
parser.add_argument('--optuna', dest="no_optuna", action='store_false')
parser.add_argument('--no-plots', action='store_true')
parser.add_argument('--test_nsamples', action='store_true')

args = parser.parse_args()

INCLUDE_PLOTS = not args.no_plots

study_name = f"PINN3D"  # Unique identifier of the study.
model_path = "./saved_models/" + study_name + f"_{args.seed}.pt"


class MassConsPINN(torch.nn.Module, DensityVelocityInterface):
    def __init__(self, space_dim,
                 hidden_features,
                 num_layers,
                 data_gen: MovingGaussians3d,
                 **kwargs):
        super().__init__()
        self.data_gen = data_gen
        self.space_dim = space_dim
        self.in_features = self.out_features = space_dim + 1
        self._t_idx = 0
        self._x_idx = [i for i in range(self.in_features) if i != self._t_idx]

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
        # self.tx_samples = self.sample_input_domain(100_000).to("cuda").requires_grad_(True)

    def forward(self, x, mods=None):
        _x = torch.empty_like(x)
        # get everything to [0,1] for SIREN
        _x[..., 0] = x[..., 0] / 1.2
        _x[..., 1] = (x[..., 1] + 4) / 8
        _x[..., 2] = (x[..., 2] + 4) / 8
        _x[..., 3] = (x[..., 3] + 4) / 8
        vel = self.network_v(_x, mods)
        rho = self.network_rho(_x, mods)
        return torch.concatenate([rho, vel], -1)
        # return self.network.forward(_x, mods)

    def sample_input_domain(self, n_samples: int):
        data_x, data_t, _, __ = self.data_gen.sample_data(n_samples, t_subset="all", subset="all")
        tx_samples = np_to_tensor(np.concatenate([data_x, data_t[..., None]], -1), device=device, dtype=torch.float32)
        return tx_samples.detach()

    def log_density(self, x, t):
        return self._log_density(torch.concatenate([t, x], -1))

    def _log_density(self, inputs):
        return self.forward(inputs)[..., [0]]

    def log_density_and_velocity(self, x, t):
        return self._log_density_and_velocity(torch.concatenate([t, x], -1))

    def _log_density_and_velocity(self, inputs):
        return self.forward(inputs)

    def velocity(self, x, t):
        return self._velocity(torch.concatenate([t, x], -1))

    def _velocity(self, inputs):
        return self.forward(inputs)[..., 1:]

    def boundary_loss(self, n_samples=4096):
        tx_samples = self.sample_input_domain(n_samples)
        tx_samples_i = [tx_samples[i::4, ...] for i in range(4)]

        tx_samples_i[0][:, 1] = -4
        tx_samples_i[1][:, 1] = 4
        tx_samples_i[2][:, 2] = -4
        tx_samples_i[3][:, 2] = 4
        tx_samples_i[2][:, 3] = -4
        tx_samples_i[3][:, 3] = 4

        tx_samples_all = torch.concatenate(tx_samples_i, 0)

        log_rho = self._log_density(tx_samples_all)
        # vel = self._velocity(tx_samples_all)
        loss = (log_rho.exp().squeeze().unsqueeze(-1)).mean()
        return loss

    def pde_loss(self, n_samples=4096):
        tx_samples = self.sample_input_domain(n_samples).requires_grad_(True)

        log_rho, pde_loss = self._pde_loss(tx_samples)
        pde_loss = pde_loss.mean().sqrt()
        # pde_term_sq = (drho_dt + div_massflux).pow(2)
        # sparsity_loss = 1e-5 * torch.nn.functional.softplus(log_rho).mean()
        return pde_loss  # + sparsity_loss

    def _pde_loss(self, tx_samples):
        outputs = self._log_density_and_velocity(tx_samples)
        log_rho = outputs[..., [0]]
        velocity = outputs[..., 1:]
        rho = torch.exp(log_rho)
        mass_flux = rho * velocity
        div_massflux = divergence(mass_flux, tx_samples, x_offset=1)
        drho_dt = gradient(rho, tx_samples)[..., [0]]
        pde_loss = ((drho_dt + div_massflux)).pow(2)  # - 1e-2*rho.mean()
        return log_rho, pde_loss


def loss_fun(pinn: MassConsPINN, hparams: dict, data_list=None,
             eval=False):
    # if data_list is not None:
    data_x, data_t, target_logprob, target_vel = [data for data in data_list]
    # else:
    #     data_x, data_t, target_logprob, target_vel = data_generator.sample_data_discrete_t(to_tensor=True,
    #                                                                                        device=device,
    #                                                                                        partial_observe_only=True)

    if SETTINGS["velocity_data"]:
        pred_logprob, pred_vel = pinn.log_density(x=data_x, t=data_t), pinn.velocity(x=data_x, t=data_t)
    else:
        pred_logprob = pinn.log_density(x=data_x, t=data_t)

    loss_mse_rho = transformed_mse(
        target_logprob.squeeze(), pred_logprob.squeeze())
    # loss_mse_rho = torch.nn.functional.mse_loss(target_logprob.squeeze().exp().sqrt(), pred_logprob.squeeze().exp().sqrt())

    if SETTINGS["velocity_data"]:
        vel_weight = hparams["vel_weight"]
        loss_mse_vel = torch.nn.functional.mse_loss(
            target_vel.squeeze(), pred_vel.squeeze())
    else:
        vel_weight = 0.
        loss_mse_vel = data_x.new_zeros(tuple())

    if SETTINGS["pde_reg"] and not eval:
        pde_weight = hparams["pde_weight"]  # 1e-6
        loss_pde = pinn.pde_loss(n_samples=hparams["pde_samples"])
    else:
        pde_weight = 0.
        loss_pde = data_x.new_zeros(tuple())

    loss_bv = 0.01 * pinn.boundary_loss(4096)
    loss = loss_mse_rho + vel_weight * loss_mse_vel + pde_weight * loss_pde + loss_bv

    loss_dict = {"total": loss.detach().cpu().numpy().item(),
                 "mse_density": loss_mse_rho.detach().cpu().numpy().item(),
                 "mse_velocity": loss_mse_vel.detach().cpu().numpy().item(),
                 "pde_loss": loss_pde.detach().cpu().numpy().item(),
                 }

    return loss, loss_dict


def train_loop(network: MassConsPINN, num_iter, optimizer, scheduler, data_gen: MovingGaussians3d,
               trial: optuna.trial.Trial,
               hparams: dict):
    pprint(hparams)
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
            if (iteration + 1) % 250 == 0:
                if INCLUDE_PLOTS:
                    plot_density_3d(network, iteration, include_vel=True, title=active_settings,
                                    base_path=dirname + active_settings, lim=data_gen.lim)
            if (iteration + 1) % 10 == 0:

                end_time = time.monotonic()
                time_diff = timedelta(seconds=end_time - start_time)
                print_loss_dict(iteration, loss_dict, time_diff)
                start_time = time.monotonic()

                try:
                    val_r2_density = data_gen.score_density(network, device=device, subset="val")
                except ValueError:
                    return np.nan

                if trial is not None:
                    trial.report(val_r2_density, step=iteration)

    except KeyboardInterrupt:
        pass
    except torch.cuda.OutOfMemoryError:
        print("OUT OF MEMORY !!!")
        raise optuna.exceptions.TrialPruned()

    return data_gen.score_density(network, device=device, subset="val")


def objective(trial: optuna.trial.Trial):
    hparams = dict(num_layers=trial.suggest_int("num_layers", low=1, high=5),  # 5,#
                   hidden_features=2 ** trial.suggest_int("layerwise_hidden_features", 6, 9),  # 128
                   sine_frequency=12,  # trial.suggest_float("sine_frequency", 5, 30),
                   pde_weight=trial.suggest_float("pde_weight", low=1e-3, high=1e-1, log=True),
                   pde_samples=2 ** 16,  # 2 ** trial.suggest_int("pde_samples", low=10, high=16),
                   vel_weight=trial.suggest_float("vel_weight", low=1e-4, high=1e-1, log=True),  # 0.005
                   num_iter=2_000 if args.test else 1_000,  # faster hyperparam search
                   lr=trial.suggest_float("lr", low=1e-5, high=1e-2, log=True),
                   optimizer="Adam",
                   mb_size=4 * 4096,
                   SEED=args.seed)
    return _objective(hparams, trial)


def _objective(hparams: dict, trial: optuna.trial.Trial = None):
    set_seeds(SEED=hparams["SEED"])

    data_gen = MovingGaussians3d()

    network = MassConsPINN(space_dim=3,
                           hidden_features=hparams["hidden_features"],
                           num_layers=hparams["num_layers"],
                           sine_frequency=hparams["sine_frequency"],
                           data_gen=data_gen
                           ).to(device)
    optimizer = getattr(optim, hparams["optimizer"])(network.parameters(), lr=hparams["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2_000, eta_min=1e-7, last_epoch=-1,
                                                           verbose=False)

    val_r2_density = train_loop(network=network, optimizer=optimizer, scheduler=scheduler,
                                data_gen=data_gen, num_iter=hparams["num_iter"],
                                trial=trial, hparams=hparams)

    # evaluate_model(data_gen, network)

    # val_score, test_score = eval_model(model=network, data_gen=data_gen)
    # write_results(hparams, test_score, val_score)
    eval_and_log_model(hparams, network, args.seed, study_name, data_gen)
    print(f"saving file to: {model_path}")
    torch.save(network, model_path)

    return max(val_r2_density, -1)


def set_seeds(SEED):
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    random.seed(SEED)


def load_and_eval_model():
    model: MassConsPINN = torch.load(model_path).to(device)
    data_gen = MovingGaussians3d()
    eval_model(data_gen, model)


if __name__ == "__main__":
    SETTINGS = OrderedDict(
        pde_reg=args.pde_loss,
        velocity_data=args.velocity
    )

    dirname = "./results/PINN_3d"
    active_settings = ""
    for setting in SETTINGS.keys():
        if SETTINGS[setting]:
            active_settings += f"_{setting}"

    os.makedirs(dirname + active_settings, exist_ok=True)

    if args.no_optuna:
        INCLUDE_PLOTS = not args.no_plots

        hparams_no_optuna = {'SEED': args.seed,
                             'hidden_features': 256,
                             'lr': 0.0032357278100168018,
                             'mb_size': 16384,
                             'num_iter': 2000,
                             'num_layers': 5,
                             'optimizer': 'Adam',
                             'pde_samples': 65536,
                             'pde_weight': 0.002022983780650226,
                             'sine_frequency': 12,
                             'vel_weight': 0.001295032246964005}
        if not args.load:
            _objective(hparams=hparams_no_optuna, trial=None)
        else:
            model: MassConsPINN = torch.load(model_path).to(device)
            data_gen = MovingGaussians3d(num_timesteps=21)
            model.eval()
            # eval_model(data_gen, model)

            with torch.no_grad():
                hparams_ = eval_and_log_model(hparams_no_optuna, model, args.seed, study_name, data_gen,
                                              split_size=2_000)

    else:

        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        set_str = [int(setting) for _, setting in SETTINGS.items()]
        settings_string = f"_pde{set_str[0]}_vel{set_str[1]}"
        storage_name = "sqlite:///optuna/{}.db".format(study_name)
        print(storage_name)

        if not args.load:

            try:
                study = optuna.create_study(study_name=study_name, storage=storage_name,
                                            load_if_exists=not args.overwrite,
                                            direction="maximize")
            except optuna.exceptions.DuplicatedStudyError:
                optuna.delete_study(study_name=study_name, storage=storage_name)
                study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=False,
                                            direction="maximize")
            except KeyboardInterrupt:
                pass

            if args.test_nsamples:
                hparams_default = study.best_trial.params
                hparams_default["pde_samples"] = args.n_samples_test
                trial = optuna.trial.FixedTrial(hparams_default)
                objective(trial)
                load_and_eval_model(args.n_samples_test)
            elif args.test:
                print(study.best_trial.params)
                objective(study.best_trial, save=True)
                load_and_eval_model()

            else:
                hparams_default = {
                    "num_layers": 3,
                    "hidden_features": int(np.log2(128)),
                    "lr": 0.01,
                    "vel_weight": 0.01,  # .1,
                    "pde_weight": 0.001,
                }

                study.enqueue_trial(hparams_default, skip_if_exists=True)
                study.optimize(objective, n_trials=60,
                               gc_after_trial=True)
