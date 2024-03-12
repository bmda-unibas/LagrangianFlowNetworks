import argparse

import os
import random
import time
from datetime import timedelta

import torch

from torch import optim
import numpy as np
import optuna
import logging
import sys
from collections import OrderedDict
import pprint

from datasets.moving_gaussians import MovingGaussians3d as MovingGaussians, eval_model
from experiments.gaussians.util.exp_gaussians_util import plot_density_3d as plot_density, \
    print_loss_dict, device, count_parameters, Timer, eval_and_log_model, write_results
from LFlow import LagrangeFlow as _LagrangeFlow, transformed_mse

from enflows.transforms import *
from enflows.distributions import *
from enflows.nn.nets import *
from enflows.utils.torchutils import np_to_tensor

from experiments.gaussians.density_velocity_interface import DensityVelocityInterface

parser = argparse.ArgumentParser()
# parser.add_argument('-norm', '--regularize_norm', action='store_true')
parser.add_argument('-no-norm', '--no_regularize_norm', dest='regularize_norm', action='store_false')
parser.add_argument('-tp', '--regularize_transport_cost', action='store_true')
# parser.add_argument('-vel', '--velocity', action='store_true')
parser.add_argument('-no-vel', '--no_velocity', dest="velocity", action='store_false')
parser.add_argument('--no-plots', '--no_plots', action='store_true')
parser.add_argument('--optuna', dest="no_optuna", action='store_false')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--seed', type=int, default=1236)
parser.add_argument('--test', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--load', action='store_true')

args = parser.parse_args()

# INCLUDE_PLOTS = False
INCLUDE_PLOTS = not args.no_plots

study_name = f"lflow_3d"   # Unique identifier of the study.
model_path = "./saved_models/" + study_name + f"_{args.seed}.pt"
os.makedirs("./saved_models", exist_ok=True)

class LagrangeFlow(_LagrangeFlow, DensityVelocityInterface):
    pass


def build_model(num_layers, context_features, hidden_features, hidden_features_shared,
                **kwargs) -> LagrangeFlow:
    base_dist = StandardNormal(shape=[3])  # ,
    transforms = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=3,
                                                densenet_depth=3,
                                                activation_function=CSin(w0=15),
                                                # activation_function=CLipSwish(),
                                                lip_coeff=.97,
                                                n_lipschitz_iters=5,
                                                context_features=context_features
                                                )

    transforms.append(InverseTransform(PointwiseAffineTransform(scale=4)))
    transforms.append(InverseTransform(Tanh()))
    # transforms.append(PointwiseAffineTransform(scale=4))
    activation = torch.nn.functional.silu
    for i in range(num_layers):
        transforms.append(RandomPermutation(3))
        transforms.append(ActNorm(3))
        transforms.append(iResBlock(densenet_builder.build_network(), brute_force=True, context_features=context_features))

        transforms.append(ConditionalSVDTransform(features=3, context_features=context_features,
                                                  hidden_features=hidden_features))

    transforms.append(ConditionalSVDTransform(features=3, context_features=context_features,
                                              hidden_features=hidden_features, activation=activation))
    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=hidden_features_shared,
                                num_blocks=2,
                                activation=activation)
    flow = LagrangeFlow(transform, base_dist, embedding_net).to(device)
    return flow


def loss_fun(flow: LagrangeFlow, data_list=None, eval=False, **kwargs):
    data_x, data_t, target_logprob, target_vel = [data for data in data_list]
    if SETTINGS["velocity_data"]:
        pred_logprob = flow.log_density(x=data_x, t=data_t)
        pred_vel = flow.velocity(x=data_x, t=data_t)
    else:
        pred_logprob = flow.log_density(x=data_x, t=data_t)

    loss_mse_rho = transformed_mse(
        target_logprob.squeeze(), pred_logprob.squeeze())

    if SETTINGS["velocity_data"]:
        vel_weight = kwargs.get(
            "vel_weight")  # trial.suggest_float("vel_weight", low=0.0001, high=1e-1, log=True)  # 0.005

        loss_mse_vel = torch.nn.functional.mse_loss(
            target_vel.squeeze(), pred_vel.squeeze())
    else:
        vel_weight = 0.
        loss_mse_vel = data_x.new_zeros(tuple())

    if SETTINGS["transport_cost_reg"] and not eval:
        tp_weight = kwargs.get("tp_weight")  # trial.suggest_float("tp_weight", low=1e-7, high=1e-1, log=True)  # 1e-6
        loss_transport_cost = flow.transport_cost_penalty(
            n_z_samples=200, n_t_samples=20, forward_mode=False)
    else:
        tp_weight = 0.
        loss_transport_cost = data_x.new_zeros(tuple())

    if SETTINGS["normalization_reg"] and not eval:
        norm_weight = kwargs.get(
            "norm_weight")  # trial.suggest_float("norm_weight", low=1e-6, high=1e-1, log=True)  # 5e-5
        loss_normalization_constant = flow.global_density_penalty()
    else:
        norm_weight = 0.
        loss_normalization_constant = data_x.new_zeros(tuple())

    loss = loss_mse_rho + vel_weight * loss_mse_vel + norm_weight * \
           loss_normalization_constant + tp_weight * loss_transport_cost

    loss_dict = {"total": loss.detach().cpu().numpy().item(),
                 "mse_density": loss_mse_rho.detach().cpu().numpy().item(),
                 "mse_velocity": loss_mse_vel.detach().cpu().numpy().item(),
                 "transport_cost": loss_transport_cost.detach().cpu().numpy().item(),
                 "norm_loss": loss_normalization_constant.detach().cpu().numpy().item()
                 }

    return loss, loss_dict


def train_loop(network, optimizer, scheduler, data_gen: MovingGaussians, trial: optuna.trial.Trial,
               hparams):
    timer = Timer()

    try:
        for iteration in range(1, hparams["num_iter"] + 1):
            timer.reset()
            optimizer.zero_grad()
            train_data_list = data_gen.sample_data(n_samples=hparams["mb_size"], subset="train", t_subset="discrete")
            train_data_list = [np_to_tensor(val, device=device).reshape(hparams["mb_size"], -1)
                               for val in train_data_list]
            loss, loss_dict = loss_fun(network, data_list=train_data_list, **hparams)

            loss.backward()
            optimizer.step()
            scheduler.step()
            timer.stop()
            timer.save_to_list()
            if iteration % 125 == 0:
                if INCLUDE_PLOTS:
                    network.eval()
                    plot_density(network, iteration - 1, title="lflow_3d",
                                 base_path=dirname, lim=4 - 1e-3,
                                 include_vel=True)
                    #
                    network.train()

            if iteration % 20 == 0:
                with torch.no_grad():
                    network.eval()
                    try:
                        val_mse_density = data_gen.score_density(network, device=device, subset="val")
                    except ValueError:
                        return np.nan
                    network.train()
                if trial is not None:
                    trial.report(val_mse_density, step=iteration)

                # Handle pruning based on the intermediate value.
                # if trial.should_prune():
                #     raise optuna.TrialPruned()

            if iteration == 1 or iteration % 5 == 0:
                print_loss_dict(iteration, loss_dict, np.round(np.mean(timer.time_list), 2))

    except KeyboardInterrupt:

        pass
    except torch.cuda.OutOfMemoryError:
        raise optuna.exceptions.TrialPruned()
    print("Saving model to './saved_models'")
    torch.save(network.state_dict(), model_path)

    # val_score, test_score = eval_model(model=network, data_gen=data_gen)
    # write_results(hparams, test_score, val_score)
    eval_and_log_model(hparams, network, args.seed, study_name, data_gen)

    return data_gen.score_density(network, device=device, subset="val")


def optuna_objective(trial: optuna.trial.Trial):
    hparams = dict(num_layers=trial.suggest_int("num_layers", low=2, high=6, step=2),
                   context_features=2 ** trial.suggest_int("context_features", 0, 8),
                   hidden_features=2 ** trial.suggest_int("hidden_features", 4, 9),
                   hidden_features_shared=2 ** trial.suggest_int("hidden_features_shared", 4, 9),
                   n_sigmoids=trial.suggest_int("n_sigmoids", low=30, high=30),
                   vel_weight=trial.suggest_float("vel_weight", low=1e-6, high=1e-1, log=True) if SETTINGS[
                       "velocity_data"] else 0,
                   norm_weight=trial.suggest_float("norm_weight", low=1e-6, high=1e-2, log=True) if SETTINGS[
                       "normalization_reg"] else 0,
                   tp_weight=trial.suggest_float("tp_weight", low=1e-7, high=1e-3, log=True) if SETTINGS[
                       "transport_cost_reg"] else 0,

                   num_iter=2_000,  # 1_00 for hyperparam tuning so its quicker..
                   lr=trial.suggest_float("lr", low=1e-5, high=1e-3, log=True),
                   optimizer="Adam",
                   mb_size=4 * 4096,
                   SEED=args.seed)

    val_mse_density = _objective(hparams, trial)
    return val_mse_density


def _objective(hparams, trial=None):
    pprint.pprint(hparams)
    set_seeds(SEED=hparams["SEED"])
    data_gen = MovingGaussians(num_timesteps=21)
    # if INCLUDE_PLOTS:
    # plot_density(data_gen, base_path=dirname + "/..", i=None, title="groundtruth_3d", lim=4 - 1e-3,
    #              include_vel=True)
    flow = build_model(**hparams)
    optimizer = getattr(optim, hparams["optimizer"])(flow.parameters(), lr=hparams["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2_000,
                                                           eta_min=1e-7, last_epoch=-1,
                                                           verbose=False)
    val_mse_density = train_loop(network=flow, optimizer=optimizer, scheduler=scheduler,
                                 data_gen=data_gen, trial=trial, hparams=hparams)
    return val_mse_density


def set_seeds(SEED):
    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    random.seed(SEED)


def load_and_eval_model(hparams):
    model = build_model(**hparams)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(f"Number of Parameters: {count_parameters(model)}")
    data_gen = MovingGaussians(num_timesteps=21)
    model.eval()

    plot_density(model, title="lflow_3d" + "_final",
                 base_path=dirname, lim=data_gen.lim, include_vel=True,
                 split_size=1_000)

    eval_and_log_model(hparams, model, args.seed, study_name, data_gen)


if __name__ == "__main__":

    SETTINGS = OrderedDict(
        normalization_reg=args.regularize_norm,
        transport_cost_reg=args.regularize_transport_cost,
        velocity_data=args.velocity
    )

    dirname = "./results/lflows_3d"
    active_settings = ""
    for setting in SETTINGS.keys():
        if SETTINGS[setting]:
            active_settings += f"_{setting}"
    os.makedirs(dirname, exist_ok=True)

    no_optuna_hparams = {'SEED': args.seed,
                         'context_features': 5,
                         'hidden_features': 64,
                         'hidden_features_shared': 128,  # 256,
                         'lr': 1e-3,  # 0.0009978760121721764,
                         'mb_size': 16384,
                         'norm_weight': 1.2411241118485362e-05,
                         'num_iter': 2000,
                         'num_layers': 10,
                         'optimizer': 'Adam',
                         'tp_weight': 0,
                         'vel_weight': 1.3966875452555474e-05}
    if args.no_optuna:

        if args.load:
            load_and_eval_model(no_optuna_hparams)
        else:
            _objective(no_optuna_hparams)

    else:
        # Add stream handler of stdout to show the messages
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        set_str = [int(setting) for _, setting in SETTINGS.items()]
        settings_string = f"_norm{set_str[0]}_tp{set_str[1]}_vel{set_str[2]}"
        storage_name = "sqlite:///optuna/{}.db".format(study_name)

        try:
            study: optuna = optuna.create_study(study_name=study_name, storage=storage_name,
                                                load_if_exists=not args.overwrite,
                                                # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=150,
                                                #                                interval_steps=1, ),
                                                direction="maximize")
        except optuna.exceptions.DuplicatedStudyError:
            optuna.delete_study(study_name=study_name, storage=storage_name)
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=False,
                                        # pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=150,
                                        #                                        interval_steps=1, ),
                                        direction="maximize")

        if args.test:
            optuna_objective(study.best_trial)
        else:

            default_params = {'SEED': 1236,
                              'context_features': int(np.log2(32)),
                              'hidden_features': int(np.log2(64)),
                              'hidden_features_shared': int(np.log2(256)),
                              'lr': 0.0009978760121721764,
                              'mb_size': 16384,
                              'n_sigmoids': 30,
                              'norm_weight': 1.2411241118485362e-05,
                              'num_iter': 2000,
                              'num_layers': 2,
                              'optimizer': 'Adam',
                              'tp_weight': 0,
                              'vel_weight': 1.3966875452555474e-05}

            if not SETTINGS["normalization_reg"]:
                default_params.pop("norm_weight")

            if not SETTINGS["transport_cost_reg"]:
                default_params.pop("tp_weight")

            if not SETTINGS["velocity_data"]:
                default_params.pop("vel_weight")

            study.enqueue_trial(default_params, skip_if_exists=True)
            study.optimize(optuna_objective, n_trials=100, gc_after_trial=True)
