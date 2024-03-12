import pandas as pd
from torch.utils.jit.log_extract import time_cpu

from datasets import bird_data
from datasets.bird_data import BirdDatasetMultipleNightsLeaveOutMiddle
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.nn.functional import softplus, mse_loss

from experiments.bird_migration.bird_utils import open_or_create_study, weighted_MSE, Timer
from experiments.bird_migration.models.model_builder import build
from experiments.bird_migration.plot_util import *
from experiments.bird_migration.models.template import DensityVelocityInterface
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import torch
from torch.utils.data import DataLoader
import pickle
import argparse
from enflows.utils.torchutils import set_seeds
from pprint import pprint
import json

import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=301)
parser.add_argument('--vel_factor', type=float, default=1.)
parser.add_argument('--norm_val', type=float, default=1e-3)
parser.add_argument('--name', type=str, default="default")
parser.add_argument('--save_dir', type=str, default="experiment")
parser.add_argument('--start_date', type=str, default="2018-04-05")  # 2018-03-12
parser.add_argument('--end_date', type=str, default="2018-04-07")  # 2018-03-14

parser.add_argument('--load', action='store_true')
parser.add_argument('--no_plots', action='store_true')
parser.add_argument('--checkpoint', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--model', type=lambda s: s.lower(), default="lflow",
                    choices=["lflow", "mlp", "pinn", "dfnn", "slda"])

parser.add_argument('--continue_train', action='store_true')
args = parser.parse_args()

BirdData = BirdDatasetMultipleNightsLeaveOutMiddle

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_seeds(args.seed)
# CHECKPOINT_DIR = f"checkpoints/{args.model}/{args.save_dir}"
CHECKPOINT_DIR = f"checkpoints/{args.model}"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint_s{args.start_date}_e{args.end_date}_{args.seed}_{args.vel_factor}.pt"

RESULTS_DIR = f"results/{args.save_dir}/{args.model}"
RESULTS_PATH = f"{RESULTS_DIR}/results_s{args.start_date}_e{args.end_date}_{args.seed}_{args.vel_factor}.json"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if args.model == "lflow" or args.model == "lflow_sso":  # or args.model == "dfnn":
    scale_vel_loss = (1. / BirdData.scale_time) * BirdData.scale_space
    BirdData.scale_space = 1.
    BirdData.scale_time = 1.
else:
    scale_vel_loss = 1.

complete_ds = BirdData(subset="all", start_date=args.start_date, end_date=args.end_date)
_train_ds = BirdData(subset="train", start_date=args.start_date, end_date=args.end_date)

CHECKPOINT = args.checkpoint
CONTINUE_TRAIN = args.continue_train

# assert torch.cuda.is_available(), "No CUDA available!"


data_sampler_seed = torch.Generator(device='cuda')


def sample_radar_area(txyz, radius=100, altitude_std=0.1 / 3):
    sample_radius = (torch.rand(txyz.shape[0], generator=data_sampler_seed, device=device,
                                dtype=torch.float32) * radius ** 2).sqrt()
    sample_angle = torch.rand(txyz.shape[0], generator=data_sampler_seed, device=device,
                              dtype=torch.float32) * 2 * np.pi
    sample_z = torch.randn(txyz.shape[0], generator=data_sampler_seed, device=device,
                           dtype=torch.float32) * altitude_std
    sample_x = sample_radius * torch.cos(sample_angle)
    sample_y = sample_radius * torch.sin(sample_angle)
    sample_xyz = torch.stack([sample_x, sample_y, sample_z], -1).detach()
    return sample_xyz, sample_radius, radius


def matern52(dists, lengthscale=1):
    K = (dists / lengthscale) * np.sqrt(5)
    K = (1.0 + K + K ** 2 / 3.0) * torch.exp(-K)
    return K


class BirdModel(L.LightningModule):
    def __init__(self, net, loss_params: dict, train_params: dict):
        super().__init__()
        self.net: DensityVelocityInterface = net
        self.total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.loss_params = loss_params
        self.train_params = train_params

        tmp = torch.ones(1, 2)
        tmp[:, 1] = 1.3
        self.uv_balance = torch.nn.Parameter(tmp, requires_grad=False)

        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        txyz, _rho, _uv, uv_mask, rho_mask = batch["txyz"], batch["rho"], batch["uv"], batch["uv_mask"], batch[
            "rho_mask"]

        rho = _rho[rho_mask].squeeze()
        uv = _uv[uv_mask]

        txyz_rho = txyz[rho_mask, ...]
        txyz_uv = txyz[uv_mask, ...]

        loss_weight_rho = loss_weight_vel = 1.

        pred_uv = self.net.velocity(x=txyz_uv[..., 1:],
                                    t=txyz_uv[..., :1])[..., :-1]

        pred_logprob = self.net.log_density(x=txyz_rho[..., 1:],
                                            t=txyz_rho[..., :1]).squeeze()
        # loss_rho = mse_loss(torch.log1p(rho), softplus(pred_logprob))
        loss_rho = weighted_MSE(torch.log1p(rho), softplus(pred_logprob), weights=loss_weight_rho)
        loss_uv = weighted_MSE(uv, pred_uv, weights=scale_vel_loss * self.uv_balance * loss_weight_vel)

        additional_loss = self.net.data_independent_loss(**self.loss_params)
        loss = loss_rho + additional_loss + self.loss_params["vel_weight"] * self.loss_params["vel_factor"] * loss_uv

        self.log("train_loss", loss)
        self.log("train_vel_loss", mse_loss(scale_vel_loss * uv, scale_vel_loss * pred_uv), prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log("train_density_loss", loss_rho, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_additional_loss", additional_loss)
        return loss

    def test_step(self, batch, batch_idx):
        txyz, _rho, _uv, uv_mask, rho_mask = batch["txyz"], batch["rho"], batch["uv"], batch["uv_mask"], batch[
            "rho_mask"]

        rho = _rho[rho_mask].squeeze()
        uv = _uv[uv_mask]

        txyz_rho = txyz[rho_mask, ...]
        txyz_uv = txyz[uv_mask, ...]

        pred_logprob = self.net.log_density(x=txyz_rho[..., 1:], t=txyz_rho[..., :1]).squeeze()
        pred_vel = self.net.velocity(x=txyz_uv[..., 1:], t=txyz_uv[..., :1])
        pred_uv: torch.Tensor = pred_vel[..., :-1]

        loss_rho = mse_loss(torch.log1p(rho),
                            softplus(pred_logprob))

        loss_uv = mse_loss(scale_vel_loss * uv, scale_vel_loss * pred_uv)

        self.log("test_vel_loss", loss_uv, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_density_loss", loss_rho, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        txyz, _rho, _uv, uv_mask, rho_mask = batch["txyz"], batch["rho"], batch["uv"], batch["uv_mask"], batch[
            "rho_mask"]

        rho = _rho[rho_mask].squeeze()
        uv = _uv[uv_mask]

        txyz_rho = txyz[rho_mask, ...]
        txyz_uv = txyz[uv_mask, ...]

        pred_logprob = self.net.log_density(x=txyz_rho[..., 1:], t=txyz_rho[..., :1]).squeeze()
        pred_vel = self.net.velocity(x=txyz_uv[..., 1:], t=txyz_uv[..., :1])
        pred_uv: torch.Tensor = pred_vel[..., :-1]

        loss_rho = mse_loss(torch.log1p(rho),
                            softplus(pred_logprob))

        loss_uv = mse_loss(scale_vel_loss * uv, scale_vel_loss * pred_uv)

        self.log("val_vel_loss", loss_uv, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_density_loss", loss_rho, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        if args.model == "dfnn":
            params = self.net.params
        elif args.model == "slda":
            params = [{"params": self.net.base_grid.parameters(), "lr": 1e-2,
                       "weight_decay": 0.},
                      {"params": self.net.cnf_to_t0.parameters()},
                      ]
        else:
            params = self.parameters()
        optimizer = torch.optim.Adam(params, lr=self.train_params["lr"],
                                     weight_decay=self.train_params["weight_decay"])
        mbs_per_epoch = _train_ds.df.shape[0] // self.train_params["mb_size"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_params["num_iter"] * mbs_per_epoch,
                                                               eta_min=1e-7, last_epoch=-1,
                                                               verbose=False
                                                               )

        lr_scheduler = {'scheduler': scheduler,
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]

    def log_density(self, x, t):
        return self.net.log_density(x, t)

    def velocity(self, x, t):
        return self.net.velocity(x, t)


def train_model(train_params, model_params, loss_params):
    train_dataloader = generate_dataloader("final_train", train_params["mb_size"], seed=train_params["SEED"])
    test_dataloader = generate_dataloader("test", train_params["mb_size"], seed=train_params["SEED"])

    net = build(train_params["model"], model_params=model_params,
                complete_dataset=complete_ds,
                train_dataset=train_dataloader.dataset)

    model = BirdModel(net=net, train_params=train_params, loss_params=loss_params)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(max_epochs=train_params["num_iter"], accelerator="cuda", devices=[0, ],
                        enable_progress_bar=True,
                        enable_model_summary=False,
                        gradient_clip_val=1,
                        check_val_every_n_epoch=5,
                        log_every_n_steps=9,
                        logger=pl_loggers.TensorBoardLogger(save_dir=args.save_dir,
                                                            # name="training_subset",
                                                            name=args.model,
                                                            sub_dir=f"{args.name}_{args.start_date}_{args.end_date}"),
                        callbacks=[lr_monitor],
                        inference_mode=False
                        )

    timer = Timer()

    # if args.model == "slda":
    #     model.net.pretrain_MLP(train_dataloader)

    try:
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        timer.stop()
    except KeyboardInterrupt:
        timer.stop()
        pass

    test_results = trainer.test(model, test_dataloader)[0]
    final_dictionary = dict(**train_params, **model_params, **loss_params, **test_results,
                            walltime=np.round(timer.time_counter, 2))

    pprint(final_dictionary)
    print(f"Dumped results to '{RESULTS_PATH}'.")
    with open(RESULTS_PATH, 'w') as fp:
        json.dump(final_dictionary, fp, sort_keys=True, indent=4)

    if args.checkpoint:
        save_model(model.net)
    return model


def save_model(model):
    torch.save({
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'test_losses': test_losses,
    }, CHECKPOINT_PATH)


def generate_dataloader(subset, mb_size, seed):
    bird_dataset = BirdData(subset=subset,
                            transform=bird_data.ToTensor(device),
                            seed=seed,
                            start_date=args.start_date,
                            end_date=args.end_date,
                            )
    sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.RandomSampler(bird_dataset),
                                                    batch_size=mb_size, drop_last=False)
    dataloader = DataLoader(bird_dataset, sampler=sampler, batch_size=None)
    return dataloader


def load_hyperparameters(args):
    shared_train_params = {
        'model': args.model,
        'SEED': args.seed,
        'optimizer': 'Adam',
        'load': args.load,
        'checkpoint': args.checkpoint,
        'overwrite': args.overwrite,
        'test': args.test,
    }
    train_params = {
        "lflow": {'lr': 1e-2, "weight_decay": 2e-3, 'mb_size': 16384,
                  "num_iter": 50, **shared_train_params},
        "slda": {'lr': 1e-3, 'mb_size': 16384, "num_iter": 300, **shared_train_params, "weight_decay": 5e-3},
        # 3e-2},
        "pinn": {'lr': 1e-3, "num_iter": 300, "weight_decay": 2e-3,  # "weight_decay": 5e-6,
                 'mb_size': 16384, **shared_train_params},
        "mlp": {'lr': 1e-3, "num_iter": 100,
                "weight_decay": 1e-3,  # 5e-6,
                'mb_size': 16384, **shared_train_params},
        "dfnn": {'lr': 1e-3, "num_iter": 100, "weight_decay": 0, 'mb_size': 1024 * 2, **shared_train_params},
    }
    shared_loss_params = {
        "vel_factor": args.vel_factor
    }

    loss_params = {
        "lflow": {
            'vel_weight': 3,
            # 'norm_weight': 1e-3,
            'norm_weight': args.norm_val,
            **shared_loss_params
        },
        "slda": {
            'vel_weight': 2.5,
            'norm_weight': 2,
            **shared_loss_params
        },
        "mlp": {
            'vel_weight': 10,
            **shared_loss_params
        },
        "pinn": {
            'vel_weight': 1,
            "pde_weight": 7e-4,
            "collocation_points": 100_000,
            **shared_loss_params
        },
        "dfnn": {
            "vel_weight": 1.5,
            **shared_loss_params
        }
    }
    model_params = {
        "lflow": {
            'context_features': 1,
            'hidden_features_shared': 128,
            'init_log_scale_norm': 18.2,
            'num_layers': 10,
        },
        "slda": {
            'hidden_features': 512,
            'init_log_scale_norm': 17.2,
            'num_layers': 5,
            'activation': 'swish'},
        "mlp": {},
        "pinn": {"num_layers": 5,
                 "hidden_features": 128,
                 "sine_frequency": 5
                 },
        "dfnn": {"num_layers": 5,
                 "hidden_features": 256,
                 "n_mixtures": 64}}
    train_params, model_params, loss_params = train_params[args.model], model_params[args.model], loss_params[
        args.model]
    return train_params, model_params, loss_params


if __name__ == '__main__':
    train_params, model_params, loss_params = load_hyperparameters(args)
    pprint(train_params)
    pprint(model_params)
    pprint(loss_params)

    dirname = "./results/birds"
    os.makedirs(dirname, exist_ok=True)

    if not train_params["load"]:
        model = train_model(train_params, model_params, loss_params)
        model.eval()
        if not args.no_plots:
            try:
                model.to(device)
            except Exception:
                pass

            density, velocity = predict_2d(complete_ds, model.net, sobol_power=6)
            with open(f'lflow_preds_{args.norm_val:.2e}.pkl', 'wb') as fp:
                pickle.dump({"density": density,
                             "velocity": velocity,
                             "norm_weight": args.norm_val}, fp)

            plot_nights(complete_ds,
                        model.net,
                        filename_prefix=f"{args.model}", n_steps=100, sobol_power=7,
                        include_vel=True,
                        plot_advected=False, simulate_paths=args.model == "lflow", mask=True)
        net = model.net


    else:
        if args.model == "dfnn":
            raise NotImplementedError("Loading of DFNNs does not yet work..")
        checkpoint = torch.load(CHECKPOINT_PATH)
        # plot_radar_positions(complete_ds, suffix=args.dataset)
        train_dataloader = generate_dataloader("final_train", train_params["mb_size"], seed=train_params["SEED"])

        net: DensityVelocityInterface = build(train_params["model"], model_params=model_params,
                                              complete_dataset=complete_ds,
                                              train_dataset=train_dataloader.dataset, load=CONTINUE_TRAIN)
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        net.eval()
        print("Plot nights..")
        plot_nights(complete_ds,
                    net,
                    filename_prefix=f"{args.model}", n_steps=100, sobol_power=7,
                    include_vel=True,
                    plot_advected=False, simulate_paths=args.model == "lflow", mask=True)

        ## evaluate consistency
        consistency_grid_sidelen = 50
        consistency_grid_timesteps = 10
        XY_3035_km, lat_Y, lon_X = get_target_grid(complete_ds, consistency_grid_sidelen)
        hour_ranges, night_begins, time_conditions = get_time_conditions(XY_3035_km, complete_ds)
        cds = complete_ds
        Z_3035 = np.random.rand(XY_3035_km.shape[0], 1) * (cds.extent_alt[1] - cds.extent_alt[0]) + \
                 cds.extent_alt[0] * cds.scale_space
        XYZ_3035_km = np.concatenate([XY_3035_km, Z_3035], -1)
        list_of_times = np.linspace(time_conditions[0][0] + 0.1 * time_conditions[0][0],
                                    time_conditions[-1][0], consistency_grid_timesteps).tolist()
        differences = net.evaluate_consistency_loss(XYZ_3035_km=XYZ_3035_km, t_reference=time_conditions[0],
                                                    list_of_times=list_of_times,
                                                    ode_split_size=8 * 4096 if args.model != "dfnn" else 512)

        print(f"Dumped results to '{RESULTS_PATH}'.")
        consistency_loss_path = f"results/{args.save_dir}/{args.model}/consistency_loss_{args.seed}.json"
        with open(consistency_loss_path, 'w') as fp:
            json.dump({"consistency_loss": str(differences)}, fp, sort_keys=True, indent=4)
        print(differences)
    #
    # plot_heatmap_differences(complete_ds,
    #                          net,
    #                          night_idx=2,
    #                          filename_prefix=f"{args.model}", n_steps=100, sobol_power=3)
    #
    # plot_nights(complete_ds,
    #             net,
    #             filename_prefix=f"{args.model}", n_steps=100, sobol_power=6,
    #             include_vel=False,
    #             plot_advected=True)
