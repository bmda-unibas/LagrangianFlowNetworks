"""
Same code as bird experiment, but same MB size for everything to compare runtime and memory consumption.
"""
from datasets import bird_data
from datasets.bird_data import BirdDatasetMultipleNightsLeaveOutMiddle
import numpy as np
from lightning.pytorch.callbacks import LearningRateMonitor
from torch.nn.functional import softplus, mse_loss

from experiments.bird_migration.bird_utils import open_or_create_study, weighted_MSE, Timer
from experiments.bird_migration.models.model_builder import build
from experiments.bird_migration.plot_util import plot_nights
from experiments.bird_migration.models.template import DensityVelocityInterface
from experiments.bird_migration.models.MLP import VanillaNN

from models.DFNN import DivFreeNetwork
import lightning as L
from lightning.pytorch import loggers as pl_loggers

import torch
from torch.utils.data import DataLoader

import argparse
from enflows.utils.torchutils import set_seeds
from pprint import pprint
import json

import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vel_factor', type=float, default=1.)
parser.add_argument('--name', type=str, default="default")
parser.add_argument('--save_dir', type=str, default="experiment")
parser.add_argument('--start_date', type=str, default="2018-04-05")  # 2018-03-12
parser.add_argument('--end_date', type=str, default="2018-04-07")  # 2018-03-14

parser.add_argument('--load', action='store_true')
parser.add_argument('--no_plots', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--eval_time', action='store_true')
parser.add_argument('--model', type=lambda s: s.lower(), default="lflow",
                    choices=["lflow", "mlp", "pinn", "dfnn", "slda"])

parser.add_argument('--continue_train', action='store_true')
args = parser.parse_args()

BirdData = BirdDatasetMultipleNightsLeaveOutMiddle

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
set_seeds(args.seed)


if args.model == "lflow":  # or args.model == "dfnn":
    scale_vel_loss = (1. / BirdData.scale_time) * BirdData.scale_space
    BirdData.scale_space = 1.
    BirdData.scale_time = 1.
else:
    scale_vel_loss = 1.

complete_ds = BirdData(subset="all", start_date=args.start_date, end_date=args.end_date)
_train_ds = BirdData(subset="train", start_date=args.start_date, end_date=args.end_date)

CONTINUE_TRAIN = args.continue_train
PLOT_STUFF = (not args.no_plots) and args.test

assert torch.cuda.is_available(), "No CUDA available!"


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

        # don't know  how else to log training time in lightning
        self.automatic_optimization = False
        self.timer = Timer()
        self.logged_training_loss = []
        self.train_iter_times = []
        self.peak_memory_mb = []
        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        torch.cuda.reset_peak_memory_stats()

        self.timer.cont()
        opt.zero_grad()

        txyz, _rho, _uv, uv_mask, rho_mask = batch["txyz"], batch["rho"], batch["uv"], batch["uv_mask"], batch[
            "rho_mask"]

        rho = _rho[rho_mask].squeeze()
        uv = _uv[uv_mask]

        txyz_rho = txyz[rho_mask, ...]
        txyz_uv = txyz[uv_mask, ...]

        pred_uv = self.net.velocity(x=txyz_uv[..., 1:],
                                    t=txyz_uv[..., :1])[..., :-1]

        pred_logprob = self.net.log_density(x=txyz_rho[..., 1:],
                                            t=txyz_rho[..., :1]).squeeze()
        loss_rho = mse_loss(torch.log1p(rho), softplus(pred_logprob))
        loss_uv = weighted_MSE(uv, pred_uv, weights=scale_vel_loss * self.uv_balance)

        additional_loss = self.net.data_independent_loss(**self.loss_params)
        loss = loss_rho + additional_loss + self.loss_params["vel_weight"] * self.loss_params["vel_factor"] * loss_uv

        self.manual_backward(loss)
        opt.step()
        self.timer.stop()
        self.peak_memory_mb.append(torch.cuda.max_memory_allocated() / (1024 ** 2))

        self.train_iter_times.append(self.timer.time_counter)
        self.logged_training_loss.append(loss.detach().cpu().item())
        self.log("train_loss", loss)
        self.log("train_vel_loss", mse_loss(scale_vel_loss * uv, scale_vel_loss * pred_uv), prog_bar=True,
                 on_step=False, on_epoch=True)
        self.log("train_density_loss", loss_rho, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_additional_loss", additional_loss)

        return loss

    def configure_optimizers(self):
        if args.model == "dfnn":
            params = self.net.params
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
    epoch_times = []
    train_dataloader = generate_dataloader("final_train", train_params["mb_size"], seed=train_params["SEED"])
    test_dataloader = generate_dataloader("test", train_params["mb_size"], seed=train_params["SEED"])

    net = build(train_params["model"], model_params=model_params,
                complete_dataset=complete_ds,
                train_dataset=train_dataloader.dataset)

    model = BirdModel(net=net, train_params=train_params, loss_params=loss_params)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(max_epochs=train_params["num_iter"] if args.eval_time else None,
                        max_steps=-1 if args.eval_time else 10,
                        accelerator="cuda", devices=[0, ],
                        enable_progress_bar=True,
                        enable_model_summary=False,
                        # gradient_clip_val=1,
                        log_every_n_steps=10,
                        logger=pl_loggers.TensorBoardLogger(save_dir=args.save_dir,
                                                            # name="training_subset",
                                                            name=args.model,
                                                            sub_dir=f"{args.name}_{args.start_date}_{args.end_date}"),
                        callbacks=[lr_monitor],
                        inference_mode=False
                        )

    # timer = Timer()
    try:
        trainer.fit(model=model, train_dataloaders=train_dataloader)
        # timer.stop()
    except KeyboardInterrupt:
        # timer.stop()
        pass

    return model


def generate_dataloader(subset, mb_size, seed):
    bird_dataset = BirdData(subset=subset,
                            transform=bird_data.ToTensor(device),
                            seed=seed,
                            start_date=args.start_date,
                            end_date=args.end_date,
                            )
    sampler = torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.RandomSampler(bird_dataset),
                                                    batch_size=mb_size, drop_last=True)
    dataloader = DataLoader(bird_dataset, sampler=sampler, batch_size=None)
    return dataloader


def load_hyperparameters(args, SHARED_MB_SIZE):
    shared_train_params = {
        'model': args.model,
        'SEED': args.seed,
        'optimizer': 'Adam',
        'load': args.load,
        'overwrite': args.overwrite,
        'test': args.test,
    }

    # aside from cnf_lflow, the memory footprint and time per mb remains steady during training
    if SHARED_MB_SIZE is None:
        train_params = {
            "lflow": {'lr': 1e-2, "weight_decay": 2e-3, 'mb_size': 16384,
                      "num_iter": 300, **shared_train_params},
        "slda": {'lr': 1e-3, 'mb_size': 16384, "num_iter": 300, **shared_train_params, "weight_decay": 0.},
            "pinn": {'lr': 1e-3, "num_iter": 300, "weight_decay": 2e-3,  # "weight_decay": 5e-6,
                     'mb_size': 16384, **shared_train_params},
            "mlp": {'lr': 1e-3, "num_iter": 300,
                    "weight_decay": 1e-3,  # 5e-6,
                    'mb_size': 16384, **shared_train_params},
            "dfnn": {'lr': 1e-3, "num_iter": 300, "weight_decay": 0, 'mb_size': 1024 * 2, **shared_train_params},
        }
    else:
        SHARED_NUM_ITER = 1
        train_params = {
            "lflow": {'lr': 1e-2, "weight_decay": 2e-3, 'mb_size': SHARED_MB_SIZE,
                      "num_iter": SHARED_NUM_ITER, **shared_train_params},
        "slda": {'lr': 1e-3, 'mb_size': 16384, "num_iter": 300, **shared_train_params, "weight_decay": 0.},
            # 3e-2},
            "pinn": {'lr': 1e-3, "num_iter": SHARED_NUM_ITER, "weight_decay": 2e-3,  # "weight_decay": 5e-6,
                     'mb_size': SHARED_MB_SIZE, **shared_train_params},
            "mlp": {'lr': 1e-3, "num_iter": SHARED_NUM_ITER,
                    "weight_decay": 1e-3,  # 5e-6,
                    'mb_size': SHARED_MB_SIZE, **shared_train_params},
            "dfnn": {'lr': 1e-3, "num_iter": SHARED_NUM_ITER, "weight_decay": 0, 'mb_size': SHARED_MB_SIZE,
                     **shared_train_params},
        }
    shared_loss_params = {
        "vel_factor": args.vel_factor
    }

    loss_params = {
        "lflow": {
            'vel_weight': 3,
            'norm_weight': 1e-3,
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
            "pde_weight": 1e-3,
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

    dirname = "./results/benchmarks"
    os.makedirs(dirname, exist_ok=True)

    if not args.eval_time:
        for mb_size in [256, 512, 1024, 2048, 1024*3]:
            print(f"{mb_size=}")
            train_params, model_params, loss_params = load_hyperparameters(args, SHARED_MB_SIZE=mb_size)

            model = train_model(train_params, model_params, loss_params)

            # evaluate last epoch only
            min_time_per_mb = np.mean(model.train_iter_times[-mb_size:])
            # maximum for peak memory
            max_peakmemory = np.max(model.peak_memory_mb)
            res_dict = dict(avg_mb_time=min_time_per_mb, avg_mb_memoryinMB=max_peakmemory,
                            model=args.model, mb_size=mb_size, num_iter = train_params["num_iter"])

            pprint(res_dict)

            path = f"{dirname}/{args.model}_mb{mb_size}_memory.json"

            print(f"Dumped results to '{path}'.")
            with open(path, 'w') as fp:
                json.dump(res_dict, fp, sort_keys=True, indent=4)
    else:
        train_params, model_params, loss_params = load_hyperparameters(args, SHARED_MB_SIZE=None)
        model = train_model(train_params, model_params, loss_params)
        # evaluate last epoch only
        # maximum for peak memory
        max_peakmemory = np.max(model.peak_memory_mb)
        res_dict = dict(#avg_mb_time=min_time_per_mb,
                        model=args.model, mb_size=train_params["mb_size"], num_iter = train_params["num_iter"],
                        train_time=model.train_iter_times, train_loss = model.logged_training_loss)

        path = f"{dirname}/{args.model}_mb{train_params['mb_size']}_time.json"

        print(f"Dumped results to '{path}'.")
        with open(path, 'w') as fp:
            json.dump(res_dict, fp, sort_keys=True, indent=4)
