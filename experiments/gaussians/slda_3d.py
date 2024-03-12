"""
Use trick for parallel odesolve from https://github.com/facebookresearch/neural_stpp/blob/main/diffeq_layers/basic.py
"""
import sys
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
from datasets.moving_gaussians import MovingGaussians3d as MovingGaussiansCombined
from pprint import pprint

from experiments.gaussians.util.exp_gaussians_util import plot_density_3d as plot_density, \
    device, eval_and_log_model

from LFlow import SemiLagrangianFlow, transformed_mse
from enflows.CNF import neural_odes

import optuna
import logging
from enflows.utils.torchutils import np_to_tensor, set_seeds
from experiments.gaussians.density_velocity_interface import DensityVelocityInterface
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--dtype', type=str, default="float32")

parser.add_argument('--no-plots', dest='viz', action='store_false')

parser.add_argument('--niters', type=int, default=3000)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--num_samples', type=int, default=4096)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--results_dir', type=str, default="./results")
parser.add_argument('--load', action='store_true')
parser.add_argument('--name', type=str, default="cnf_model.pt")
# parser.add_argument('--no-optuna', action='store_true')
parser.add_argument('--optuna', dest="no_optuna", action='store_false')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--test', action='store_true')

#####################
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])

SOLVERS = ["dopri5", "dopri8", "bosh3", "fehberg2", "adaptive_heun"]

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)

parser.add_argument(
    "--nonlinearity", type=str, default="tanh", choices=["tanh", "relu", "softplus", "elu", "swish"]
)

# parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
######################
args = parser.parse_args()

options = f"{args.solver}/rtol{args.rtol:.0E}_atol{args.atol:.0E}"
study_name = f"SLDA_3D"
storage_name = "sqlite:///optuna/{}.db".format(study_name)
base_path = f"./results/{study_name}/{options}"
model_path = "./saved_models/" + study_name + f"_{args.seed}.pt"

os.makedirs(base_path, exist_ok=True)

data_gen = MovingGaussiansCombined(num_timesteps=21)

TORCH_DTYPES = {
    'float32': torch.float32,
    'float64': torch.float64
}

# dtype = torch.cuda.FloatTensor
dtype = TORCH_DTYPES[args.dtype]  # torch.float64
dtype_long = torch.cuda.LongTensor

torch.set_default_dtype(dtype)
set_seeds(args.seed)

device = torch.device('cuda:' + str(args.gpu)
                      if torch.cuda.is_available() else 'cpu')
print(device)


class SemiLagrangianNODE(SemiLagrangianFlow, DensityVelocityInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, dim=3)

    def transform_to_basegrid_range(self, x):
        return x / 6

    def boundary_loss(self, n_samples=4096):
        # basically a slip-boundary so nothing leaves the boundary and density=0 is then indirectly enforced
        tx_samples = self.sample_input_domain(n_samples)
        tx_samples_i = [tx_samples[i::6, ...] for i in range(6)]

        tx_samples_i[0][:, 1] = -4
        tx_samples_i[1][:, 1] = 4

        tx_samples_i[2][:, 2] = -4
        tx_samples_i[3][:, 2] = 4

        tx_samples_i[4][:, 3] = -4
        tx_samples_i[5][:, 3] = 4

        vels = [self.velocity(tx_sample[..., 1:], tx_sample[..., [0]]) for tx_sample in tx_samples_i]
        loss = (vels[0][:, 0] ** 2).mean() + (vels[1][:, 0] ** 2).mean() + \
               (vels[2][:, 1] ** 2).mean() + (vels[3][:, 1] ** 2).mean() + \
               (vels[4][:, 2] ** 2).mean() + (vels[2][:, 1] ** 2).mean()

        return loss

    def sample_input_domain(self, n_samples: int):
        data_x, data_t, _, __ = data_gen.sample_data(n_samples, t_subset="all", subset="all")
        tx_samples = np_to_tensor(np.concatenate([data_t[..., None], data_x], -1), device=device, dtype=torch.float32)
        return tx_samples.detach()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    hparams_no_optuna = dict(
        hidden_dims_velocity=int(np.log2(64)),
        layers_velocity=2,
        lr=0.003293887034400879,
        lr_grid=5e-2,
        lbfgs=True,
        grid_len=50
    )
    if args.no_optuna and not args.load:
        trial = optuna.trial.FixedTrial(hparams_no_optuna)
        objective(trial, save_model=True)

    elif not args.load:
        try:
            study: optuna = optuna.create_study(study_name=study_name, storage=storage_name,
                                                load_if_exists=not args.overwrite,
                                                direction="maximize")
        except optuna.exceptions.DuplicatedStudyError:
            optuna.delete_study(study_name=study_name, storage=storage_name)
            study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=False,
                                        direction="maximize")
        if args.test:
            hparams_default = study.best_trial.params
            # run best setting with higher precision
            trial = optuna.trial.FixedTrial(hparams_default)
            objective(trial, save_model=True)

        else:
            default_params = dict(
                hidden_dims_velocity=int(np.log2(256)),
                layers_velocity=3,
                lr=1e-3
            )
            study.enqueue_trial(default_params, skip_if_exists=True)
            study.optimize(objective, n_trials=20, gc_after_trial=True)
    else:

        model = torch.load(model_path)

        # small hack, some stored models dont have this attribute ..
        if not hasattr(model.base_grid, "padding_mode"):
            model.base_grid.padding_mode = "border"

        model.cnf_to_t0.odeint_kwargs["test"]["rtol"] = args.rtol
        model.cnf_to_t0.odeint_kwargs["test"]["atol"] = args.atol
        model.cnf_to_t0.odeint_kwargs["test"]["method"] = args.solver
        model.eval()

        with torch.no_grad():
            plot_density(model, 10000, title="slda", base_path=base_path, lim=4 - 1e-3, device=device,
                         condition=np.arange(0., 1.1, 0.25), include_vel=True, vmin=0, vmax=0.46)
        # model = SemiLagrangianNODE(odefunc_base=None, odefunc_flow=None, solver="dopri5", atol=1e-5, rtol=1e-5)
        # model.load_state_dict(torch.load(model_path))

        if args.test:
            eval_and_log_model(hparams_no_optuna, model, args.seed, study_name, data_gen)

        # print(f"Val:  {data_gen.score_density(model, subset='val', device=device):.2%} ")
        # print(f"Test:  {data_gen.score_density(model, subset='test', device=device):.2%}")


def objective(trial: optuna.trial.Trial, save_model=False):
    hparams = dict(
        hidden_dims_velocity=2 ** trial.suggest_int("hidden_dims_velocity", low=5, high=9),
        layers_velocity=trial.suggest_int("layers_velocity", low=2, high=5),
        lr=trial.suggest_float("lr", 1e-5, 1e-2),
        lr_grid=trial.suggest_float("lr_grid", 1e-5, 1e-1),
        do_lbfgs=trial.suggest_categorical("lbfgs", [True, False]),
        grid_len=trial.suggest_int("grid_len", low=10, high=100, step=10),
        rtol=args.rtol if not args.test else 1e-05,
        atol=args.atol if not args.test else 1e-05)

    odenet_flow = neural_odes.ODEnet(
        hidden_dims=tuple([hparams["hidden_dims_velocity"]] * hparams["layers_velocity"]),
        input_shape=(3,),
        strides=None,
        conv=False,
        layer_type="concat_v2",
        nonlinearity=args.nonlinearity,
    )

    model = SemiLagrangianNODE(odenet_flow=odenet_flow, solver=args.solver,
                               rtol=hparams["rtol"],
                               atol=hparams["atol"],
                               divergence_fn="approximate",
                               grid_len=hparams["grid_len"]
                               ).to(device)
    pprint(hparams)
    # optimizer = optim.Adam(model.parameters(), lr=hparams["lr"])

    optimizer = optim.Adam([{"params": model.base_grid.parameters(), "lr": hparams["lr_grid"]},
                            {"params": model.cnf_to_t0.parameters()},
                            ], lr=hparams["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=3_000)
    loss_meter = RunningAverageMeter()
    loss_meter_vel = RunningAverageMeter()
    print("initializing")
    # init_model_gp(data_gen, model)
    print("training")
    train_model(data_gen, loss_meter, loss_meter_vel, model, optimizer, scheduler, base_path=base_path)
    if args.test:
        eval_and_log_model(hparams, model, args.seed, study_name, data_gen)

    print('Training complete.')

    if save_model:
        print(f"saving file to: {model_path}")
        torch.save(model, model_path)

    return data_gen.score_density(model, subset="val", device=device)


def train_model(data_gen, loss_meter, loss_meter_vel, model: SemiLagrangianNODE, optimizer, scheduler, base_path,
                do_lbfgs=True):
    if do_lbfgs:
        # L-BFGS
        lbfgs = optim.LBFGS(model.parameters(),
                            history_size=10,
                            max_iter=20,
                            line_search_fn="strong_wolfe")

        def get_closure():

            train_data_list = data_gen.sample_data(n_samples=8 * args.num_samples, subset="train",
                                                   t_subset="t0")
            data_x, data_t, logpdf, vel = [np_to_tensor(val, device=device).reshape(8 * args.num_samples, -1)
                                           for val in train_data_list]

            def closure():
                lbfgs.zero_grad()
                objective = lbfgs_loss(data_t, data_x, logpdf, model, vel)
                objective.backward()
                return objective

            return closure

        closure = get_closure()
        try:

            for i in tqdm(range(5)):
                lbfgs.step(closure)

                train_data_list = data_gen.sample_data(n_samples=4 * args.num_samples, subset="train",
                                                       t_subset="discrete")
                data_x, data_t, logpdf, vel = [np_to_tensor(val, device=device).reshape(4 * args.num_samples, -1)
                                               for val in train_data_list]
                print(lbfgs_loss(data_t, data_x, logpdf, model, vel).item())

        except KeyboardInterrupt:
            print("Aborted pretraining.")

    print("beginning SGD")
    try:
        for itr in range(1, args.niters + 1):

            train_data_list = data_gen.sample_data(n_samples=args.num_samples, subset="train",
                                                   t_subset="discrete")
            data_x, data_t, logpdf, vel = [np_to_tensor(val, device=device).reshape(args.num_samples, -1)
                                           for val in train_data_list]

            logp_x = model.log_density(data_x, data_t)
            vel_pred = model.velocity(x=data_x, t=data_t)
            radius = (data_x ** 2).sum(-1)

            mask = (radius > 0).unsqueeze(-1)

            density_loss = transformed_mse(logp_x.squeeze(), logpdf.squeeze())
            vel_loss = torch.nn.functional.mse_loss(mask * vel_pred.squeeze(), mask * vel.squeeze())
            loss = 100 * density_loss + (5e-2 * vel_loss) + 1e-2 * model.boundary_loss(1024) + (
                model.base_grid.base_grid.exp()).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss_meter.update(density_loss.detach().item())
            loss_meter_vel.update(vel_loss.detach().item())
            estimated_r2_log1prho = loss_meter.avg  # 1 - loss_meter.avg / torch.nn.functional.softplus(logpdf).std()
            estimated_r2_vel = 1 - loss_meter_vel.avg / vel.std()
            if itr % 50 == 0:
                print(
                    'Iter: {}, running avg loss: {:.2E}, vel loss R2: {:.2%}'.format(itr,
                                                                                     estimated_r2_log1prho,
                                                                                     estimated_r2_vel))
            if (itr + 1) % 50 == 0:
                model.eval()
                r2_density = data_gen.score_density(model, device=device, n_space=1_000, n_times=5, split_size=5_000,
                                                    subset="val")
                # r2_vel = data_gen.score_velocity(model, device=device, n_samples=100_00)
                print(f"(Val R2) Density: {r2_density:.3%}")
                model.train()

            if (itr + 1) % 50 == 0:
                if args.viz:
                    model.eval()
                    with torch.no_grad():
                        plot_density(model, itr, title="slda", base_path=base_path, lim=4 - 1e-3, device=device,
                                     condition=np.arange(0., 1.1, 0.25), include_vel=True, vmin=0, vmax=0.46)
                    #
                    # with torch.no_grad():
                    #     cons_loss = model.evaluate_consistency_loss(data_gen)
                    # print(f"Consistency Loss: {cons_loss}")
                    model.train()
    except KeyboardInterrupt:
        print("Aborted training.")

    if args.viz:
        model.eval()
        with torch.no_grad():
            plot_density(model, itr, title="slda", base_path=base_path, lim=4 - 1e-3, device=device,
                         condition=np.arange(0., 1.1, 0.25), include_vel=True, vmin=0, vmax=0.46)
        model.train()


def lbfgs_loss(data_t, data_x, logpdf, model, vel):
    logp_x = model.log_density(data_x, data_t)
    density_loss = transformed_mse(logp_x.squeeze(), logpdf.squeeze())
    loss = 100 * density_loss + 2e-1 * (model.base_grid.base_grid.exp()).mean()
    return loss


if __name__ == '__main__':
    main()
