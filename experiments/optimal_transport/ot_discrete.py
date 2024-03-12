"""
https://github.com/facebookresearch/neural-conservation-law/blob/main/pytorch/solve_ot.py

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file  https://github.com/facebookresearch/neural-conservation-law/blob/main/LICENSE
Attribution-NonCommercial 4.0 International

"""
import os
import pickle

import numpy as np

import ot
import ot.plot
import pandas as pd

from data import load_2dtarget
import argparse
import torch
import random
import matplotlib.pyplot as plt
import pickle as pkl
from pathlib import Path
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--target0', type=str, default="circles")
parser.add_argument('--target1', type=str, default="pinwheel")
parser.add_argument('--nsamples', type=int, default=20_000)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--plot_samples', action='store_true')
parser.add_argument('--alpha', type=float, default = 0.01)

args = parser.parse_args()


def plot_path_at_t(samples0, samples1, G, t, thr=1e-8, ax=None, **kwargs):
    mx = G.max()
    assert 0. <= t <= 1.
    if ax is None:
        ax = plt
    # (1-t) * samples0.T + t * samples1
    points = ((1-t) * samples0[:, np.newaxis, :] + t * samples1[np.newaxis, :, :])[(G/mx)>thr]
    # point_list = []
    # for i in range(samples0.shape[0]):
    #     for j in range(samples1.shape[0]):
    #         if G[i, j] / mx > thr:
    #             x = (1-t) * samples0[i, 0] + t * samples1[j, 0]
    #             y = (1-t) * samples0[i, 1] + t * samples1[j, 1]
    #             point_list.append((x,y))

    ax.scatter(*points.T, marker='+', color="black", **kwargs)
    ax.set_aspect('equal')


def solve_ot(target0: str, target1: str, nsamples: int, seed: int, plot_samples:bool, alpha):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    target0 = load_2dtarget(target0)
    target1 = load_2dtarget(target1)
    name = f"{args.target0}_{args.target1}"

    wdists = []
    if not plot_samples:
        for i in tqdm(range(5)):
            samples0 = target0.sample(nsamples).double().cpu().numpy()
            samples1 = target1.sample(nsamples).double().cpu().numpy()

            prod0 = np.ones(nsamples) / nsamples
            prod1 = np.ones(nsamples) / nsamples

            M = ot.dist(samples0, samples1)
            wdist = ot.emd2(prod0, prod1, M, numItermax=5000000, numThreads=40, return_matrix=False)
            wdists.append(wdist)
            print(f"{i=} {wdist=:.4f}")

            mean = np.mean(wdists)
            std = np.std(wdists)
            df = pd.DataFrame({"Wasserstein Distance": wdists})
            df["method"] = "discrete OT"
            df["dataset"] = name
            print(f"final: {mean=:.4f} {std=:.4f}")

            target_path = f"./results/final/discrete_{name}.pkl"
            os.makedirs("./results/final", exist_ok=True)
            print(f"Saving to: '{target_path}'")
            with open(target_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        n_points = 20_000
        samples0 = target0.sample(n_points).double().cpu().numpy()
        samples1 = target1.sample(n_points).double().cpu().numpy()

        prod0 = np.ones(n_points) / nsamples
        prod1 = np.ones(n_points) / nsamples

        M = ot.dist(samples0, samples1)
        wdist, log = ot.emd2(prod0, prod1, M, numItermax=5000000, numThreads=40, return_matrix=True)
        wdists.append(wdist)

        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for t, ax in tqdm(zip(np.linspace(0, 1, 5), axs)):
            ax.imshow(np.zeros((100, 100)), origin='lower', extent=(-2,2,-2,2), alpha=0.)
            plot_path_at_t(samples0, samples1, G=log["G"], t=t, ax=ax, alpha=alpha)
            ax.set_xlim(-2,2)
            ax.set_ylim(-2,2)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            # ax.axis('off')
        plt.savefig(f"trajectories_{name}.png")


if __name__ == "__main__":
    solve_ot(**vars(args))
