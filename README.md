# Lagrangian Flow Networks (LFlows)

Code corresponding to the "Lagrangian Flow Networks for Conservation Laws" paper published in ICLR24.
[Openreview Link](https://openreview.net/forum?id=Nshk5YpdWE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))


```
@inproceedings{
torres2024lagrangian,
title={Lagrangian Flow Networks for Conservation Laws},
author={Fabricio Arend Torres and Marcello Massimo Negri and Marco Inversi and Jonathan Aellen and Volker Roth},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Nshk5YpdWE}
}
```


## Setting up the Environment.
The environment set-up was tested with [conda 23.7.3](https://docs.conda.io/projects/miniconda/en/latest/),
using the relatively new [libmamba solver](https://www.anaconda.com/blog/conda-is-fast-now).
Older conda versions could take longer, but should in principle also work.
We assume cuda is available, and did not explicitely test a CPU setup.

The `.yml` file for the environment is given in `conda_env.yml` in the base directory via.
Aside from general package requirements, creating the environment as follows direc[plot_radar_stations.py](experiments%2Fbird_migration%2Fplot_radar_stations.py)tly installs the local `enflows_extended` package with pip,
and installs the experiments in development mode. 


```
(base) /lflows_iclr24  conda env create --file conda_env.yml
(base) /lflows_iclr24  conda activate lflows_neurips
# This code should then work:
(lflows_iclr24) /lflows_iclr24  python examples/lflows_conditional_moons.py
```

If you do not create the environment from the base directory, the pip install of the local package will not work.
In that case, you can try fixing it by running afterwards:

`(lflows_iclr24) /lagrangian_flow_net$ pip install nflows_extended`
`(lflows_iclr24) /lagrangian_flow_net$ pip install -e .`

Ideally, after a successfull install you should  be able to run and pass the unit tests with:
` 
(lflows_neurips) /lagrangian_flow_net$  pytest
`

## A very basic example of LFlows

For a basic example of how to use the LFlows code, you can run and take a look at a conditional two moons experiment:

`(lflows_neurips) /lagrangian_flow_net$  python examples/lflows_conditional_moons.py`

## Experiments: Simulated Fluid Flow
The simulated data experiments are in the directory `experiments/gaussians/`, with the filenames indicating the model and setting.
The code for the data creation is in `datasets/moving_gaussians.py`. 

Please be aware that we ran the experiments with multiple different random seeds.
An individual run of the respective experiments might not necessarily be representative.

The experiments in the code were performed with seeds 1234, 1235, ..., 1243. 

## Experiments: Dynamical Optimal Transport
The dynamical optimal transport experiments are in the directory `experiments/optimal_transport/`.
The LFlow code is given in `ot_2d.py`, the reference solution with a discrete OT solver in `ot_discrete.py`.
The DFNN and ICNN code is in the `DFNN` directory.

The experiments and plots in the code were performed with the default seed 1234. 
The results should remain stable with different seeds.

## Experiments: Bird Migration Modeling
The Bird Migration Modeling experiment can be run with `experiments/bird_data/bird_experiment.py`.
Note, that the script will prompt for downloading and preprocessing the data.
As we did not optimize this step, the preprocessing needs a lot of RAM (easily more than 10GB), and could crash small machines.

The code for data downloading and preprocessing can be found at `datasets/bird_data.py` and `datasets/download_radar_data.py`.

The experiments in the code were performed with seeds 301, 302, ..., 315. 

# About the Code & Library
The code consists of two main packages, the `LFlow` directory, and the `extended_nflows` directory.
The `Lflow` folder contains the main logic required for Lagrangian Flow Networks, i.e. the density and velocity computation,
given bijective networks.

The `extended_nflows` package contains the bijective layers.
Quite some of the core code for the bijective layers is based on the [nflows package](https://github.com/bayesiains/nflows).
`nflows` is a collection of [normalizing flows](https://arxiv.org/abs/1912.02762)
using [PyTorch](https://pytorch.org).
The main changes compared to the nflows repository mainly concern the addition of bijective layers.

The added layers with the `extended_nflows` compared to the current version of nflows are roughly as follows:
- Lipschitz-constrained invertible networks: Deep Residual Flows, invertible dense nets.
  Includes various variations for estimating the log determinant, as well as for enforcing and estimating the lipschitz constant.
  Importantly, we provide an implementation that allows to use them as conditional bijective layers.
- Sum-of-Sigmoid Transformations. Available as unconditional, conditional, and masked autoregressive. (tested)
- Transformations without an analytical Inverse: 
  - [Planar Flow](https://arxiv.org/abs/1912.02762) (tested), 
  - [Sylvester Flow](https://arxiv.org/abs/1803.05649) (untested)
- Conditional Versions of existing non-conditional transformations from nflows. Can be found for imports at `nflows.transforms.conditional.*`:
    - Planar Flow, Sylvester Flow
    - LU Transform
    - Orthogonal Transforms based on parameterized Householder projections
    - SVD based on the Orthogonal transforms
    - Shift Transform
- Conditional Versions of existing auto-regressive Variations, i.e. getting rid of the autoregressive parts.
    - [ConditionalPiecewiseRationalQuadraticTransform](https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html)
    - [ConditionalUMNNTransform](https://arxiv.org/abs/1908.05164)
