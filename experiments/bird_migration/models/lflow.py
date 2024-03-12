import numpy as np
import torch
from cartopy import crs as ccrs

from datasets.bird_data import BirdDatasetMultipleNightsLeaveoutEW as BirdData
from datasets.bird_util import project_coords
from experiments.bird_migration.models.util import device, build_scale_to_domain_transform, ScaledResidualNet
from LFlow import LagrangeFlow
from experiments.bird_migration.models.template import DensityVelocityInterface

from enflows.distributions import StandardNormal
from enflows.transforms import *
from enflows.utils.torchutils import np_to_tensor, tensor_to_np
from enflows.nn.nets import *
from enflows.transforms.lipschitz import iResBlock, LipschitzDenseNetBuilder, LipschitzFCNNBuilder



class BirdLagrangeFlow(LagrangeFlow, DensityVelocityInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._num_evals = 0
        self._e = None

    def data_independent_loss(self, **hparams):
        if hparams.get("tp_weight", 0.) > 0.:
            tp_penalty = hparams.get("tp_weight") * self.transport_cost_penalty(n_z_samples=100,
                                                                                n_t_samples=10,
                                                                                forward_mode=True)
        else:
            tp_penalty = 0.
        return hparams.get("norm_weight") * self.global_density_penalty() \
            + tp_penalty





def build_lflow_densenet(num_layers, context_features, hidden_features_shared,
                         complete_dataset: BirdData,
                         **kwargs) -> LagrangeFlow:
    densenet_builder = LipschitzDenseNetBuilder(input_channels=3,
                                                densenet_depth=5,
                                                context_features=context_features,
                                                activation_function=CSin(w0=10),
                                                # activation_function=CLipSwish(),
                                                lip_coeff=.97,
                                                n_lipschitz_iters=5
                                                )

    base_dist = StandardNormal(shape=[3])  # ,
    transforms = []
    scale_to_domain_transform = build_scale_to_domain_transform(mins=complete_dataset.mins,
                                                                maxs=complete_dataset.maxs,
                                                                max_abs_transformed=0.1
                                                                )
    transforms.append(scale_to_domain_transform)
    transforms.append(InverseTransform(Tanh()))
    activation = torch.nn.functional.silu
    for i in range(num_layers):
        transforms.append(ActNorm(features=3))
        transforms.append(iResBlock(densenet_builder.build_network(),
                                    brute_force=True,
                                    # unbiased_estimator=True,
                                    # time_nnet=TimeNetwork(context_features),
                                    # n_exact_terms=1,
                                    # n_samples=1
                                    ))

    transform = CompositeTransform(transforms)
    embedding_net = ScaledResidualNet(in_features=1,
                                      out_features=context_features,
                                      hidden_features=hidden_features_shared,
                                      num_blocks=2,
                                      activation=activation,
                                      min_time=complete_dataset.df[complete_dataset.selected_time].min(),
                                      max_time=complete_dataset.df[complete_dataset.selected_time].max(),
                                      )
    flow = BirdLagrangeFlow(transform, base_dist, embedding_net,
                            init_log_scale_norm=kwargs.get("init_log_scale_norm"),
                            flexible_norm=True,
                            ).to(device)

    flow.max_time = 2_000
    return flow



def z_to_lat_lon(model, time, zs_sampled):
    ts = np_to_tensor(np.ones((zs_sampled.shape[0], 1)) * time, device=device)
    xs_sampled = model.z_to_x(zs_sampled, ts)
    xs = tensor_to_np(xs_sampled)
    #     # zs = tensor_to_np(model.sample(100, context=torch.ones(1, 1).to(device) * cur_time)).squeeze()
    lon, lat = project_coords(xs[:, [0]] * 1_000, xs[:, [1]] * 1_000, src_proj=ccrs.epsg(3035),
                              target_proj=ccrs.PlateCarree())
    return lat, lon, xs
