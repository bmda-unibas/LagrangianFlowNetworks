from experiments.bird_migration.models.MLP import build_MLP
from experiments.bird_migration.models.lflow import build_lflow_densenet
from experiments.bird_migration.models.PINN import build_PINN
from experiments.bird_migration.models.DFNN import build_DFNN
from experiments.bird_migration.models.SLDA import build_SLDA

builders = {"mlp": build_MLP,
            "lflow": build_lflow_densenet,
            "pinn": build_PINN,
            "dfnn": build_DFNN,
            "slda": build_SLDA
            }


def build(model: str, model_params: dict, **kwargs):
    return builders[model](**model_params, **kwargs)
