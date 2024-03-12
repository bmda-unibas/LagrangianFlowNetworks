"""
Based on https://github.com/bayesiains/nsf/blob/master/data/base.py
"""


from .plane import *



def load_plane_dataset(name, num_points, flip_axes=False):
    """Loads and returns a plane dataset.

    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        return {
            'gaussian': GaussianDataset,
            'crescent': CrescentDataset,
            'crescent_cubed': CrescentCubedDataset,
            'sine_wave': SineWaveDataset,
            'abs': AbsDataset,
            'sign': SignDataset,
            'four_circles': FourCircles,
            'diamond': DiamondDataset,
            'two_spirals': TwoSpiralsDataset,
            'checkerboard': CheckerboardDataset,
            "eight_gaussians": EightGaussianDataset,
            'two_circles': TwoCircles,
            'two_moons': TwoMoonsDataset,
            'pinwheel': PinWheelDataset,
            'swissroll': SwissRollDataset
        }[name](num_points=num_points, flip_axes=flip_axes)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


