from matplotlib import pyplot as plt
import numpy as np

def quiver_skip(x, y, u, v, grid_shape, skip=2, ax=None, **kwargs):
    x_skipped, y_skipped, u_skipped, v_skipped = [element.reshape(grid_shape)[::skip, ::skip] for element in
                                                  [x, y, u, v]]
    if ax is None:
        ax = plt
    ax.quiver(x_skipped, y_skipped, u_skipped, v_skipped, **kwargs)

def stream_skip(x, y, u, v, grid_shape, skip=2, ax=None, **kwargs):
    x_skipped, y_skipped, u_skipped, v_skipped = [element.reshape(grid_shape)[::skip, ::skip] for element in
                                                  [x, y, u, v]]

    speed = np.sqrt(u_skipped**2+v_skipped**2)
    lw = 1.* speed / speed.max() + .1
    if ax is None:
        ax = plt
    ax.streamplot(x_skipped, y_skipped, u_skipped, v_skipped, density=0.6, linewidth=lw, **kwargs)
