import pandas as pd
import numpy as np
import cartopy.crs as ccrs

from typing import *


def project_coords(lon: np.ndarray, lat: np.ndarray, u: Optional[np.ndarray] = None, v: Optional[np.ndarray] = None,
                   src_proj: ccrs.Projection = ccrs.PlateCarree(),
                   target_proj: ccrs.Projection = ccrs.epsg(3035)):
    """
    Project coordinates lon, lat and optionall vectors u, v into another target projection.
    Default is PlateCarree -> EPSG:3035 (i.e. lon/lat -> europe-specific projection).
    :param lon:
    :param lat:
    :param u:
    :param v:
    :param src_proj:
    :param target_proj:
    :return:
    """
    # function returns ndarray of shape (:, 1, 3),, where the 3rd coordinate is z, which is 0 for us
    x, y = target_proj.transform_points(src_proj, lon, lat)[:, 0, :-1].T
    if u is not None and v is not None:
        u_proj, v_proj = target_proj.transform_vectors(src_proj, lon, lat, u, v)

        return x, y, u_proj, v_proj

    return x, y


def project_coords_df(df: pd.DataFrame) -> pd.DataFrame:
    x_3035, y_3035 = project_coords(df.lon.values.reshape(-1, 1),
                                    df.lat.values.reshape(-1, 1))
    return df.assign(x_3035=x_3035,
                     y_3035=y_3035)


def project_coords_and_vel_df(df: pd.DataFrame) -> pd.DataFrame:
    x_3035, y_3035, u_3035, v_3035 = project_coords(df.lon.values.reshape(-1, 1),
                                    df.lat.values.reshape(-1, 1),
                                    u=df.ub.values.reshape(-1, 1),
                                    v=df.vb.values.reshape(-1, 1))
    return df.assign(x_3035=x_3035,
                     y_3035=y_3035,
                     u_3035=u_3035,
                     v_3035=v_3035
                     )


import sys


def query_yes_no(question, default="yes"):
    """
    https://code.activestate.com/recipes/577058/

    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
