"""
Download Radar derived bird density Data from Zenodo, parse it, reformat it, add a few useful features,
and then save it as a parquet file.
"""
import os
import platform
import subprocess

import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import glob
import gc
from datasets.bird_util import *

url_data_2018 = "https://zenodo.org/record/4587338/files/dc_zeno.zip?download=1"

tqdm.pandas()

ELEVATION_BINS = np.linspace(100, 4900, 25)

DATA_KEYS = ["dens", "ub", "vb",
             "bird", "insect",
             "ui", "vi",
             "sd_vvp", "volDir"]
DATA_TYPES = ["float32"] * len(DATA_KEYS)
RADAR_KEYS = ["name", "lon", "lat", "height", "heightDEM"]
RADAR_TYPES = [str, "float32", "float32", "int32", "int32"]
min_date = pd.to_datetime('2000-01-01')

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

DO_PREPROCESSING = True

# not used in this stub but often useful for finding various files

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[1]
data_raw_dir = project_dir / "data/radar_data_2018_3D/"


def download_and_preprocess_data(path_data_raw):
    vertical_profiles_path = path_data_raw / "vertical_profiles"
    download_vertical_profiles(path_data_raw, target_path=vertical_profiles_path)
    df_radar = parse_vertical_profiles_json(vertical_profiles_path)
    # df_radar.to_parquet(path_data_raw/"tmp_file.parquet")
    # gc.collect()
    # df_radar = pd.read_parquet(path_data_raw/"tmp_file.parquet")

    df_radar = preprocess_dataframe(df_radar)

    parquet_path = f"{path_data_raw}/bird_densities_3D_2018.parquet"
    print(f"Saving data to '{parquet_path}'")
    df_radar.to_parquet(parquet_path)

    # df_radar = pd.read_parquet(f"{data_raw_dir}/bird_densities_3D_2018.parquet")


def preprocess_dataframe(df_radar):
    df_radar = (df_radar
                # .assign(**{key: (lambda _df: _df[key].astype(float)) for key in DATA_KEYS})
                # .assign(lon=lambda x: x.lon.astype(float),
                #         lat=lambda x: x.lat.astype(float),
                #         height=lambda x: x.height.astype(int),
                #         heightDEM=lambda x: x.heightDEM.astype(int),
                #         radar_station=lambda x: x.radar_station.astype("category"),
                #         )
                .assign(time_hour=lambda x: x.time.dt.hour,
                        date_of_night=lambda x: pd.to_datetime(pd.to_datetime(x.time.dt.date) - (x.time.dt.hour <= 12) * timedelta(days=1)))
                # date_of_night=lambda x: pd.to_datetime(x.time.dt.date) - (x.time.dt.hour <= 12) * timedelta(days=1))
                .assign(night=lambda x: (x.loc[:, "date_of_night"] - min_date).dt.days)
                .pipe(project_coords_and_vel_df)
                )
    print("assigned..")
    df_radar = (df_radar
                .assign(x_3035=lambda _df: _df.x_3035 / 1_000,
                        y_3035=lambda _df: _df.y_3035 / 1_000,
                        z_km=lambda _df: (_df.elevation + _df.height) / 1_000,
                        dens = lambda _df: _df.dens * _df.bird,
                        # birds_per_km3=lambda _df: _df.dens,
                        u_3035=lambda _df: _df.u_3035 * (60 / 1_000),
                        v_3035=lambda _df: _df.v_3035 * (60 / 1_000),
                        )
                .drop(columns=["elevation", "height", "bird","insect",
                               "ui", "vi", "sd_vvp"])
                .rename(columns={"x_3035": "x_3035_km",
                                 "y_3035": "y_3035_km",
                                 "u_3035":"u_3035_km_per_min",
                                 "v_3035":"v_3035_km_per_min",
                                 "dens": "birds_per_km3"
                                 })
                .assign(time_minutes_total=lambda _df: (_df.time - _df.time.min()).dt.total_seconds() / 60.0))

    print("Grouping stuff")
    nightwise_min_minutes = df_radar.groupby(["radar_station", "night"])["time_minutes_total"].transform('min')
    df_radar["time_minutes_night"] = df_radar["time_minutes_total"] - nightwise_min_minutes
    del nightwise_min_minutes
    df_radar["radar_station"] = df_radar["radar_station"].astype("category")
    float64_cols = list(df_radar.select_dtypes(include='float64'))
    df_radar[float64_cols] = df_radar[float64_cols].astype("float32")
    int64_cols = list(df_radar.select_dtypes(include='int64'))
    df_radar[int64_cols] = df_radar[int64_cols].astype("int32")
    return df_radar


def parse_vertical_profiles_json(vertical_profiles_path) -> pd.DataFrame:
    vp_files = glob.glob(f"{vertical_profiles_path}/dc*.json")
    df_times = parse_times(vertical_profiles_path)
    radar_df_list = []
    for vp_file in tqdm(vp_files):
        df_single_radar = parse_radar_json(vp_file=vp_file, df_times=df_times)
        df_single_radar = df_single_radar.dropna(subset=["dens"])
        radar_df_list.append(df_single_radar)
    df_radar = pd.concat(radar_df_list, axis=0)
    del radar_df_list
    df_radar.insert(0, 'radar_station', df_radar.pop('name'))

    return df_radar


def parse_radar_json(vp_file, df_times) -> pd.DataFrame:
    with open(vp_file, "rb") as f:
        raw_json = json.load(f)

    pandas_list = []
    for key, dtype in zip(DATA_KEYS, DATA_TYPES):
        tmp = raw_json[key]
        tmp_df_wide = pd.DataFrame({f"{key}{int(elevation)}": np.array(vals).astype(dtype)
                                    for elevation, vals in zip(ELEVATION_BINS, tmp)})

        tmp_df_long = pd.wide_to_long(tmp_df_wide.reset_index(),
                                      i="index", j="elevation",
                                      stubnames=key)  # .reset_index().drop(columns="index")
        pandas_list.append(tmp_df_long)
    df_joined = (pd.concat(pandas_list, axis=1).reset_index()
                 .join(df_times.reset_index(), how="left", on="index", lsuffix="_left")
                 .drop(columns="index_left")
                 .rename(columns={"index": "time_index"}))

    for key, val, dtype in zip(RADAR_KEYS,
                               np.array([raw_json[key] for key in RADAR_KEYS]),
                               RADAR_TYPES):
        df_joined[key] = val.astype(dtype)

    df_joined = (df_joined.assign(
        name = lambda x: x.name.astype("category"),
    #     lon=lambda x: x.lon.astype("float32"),
    #     lat=lambda x: x.lat.astype("float32"),
    #     height=lambda x: x.height.astype("int32"),
    #     heightDEM=lambda x: x.heightDEM.astype("int32"),
    ))
    return df_joined


def parse_times(vertical_profiles_path) -> pd.DataFrame:
    time_file = Path(vertical_profiles_path / "time.json")
    assert time_file.exists(), f"'{time_file}' does not exist."
    with open(time_file, "rb") as f:
        # times = np.array()
        df_times = pd.DataFrame(dict(time=pd.to_datetime(json.load(f),
                                                         format="%d-%b-%Y %H:%M:%S")))
        df_times.index.name = "index"
    return df_times


def download_vertical_profiles(path_data_raw, target_path) -> None:
    # res = urllib.request.urlretrieve(url_data_2018, path_data_raw / "vertical_profiles.json")
    vertical_profiles_zipped_path = path_data_raw / "vertical_profiles.zip"
    os.makedirs(target_path, exist_ok=True)
    is_empty_dir = os.listdir(target_path)
    if platform.system() != 'Linux':
        logging.info(f"This script only supports Linux, as it uses wget to download the data. "
                     f"If you are not on Linux, please manually download the zip from the following URL: \n {url_data_2018}  ")
        exit()
    # download zip
    if not vertical_profiles_zipped_path.exists():
        logging.info("Downloading vertical profiles from Zenodo. ")
        res = subprocess.call(["wget", "-O", vertical_profiles_zipped_path, url_data_2018])
    # extract zip
    logging.info("Extracting vertical profiles from .zip file. ")
    res = subprocess.call(["unzip", vertical_profiles_zipped_path, "-d", target_path])


if __name__ == '__main__':
    download_and_preprocess_data(data_raw_dir)
