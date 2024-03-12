import abc
import os

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

from datasets.download_radar_data import download_and_preprocess_data
from datasets.download_radar_data import query_yes_no, url_data_2018
from typing import *
import iteround

cur_dir = Path(os.path.dirname(__file__))


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device="cpu"):
        self.device = device

    def __call__(self, sample: OrderedDict):
        return {key: torch.from_numpy(val).to(self.device) for key, val in sample.items()}


class BirdDatasetSingleNight(Dataset):
    scale_time = 1. / (60 * 24)  # / 10
    scale_space = 1. / 1000

    def __init__(self, subset="train",
                 seed=1235,
                 transform=None):
        assert subset in ["train", "val", "test", "all", "final_train"]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.transform = transform

        self.root_dir = cur_dir.parent
        self.data_dir = self.root_dir / "datasets/data/radar_data_2018_3D"
        self.data_filename = "bird_densities_3D_2018.parquet"

        try:
            if not Path(self.data_dir / self.data_filename).exists():
                raise FileNotFoundError(f"{self.data_dir / self.data_filename} does not exist.")
            self._df = pd.read_parquet(self.data_dir / self.data_filename)
            self._df["log1p_density"] = np.log1p(self._df["birds_per_km3"])
        except FileNotFoundError as e:
            download_bird_data = query_yes_no(f"File '{str(self.data_dir / self.data_filename)}' not found.\n"
                                              f"Do you wish to download and preprocess it from '{url_data_2018}'?\n"
                                              f"Note, that this will require a lot of RAM (10GB+) for the pandas preprocessing and may take a few minutes.")
            print(download_bird_data)
            if download_bird_data:
                download_and_preprocess_data(self.data_dir)
                self._df = pd.read_parquet(self.data_dir / self.data_filename)
            else:
                raise e

        self.subset = subset
        self.df = self._split_train_val_test(self._filter(self._df), subset=subset)
        self.selected_time = "time_minutes_night"

        self.extent = [
            [self.df.x_3035_km.min(), self.df.x_3035_km.max()],
            [self.df.y_3035_km.min(), self.df.y_3035_km.max()],
            [self.df.z_km.min(), self.df.z_km.max()],
        ]
        self.mins = [self.df.x_3035_km.min(),
                     self.df.y_3035_km.min(),
                     # self.df.z_km.min()
                     0.
                     ]
        self.maxs = [self.df.x_3035_km.max(),
                     self.df.y_3035_km.max(),
                     self.df.z_km.max()
                     ]

        self.std_rho = self.df.birds_per_km3.std()
        self.std_log1p_rho = np.log1p(self.df.birds_per_km3.values).std()
        self.var_log1p_rho = np.log1p(self.df.birds_per_km3.values).var()
        self.std_vel_u = self.df.u_3035_km_per_min.std()
        self.std_vel_v = self.df.v_3035_km_per_min.std()

        # scale inputs
        self.df.loc[:, self.selected_time] *= self.scale_time
        self.df.loc[:, ["x_3035_km", "y_3035_km"]] *= self.scale_space

        # adjust outputs
        self.df.loc[:, ["u_3035_km_per_min", "v_3035_km_per_min"]] *= (1. / self.scale_time) * self.scale_space

        self.std_vel_uv = self.df.loc[:, ["u_3035_km_per_min", "v_3035_km_per_min"]].std()
        self.var_vel_uv = np.nanvar(self.df.loc[:, ["u_3035_km_per_min", "v_3035_km_per_min"]].values)
        self.num_points = self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.df.iloc[idx]
        txyz = item[self._txyz_column_names()].astype("float32").values  # / 10
        # scale unit km to 10 km
        rho = item[self._rho_column_name()].astype("float32").values
        uv = item[self._uv_column_names()].astype("float32").values
        night = item["night"].astype("float32").values

        mask = np.logical_not(np.logical_or.reduce(np.isnan(uv), -1))
        rho_mask = np.logical_not(np.logical_or.reduce(np.isnan(rho), -1))
        sample = OrderedDict(txyz=txyz, rho=rho, uv=uv, uv_mask=mask, rho_mask=rho_mask, night=night)

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.num_points

    def _txyz_column_names(self) -> List[AnyStr]:
        return [self.selected_time, "x_3035_km", "y_3035_km", "z_km"]

    @staticmethod
    def _rho_column_name() -> List[AnyStr]:
        return ["birds_per_km3"]

    @staticmethod
    def _uv_column_names() -> List[AnyStr]:
        return ["u_3035_km_per_min", "v_3035_km_per_min"]

    @abc.abstractmethod
    def _filter(self, df) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def _split_train_val_test(self, _df, subset) -> pd.DataFrame:
        pass

    @staticmethod
    def get_radar_lonlat(df):
        radars_train = df[df["lon"].notna()].dropna(subset=["lon"]).groupby(
            "radar_station").first().reset_index().dropna(subset="lon")[["radar_station", "lon", "lat"]]
        return radars_train


class BirdDatasetMultipleNights(BirdDatasetSingleNight):
    def __init__(self, start_date="2018-04-05", end_date="2018-04-07", *args, **kwargs):
        self.start_date = start_date
        self.end_date = end_date
        super().__init__(*args, **kwargs)
        self.extent_lat = self.df.lat.min() - 1.3, self.df.lat.max() + 1
        self.extent_lon = self.df.lon.min(), self.df.lon.max() + 1
        self.extent_alt = self.df.z_km.min(), self.df.z_km.max()
        self.extent_time = self.df.time_minutes_night.min(), self.df.time_minutes_night.max()

    def _filter(self, df) -> pd.DataFrame:
        df_subset = df.query(f"(date_of_night>='{self.start_date}') and (date_of_night<='{self.end_date}')")
        df_subset = df_subset.sort_values(by="time_minutes_total").copy()
        df_subset["time_minutes_total"] = df_subset["time_minutes_total"] - df_subset["time_minutes_total"].min()
        df_subset["night"] = df_subset["night"] - df_subset["night"].min()
        max_minutes_per_night = df_subset.groupby("night").time_minutes_night.max().cumsum()
        for i in df_subset.night.unique():
            if i > 0:
                df_subset.loc[df_subset.night == i, "time_minutes_night"] = df_subset.loc[
                                                                                df_subset.night == i, "time_minutes_night"] + \
                                                                            max_minutes_per_night[i - 1]  # + 120
        return df_subset

    def _split_train_val_test(self, _df, subset):
        _df["night_radar_index"] = _df.groupby(['radar_station', 'date_of_night']).ngroup().add(1)
        unique_radars = _df["night_radar_index"].unique()

        all_sizes = [ratio * len(unique_radars) for ratio in self.train_val_test_ratios]
        all_sizes = [int(val) for val in iteround.saferound(all_sizes, 0)]
        assert sum(all_sizes) == len(unique_radars)

        train_val_idx = self.rng.choice(np.arange(len(unique_radars)), size=all_sizes[0] + all_sizes[1], replace=False)
        test_idx = [idx for idx in np.arange(len(unique_radars)) if idx not in train_val_idx]

        train_idx = self.rng.choice(train_val_idx, size=all_sizes[0], replace=False)
        val_idx = [idx for idx in train_val_idx if idx not in train_idx]

        assert ((set(train_val_idx) & set(test_idx)) == set()) and ((set(train_idx) & set(val_idx)) == set())
        assert ((set(val_idx) & set(test_idx)) == set())

        radar_subset = {"train": _df.query("not radar_station.str.contains('de')"),
                        "val": _df.query("not radar_station.str.contains('de')"),
                        "test": _df.query("radar_station.str.contains('de')"),
                        "final_train": _df.query("not radar_station.str.contains('de')"),
                        "all": _df}[subset]

        return radar_subset


class BirdDatasetMultipleNightsLeaveOutMiddle(BirdDatasetMultipleNights):
    def _split_train_val_test(self, _df, subset):
        test_radars = ["frtra", "frabb", "frave", "frtro", "frbla", "frmtc",
                       "frnan", "bewid", "denhb", "deess",
                       "nlhrw", "nldhl", "defld", "deoft", "detur", "demem", "desna"]
        val_radars = ["frbou", "deneu", "deeis", "frniz"]

        train_radars = [radar for radar in _df.radar_station.unique() if
                        (radar not in test_radars) and
                        (radar not in val_radars)
                        ]
        df_train = _df.query("radar_station in @train_radars")
        df_val = _df.query("radar_station in @val_radars")
        df_test = _df.query("radar_station in @test_radars")

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        radar_subset = {"train": df_train,
                        "val": df_val,
                        "test": df_test,
                        "final_train": _df.query("radar_station in @val_radars or radar_station in @train_radars"),
                        "all": _df}[subset]
        return radar_subset


class BirdDatasetMultipleNightsLeaveoutEW(BirdDatasetMultipleNights):
    def __init__(self, *args, include_test_velocity=False, **kwargs):
        self.include_test_velocity = include_test_velocity
        super().__init__(*args, **kwargs)

    def _split_train_val_test(self, _df, subset):
        test_radars = ["deros", "depro", "dedrs", "deeis", "desna", "deneu", "deumd", "deboo", "dehnr", "defld",
                       "detur", "demem", "frmom", "frbor", "frtou", "frgre", "frmcl", "frche", "frtre", "frpta"]
        val_radars = test_radars

        train_radars = [radar for radar in _df.radar_station.unique() if
                        (radar not in test_radars) and
                        (radar not in val_radars)
                        ]
        df_val = _df.query("radar_station in @val_radars")
        df_test = _df.query("radar_station in @test_radars")
        if self.include_test_velocity:
            df_train = _df.query("radar_station in @train_radars").copy()
            df_val = _df.query("radar_station in @val_radars").copy()
            df_test = _df.query("radar_station in @test_radars").copy()
            df_tmp = df_test.copy().dropna(subset="ub")
            df_tmp.loc[:, self._rho_column_name()] = np.nan

            df_train = pd.concat([df_train, df_tmp]).sample(frac=1)
        else:
            df_train = _df.query("radar_station in @train_radars")

        radar_subset = {"train": df_train,
                        "val": df_val,
                        "test": df_test,
                        "final_train": df_train,
                        "all": _df}[subset]
        return radar_subset


class BirdDatasetMultipleNightsForecast(BirdDatasetSingleNight):
    def __init__(self, start_date="2018-04-05", end_date="2018-04-07", *args, **kwargs):
        self.start_date = start_date
        self.end_date = end_date


        # train_val_test_proportion = (0.7, 0., 0.3)
        # if not np.isclose(sum(train_val_test_proportion), 1.):
        #     train_val_test_proportion = [val / sum(train_val_test_proportion) for val in train_val_test_proportion]
        # self.train_val_test_ratios = train_val_test_proportion

        super().__init__(*args, **kwargs)
        self.nights_by_density = self._df.groupby("date_of_night")["birds_per_km3"].max().sort_values(ascending=False)
        self.extent_lat = self.df.lat.min() - 1.3, self.df.lat.max() + 1
        self.extent_lon = self.df.lon.min(), self.df.lon.max() + 1
        self.extent_alt = self.df.z_km.min(), self.df.z_km.max()
        self.extent_time = self.df.time_minutes_night.min(), self.df.time_minutes_night.max()


    def _filter(self, df) -> pd.DataFrame:
        df_subset = df.query(f"(date_of_night>='{self.start_date}') and (date_of_night<='{self.end_date}')")

        df_subset = df_subset.sort_values(by="time_minutes_total").copy()
        df_subset["time_minutes_total"] = df_subset["time_minutes_total"] - df_subset["time_minutes_total"].min()
        df_subset["night"] = df_subset["night"] - df_subset["night"].min()

        max_minutes_per_night = df_subset.groupby("night").time_minutes_night.max().cumsum()
        for i in df_subset.night.unique():
            if i > 0:
                df_subset.loc[df_subset.night == i, "time_minutes_night"] = df_subset.loc[
                                                                                df_subset.night == i, "time_minutes_night"] + \
                                                                            max_minutes_per_night[i - 1]  # + 120
        return df_subset

    def _split_train_val_test(self, _df, subset):
        _df["night_radar_index"] = _df.groupby(['radar_station', 'date_of_night']).ngroup().add(1)
        unique_radars = _df["night_radar_index"].unique()

        # all_sizes = [ratio * len(unique_radars) for ratio in self.train_val_test_ratios]
        # all_sizes = [int(val) for val in iteround.saferound(all_sizes, 0)]
        # assert sum(all_sizes) == len(unique_radars)

        df_train = _df.query(f" (date_of_night<'{self.end_date}') ")
        df_train = df_train[pd.to_datetime(df_train.time.dt.date) != self.end_date]
        df_val = _df.query(f"(date_of_night<'{self.end_date}')")
        df_val = df_val[pd.to_datetime(df_val.time.dt.date) == self.end_date]
        df_final_train = _df.query(f"(date_of_night<'{self.end_date}')")
        df_test = _df.query(f"(date_of_night>='{self.end_date}')")
        self.df_train = df_train
        self.df_val = df_val
        self.df_final_train = df_final_train
        self.df_test = df_test

        radar_subset = {"train": df_train,
                        "val": df_val,
                        "test": df_test,
                        "final_train": df_final_train,
                        "all": _df}[subset]

        return radar_subset


if __name__ == "__main__":
    bird_data = BirdDatasetSingleNight(subset="train")
