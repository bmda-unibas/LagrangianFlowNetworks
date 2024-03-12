import os

import numpy as np
import pandas as pd
import torch
from cartopy import crs as ccrs
from matplotlib import pyplot as plt, colors
from scipy.stats._qmc import Sobol
from tqdm import tqdm
import shapely.geometry as sgeom

from datasets.bird_util import project_coords
from experiments.bird_migration.bird_utils import st_proj, build_cartopy_map, get_mask_out_of_region, get_cartopy_polygons
from experiments.bird_migration.models.lflow import z_to_lat_lon
from experiments.bird_migration.models.util import device
from experiments.gaussians.util.plot_util import quiver_skip
from enflows.utils.torchutils import tensor_to_np, np_to_tensor
import datasets.bird_data as bird_data

from experiments.bird_migration.models.template import DensityVelocityInterface
from experiments.bird_migration.models.MLP import VanillaNN


def plot_radar_positions(cds: bird_data.BirdDatasetMultipleNights, suffix=None):
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.epsg(3035), "aspect": 1}, figsize=(20 / 3, 4))
    # for i, (cur_time, ax) in tqdm(enumerate(zip(times, axs)), total=times.shape[0]):
    poly = get_cartopy_polygons()
    fig, ax = build_cartopy_map(fig, ax, poly=poly, add_stamen=True)

    gl = ax.gridlines(draw_labels=True, zorder=5)

    radars_train = cds.get_radar_lonlat(cds.df_train)
    radars_val = cds.get_radar_lonlat(cds.df_val)
    radars_test = cds.get_radar_lonlat(cds.df_test)
    radars_test.loc[radars_test.radar_station == "nldhl", "lat"] -= 0.15
    radars_test.loc[radars_test.radar_station == "nldhl", "lon"] += 0.15
    ax.scatter(radars_train["lon"], radars_train["lat"], marker='o', transform=ccrs.PlateCarree(), s=30,
               color="lightgreen", linewidth=1, edgecolor='black', zorder=13)
    ax.scatter(radars_val["lon"], radars_val["lat"], marker='o', transform=ccrs.PlateCarree(), s=30,
               color="lightgreen", linewidth=1, edgecolor='black', label="train", zorder=13)
    ax.scatter(radars_test["lon"], radars_test["lat"], marker='o', transform=ccrs.PlateCarree(), s=30,
               color="darkred", linewidth=1, edgecolor='black', label="test", zorder=13)
    set_map_extent(ax)

    plt.legend(loc="lower right")
    if suffix is None:
        plt.savefig("plots/radar_stations.pdf")
    else:
        plt.savefig(f"plots/radar_stations_{suffix}.pdf")


def predict_2d(complete_ds, model, n_steps=100,
               sobol_power=7, ):
    XY_3035_km, lat_Y, lon_X = get_target_grid(complete_ds, n_steps)

    cds = complete_ds

    hour_ranges, night_begins, time_conditions = get_time_conditions(XY_3035_km, complete_ds)

    altitudes_sobol = Sobol(d=1, scramble=True).random_base2(sobol_power) * (cds.extent_alt[1] - cds.extent_alt[0]) + \
                      cds.extent_alt[0] * cds.scale_space

    time_condition = time_conditions[-1]
    density, velocity = predict_on_grid(XY_3035_km, altitudes_sobol, complete_ds, complete_ds.extent_alt, lon_X,
                                        model, predict_advected=False, time_condition=time_condition)

    return density, velocity


def plot_nights(complete_ds, model, filename_prefix="", n_steps=100,
                sobol_power=7, plot_advected=False, include_vel=True,
                simulate_paths=False, mask=True):
    XY_3035_km, lat_Y, lon_X = get_target_grid(complete_ds, n_steps)

    cds = complete_ds

    hour_ranges, night_begins, time_conditions = get_time_conditions(XY_3035_km, complete_ds)

    altitudes_sobol = Sobol(d=1, scramble=True).random_base2(sobol_power) * (cds.extent_alt[1] - cds.extent_alt[0]) + \
                      cds.extent_alt[0] * cds.scale_space

    if simulate_paths:
        samples_per_night = simulate_trajectories(cds, model, time_conditions)

    # norm = colors.PowerNorm(gamma=0.5, vmin=0, vmax=90)
    norm = colors.SymLogNorm(linthresh=10, vmax=90)
    poly = get_cartopy_polygons()

    for night_idx, (time_condition, hour_range) in enumerate(zip(time_conditions, hour_ranges)):
        if plot_advected and night_idx == 0:
            continue

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.epsg(3035), "aspect": 1}, figsize=(20 / 3, 4))
        # for i, (cur_time, ax) in tqdm(enumerate(zip(times, axs)), total=times.shape[0]):
        # ax.set_extent((cds.extent_lon + cds.extent_lat), crs=ccrs.PlateCarree())
        fig, ax = build_cartopy_map(fig, ax, poly=poly, mask=mask)
        gl = ax.gridlines(draw_labels=False, zorder=1_000)
        # i = 1
        # if i < len(axs):
        #     gl.right_labels = False
        #
        # if i > 0:
        #     gl.left_labels = False

        density, velocity = predict_on_grid(XY_3035_km, altitudes_sobol, complete_ds, complete_ds.extent_alt, lon_X,
                                            model, plot_advected, time_condition)
        im = ax.imshow(density, transform=ccrs.PlateCarree(),
                       extent=[lon_X.min(), lon_X.max(), lat_Y.min(), lat_Y.max()],
                       origin='lower', cmap="BuPu", norm=norm)
        set_map_extent(ax)

        if include_vel:
            quiver_skip(lon_X, lat_Y, velocity[..., 0], velocity[..., 1], transform=ccrs.PlateCarree(),
                        grid_shape=lon_X.shape, ax=ax,
                        skip=7)

        if simulate_paths:
            plot_trajectories(ax, night_idx, samples_per_night)

        ax.set_title(str(hour_range))

        plt.tight_layout()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=[1, 10, 20, 30, 50, 90])
        cbar.ax.set_yticklabels([1, 10, 20, 30, 50, 90])  # vertically oriented colorbar
        os.makedirs("plots", exist_ok=True)
        print(f"Saving for night {night_begins[night_idx].date()}")
        if plot_advected:
            plt.savefig(f"plots/advected_{filename_prefix}_{str(night_begins[night_idx].date())}.pdf",
                        bbox_inches='tight')
        else:
            plt.savefig(f"plots/{filename_prefix}_{str(night_begins[night_idx].date())}.pdf", bbox_inches='tight')


def set_map_extent(ax):
    ax.set_extent((-4.5, 14.5, 42.5, 54.5), crs=ccrs.PlateCarree())


def plot_heatmap_differences(complete_ds, model, night_idx, filename_prefix="", n_steps=100,
                             sobol_power=7):
    XY_3035_km, lat_Y, lon_X = get_target_grid(complete_ds, n_steps)

    cds = complete_ds

    hour_ranges, night_begins, time_conditions = get_time_conditions(XY_3035_km, complete_ds)

    altitudes_sobol = Sobol(d=1, scramble=True).random_base2(sobol_power) * (cds.extent_alt[1] - cds.extent_alt[0]) + \
                      cds.extent_alt[0] * cds.scale_space

    # norm = colors.PowerNorm(gamma=0.5, vmin=0, vmax=90)
    # norm = colors.SymLogNorm(linthresh=10, vmax=90)
    norm = colors.Normalize(vmin=0, vmax=1)
    poly = get_cartopy_polygons()
    time_condition, hour_range = time_conditions[night_idx], hour_ranges[night_idx]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.epsg(3035), "aspect": 1}, figsize=(20 / 3, 4))
    # ax.set_extent((cds.extent_lon + cds.extent_lat), crs=ccrs.PlateCarree())

    fig, ax = build_cartopy_map(fig, ax, poly=poly)
    gl = ax.gridlines(draw_labels=True, zorder=1_000)

    density, _ = predict_on_grid(XY_3035_km, altitudes_sobol, complete_ds, complete_ds.extent_alt, lon_X, model, False,
                                 time_condition, t_reference=time_conditions[0])

    density_advected, _ = predict_on_grid(XY_3035_km, altitudes_sobol, complete_ds, complete_ds.extent_alt, lon_X,
                                          model, True, time_condition, t_reference=time_conditions[0])
    im = ax.imshow(np.abs(density - density_advected) / (np.abs(density) + np.abs(density_advected) + 1e-1),
                   transform=ccrs.PlateCarree(),
                   extent=[lon_X.min(), lon_X.max(), lat_Y.min(), lat_Y.max()],
                   origin='lower', cmap="plasma", norm=norm)

    ax.set_title(str(hour_range))
    set_map_extent(ax)
    plt.tight_layout()
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0.01, 0.25, 0.5, 0.75, 1])
    cbar.ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])  # vertically oriented colorbar
    os.makedirs("plots", exist_ok=True)
    print(f"Saving for night {night_begins[night_idx].date()}")

    plt.savefig(f"plots/differences_{filename_prefix}_{str(night_begins[night_idx].date())}.pdf", bbox_inches='tight')


def plot_trajectories(ax, night_idx, samples_per_night):
    if night_idx > 0:
        for line in range(samples_per_night[night_idx].shape[1]):
            try:
                # track = sgeom.LineString(zip(lon_arr[:, line], lat_arr[:, line]))
                track = sgeom.LineString(zip(samples_per_night[night_idx][:, line, 0],
                                             samples_per_night[night_idx][:, line, 1]))

                ax.add_geometries([track],
                                  ccrs.PlateCarree(),
                                  facecolor='none',
                                  edgecolor='darkorange',
                                  linestyle="--",
                                  linewidth=1.5, alpha=0.8)
            except Exception:
                pass
    else:
        lon = samples_per_night[night_idx][0, :, 0]
        lat = samples_per_night[night_idx][0, :, 1]
        ax.scatter(lon, lat, marker='+', transform=ccrs.PlateCarree(), s=20, color='darkorange',
                   alpha=0.6)


def simulate_trajectories(cds, model, time_conditions):
    samples_per_night = {}
    n_samples = 20_000
    zs_sampled = torch.randn((n_samples, 3), dtype=torch.float32,
                             device=device) * 0.5  # I want high density samples
    lat, lon, _xs = z_to_lat_lon(model, (
        time_conditions[0][0].item()),
                                 # (complete_ds.extent_time[0] + complete_ds.extent_time[1]) * complete_ds.scale_time) / 2,
                                 zs_sampled)
    keep = get_mask_out_of_region(cds.extent_lat, cds.extent_lon, lat, lon)
    keep &= lat < 50
    zs_inregion = zs_sampled[keep][:10]
    for night in range(len(time_conditions)):
        intermediate_times = np.linspace(time_conditions[0][0].item(),
                                         time_conditions[night][0].item(), 20)
        longitudes, latitudes = [], []
        lon_lat_overtime = []
        for t in intermediate_times:
            lat, lon, _xs = z_to_lat_lon(model, t, zs_inregion)
            # keep = get_mask_out_of_region(cds.extent_lat, cds.extent_lon, lat, lon)
            lon_lat = np.stack([lon, lat], -1)
            lon_lat_overtime.append(lon_lat)
            longitudes.append(lon)
            latitudes.append(lat)

        lonlat_arr = np.array(lon_lat_overtime)
        samples_per_night[night] = lonlat_arr
    return samples_per_night


def get_time_conditions(XY_3035_km, complete_ds):
    nights = np.sort(complete_ds.df.night.unique())
    df_2 = complete_ds.df.copy().sort_values("time_minutes_night")
    df_2.loc[:, "rounded_time"] = df_2.time.dt.round("2h")
    night_begins = df_2.groupby("night")["time"].min().dt.round("1h")
    night_ends = df_2.groupby("night")["time"].max().dt.round("1h")
    night_offset = df_2.groupby("night")["time_minutes_night"].min() / complete_ds.scale_time
    time_conditions = []
    hour_ranges = []
    for night_index in range(0, nights.shape[0]):
        hour_range = pd.date_range(start=night_begins[night_index], end=night_ends[night_index], periods=3)
        times = (hour_range - df_2.query("night==@night_index")["time"].min()).astype('timedelta64[m]').values
        times = times + night_offset[night_index]
        cur_time = times[1]
        time_conditions.append(np.ones((XY_3035_km.shape[0], 1)) * cur_time * complete_ds.scale_time)
        hour_ranges.append(hour_range[1])
    return hour_ranges, night_begins, time_conditions


def predict_on_grid(XY_3035_km, altitudes_sobol, complete_ds, extent_alt, lon_X, model:DensityVelocityInterface, predict_advected,
                    time_condition, t_reference=None):
    density = 0
    velocity = 0  # torch.zeros((XY_3035_km.shape[0], 3))
    for alt in tqdm(altitudes_sobol):
        Z_km = alt * np.ones((XY_3035_km.shape[0], 1))
        XYZ_3035_km = np.concatenate([XY_3035_km, Z_km], -1)

        if predict_advected:
            cur_density = np.exp(model.predict_logdensity_viaode_split(np.concatenate([XYZ_3035_km, time_condition], -1),
                                                                t_reference=t_reference,
                                                                split_size=512))
        else:
            cur_density = np.exp(model.predict_logdensity_split(np.concatenate([XYZ_3035_km, time_condition], -1)))
        cur_density = cur_density.reshape(lon_X.shape)
        cur_velocity = model.predict_vel_split(np.concatenate([XYZ_3035_km, time_condition], -1))
        cur_velocity = cur_velocity.reshape((lon_X.shape[0],
                                             lon_X.shape[1],
                                             3))

        density += cur_density
        velocity += ((cur_density[..., np.newaxis]) * cur_velocity / (
                (1. / complete_ds.scale_time) * complete_ds.scale_space))
    velocity /= len(altitudes_sobol) #* density[..., np.newaxis] + 1e-7
    velocity /= np.linalg.norm(velocity[..., :2], keepdims=True, axis=-1)
    density /= len(altitudes_sobol)
    density *= (extent_alt[1] - extent_alt[0])
    return density, velocity


def predict_on_grid_3d(XY_3035_km, altitudes_sobol, complete_ds, extent_alt, lon_X, model, predict_advected,
                       time_condition, time_conditions):
    densities = []
    for alt in tqdm(altitudes_sobol):
        Z_km = alt * np.ones((XY_3035_km.shape[0], 1))
        XYZ_3035_km = np.concatenate([XY_3035_km, Z_km], -1)

        if predict_advected:
            model: VanillaNN
            with torch.no_grad():
                cur_density = model.log_density_via_ode(np_to_tensor(XYZ_3035_km, device=device),
                                                        t_now=np_to_tensor(time_condition, device=device),
                                                        t_reference=np_to_tensor(time_conditions[0],
                                                                                 device=device),
                                                        vel_scale=1.
                                                        ).exp()
                cur_density = tensor_to_np(cur_density).reshape(lon_X.shape)
        else:
            cur_density = tensor_to_np(model.log_density(x=np_to_tensor(XYZ_3035_km, device=device),
                                                         t=np_to_tensor(time_condition,
                                                                        device=device)).exp()
                                       ).reshape(lon_X.shape)
        densities.append(cur_density)
    return np.stack(densities, -1)


def get_target_grid(complete_ds, n_steps):
    extent_alt, extent_lat, extent_lon, extent_time = complete_ds.extent_alt, complete_ds.extent_lat, complete_ds.extent_lon, complete_ds.extent_time,
    lon_xx, lat_yy = np.linspace(extent_lon[0] - 4, extent_lon[1] + 0.2, n_steps), \
        np.linspace(extent_lat[0] - 1, extent_lat[1] + 0.2, n_steps)
    lon_X, lat_Y = np.meshgrid(lon_xx, lat_yy)
    ll_XY = np.stack([lon_X, lat_Y], -1).reshape(-1, 2)
    XY_3035_km = np.stack(project_coords(ll_XY[..., [0]], ll_XY[..., [1]]), -1) / 1_000 * complete_ds.scale_space
    return XY_3035_km, lat_Y, lon_X
