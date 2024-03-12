import cartopy.feature as cf
import optuna
import pandas as pd
import torch
import time

from enflows.utils.torchutils import tensor_to_np
import matplotlib.pyplot as plt
import os

os.environ['GDAL_DATA'] = '/home/arefab00/miniconda3/envs/lflows/share/gdal'
os.environ['PROJ_LIB'] = '/home/arefab00/miniconda3/envs/lflows/share/proj'

from shapely.geometry import Polygon
from cartopy.io import shapereader
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
import geopandas

from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

stamen_terrain = cimgt.Stamen('terrain-background')
# projections that involved
st_proj = stamen_terrain.crs  # projection used by Stamen images


class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.time_counter = 0.
        self.end_time = None

    def reset(self):
        self.start_time = time.time()
        self.time_counter = 0.

    def stop(self):
        self.end_time = time.time()
        self.time_counter += self.end_time - self.start_time

    def cont(self):
        self.start_time = time.time()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.95):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        val = tensor_to_np(val)
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def explode(indf):
    """https://gist.github.com/mhweber/cf36bb4e09df9deee5eb54dc6be74d26"""
    outdf = geopandas.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row, ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = geopandas.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row] * recs, ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom, 'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf, ignore_index=True)
    return outdf


def build_cartopy_map(fig, ax, poly, add_stamen=False, mask=True):
    def rect_from_bound(xmin, xmax, ymin, ymax):
        """Returns list of (x,y)'s for a rectangle"""
        buffer =  5
        xs = [xmax + buffer, xmin - buffer, xmin - buffer, xmax + buffer, xmax + buffer]
        ys = [ymax + buffer, ymax + buffer, ymin - buffer, ymin - buffer, ymax + buffer]
        return [(x, y) for x, y in zip(xs, ys)]

    stamen_terrain = cimgt.Stamen('terrain-background')
    # projections that involved
    st_proj = stamen_terrain.crs  # projection used by Stamen images

    ll_proj = ccrs.PlateCarree()  # CRS for raw long/lat
    # create fig and axes using intended projection
    ax.add_geometries(poly, crs=ll_proj, facecolor='none', edgecolor='black')
    exts = [-5.5, 16, 41, 55]
    ax.set_extent(exts, crs=ll_proj)
    # make a mask polygon by polygon's difference operation
    # base polygon is a rectangle, another polygon are the simplified countries

    ax.add_feature(cf.COASTLINE, lw=1)
    ax.add_feature(cf.BORDERS, lw=1)
    if mask:
        msk = Polygon(rect_from_bound(*exts)).difference(poly[0].simplify(0.01))
        msk_stm = st_proj.project_geometry(msk, ll_proj)  # project geometry to the projection used by stamen
        ax.add_geometries(msk_stm, st_proj, zorder=5, facecolor='white', edgecolor='none', alpha=.9)

    # get and plot Stamen images
    if add_stamen:
        ax.add_image(stamen_terrain, 8)  # this requests image, and plot
    # ax.add_feature(cf.OCEAN)
    # ax.add_feature(cf.LAND)
    # ax.stock_img()
    # plot the mask using semi-transparency (alpha=0.65) on the masked-out portion
    return fig, ax


def get_cartopy_polygons():
    # request data for use by geopandas
    resolution = '50m'
    category = 'cultural'
    name = 'admin_0_countries'
    shpfilename = shapereader.natural_earth(resolution, category, name)
    df = geopandas.read_file(shpfilename)
    countries = df[df['ADMIN'].isin(['Germany', 'Netherlands', 'Belgium', 'Luxembourg'])].copy()
    france = df[df['ADMIN'].isin(['France'])].copy()
    mp = france.geometry.item()
    new_poly_list = []
    for P in mp.geoms:
        if P.bounds[0] > -10 and P.bounds[0] < 16 and P.bounds[1] > 42 and P.bounds[1] < 56:
            new_poly_list.append(P)

    M2 = MultiPolygon(new_poly_list)
    france.geometry.iloc[0] = M2
    concat_df = pd.concat([countries, france])
    # countries.loc[france.index].geometry.iloc[0] = M2
    # Exclude French Guiana from the map (also Corsika though),
    # tmp = [x.replace(')', '') for x in str(scan3.loc[43, 'geometry']).split('((')[1:]][1]
    # tmp2 = [x.split(' ') for x in tmp.split(', ')][:-1]
    # tmp3 = [(float(x[0]), float(x[1])) for x in tmp2]
    # France_mainland = Polygon(tmp3)
    # scan3.loc[scan3['name'] == 'France', 'geometry'] = France_mainland
    countries_dissolved = concat_df.dissolve(by='LEVEL')
    poly = [countries_dissolved['geometry'].values[0]]
    return poly


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 5, subplot_kw={'projection': st_proj, "aspect": 1}, figsize=(20, 4),
                            sharey="all")

    for i, ax in enumerate(axs):
        build_cartopy_map(fig, ax)
        gl = ax.gridlines(draw_labels=True)
        if i < len(axs):
            gl.right_labels = False

        if i > 0:
            gl.left_labels = False

    plt.show()


def get_mask_out_of_region(extent_lat, extent_lon, lat, lon):
    keep = (lon < extent_lon[1]) & (extent_lon[0] < lon)
    keep &= (lat < extent_lat[1]) & (extent_lat[0] < lat)
    return keep


def open_or_create_study(storage_name, study_name, overwrite=False):
    try:
        study: optuna = optuna.create_study(study_name=study_name, storage=storage_name,
                                            load_if_exists=not overwrite,
                                            directions=["maximize", "maximize"]
                                            )
    except optuna.exceptions.DuplicatedStudyError:
        optuna.delete_study(study_name=study_name, storage=storage_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=False,
                                    directions=["maximize", "maximize"]
                                    )
    return study


def weighted_MSE(x: torch.Tensor, y: torch.Tensor, weights):
    diff = (x.squeeze() - y.squeeze()) ** 2
    weighted_diff = weights * diff
    return weighted_diff.mean()
