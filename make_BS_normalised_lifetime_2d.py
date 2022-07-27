# %run fig_BS_normalised_lifetime_2d.py

"""
Normalised Eddy Lifetimes
=========================

"""

# import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from numpy import (
    unique,
    linspace,
    interp,
    where,
    zeros,
    float32,
    meshgrid,
    array,
    isnan,
    rint,
    arange,
    sort,
)
from numba import njit, prange
import os

from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data

import xarray as xr
from glob import glob
import datetime
from matplotlib.dates import num2date


@njit(cache=True, parallel=True, fastmath=True)
def eddy_norm_lifetime(eddy_var, tracks, xvals, unique_tracks, lifetime_max, out):
    """"""
    for i in prange(unique_tracks.size):
        trk1d_i = where(tracks == unique_tracks[i])[0]
        out += interp(xvals, linspace(0, 1, trk1d_i.size), eddy_var[trk1d_i])
    return out / len(unique_tracks)


def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_xlim(27.37, 41.96), ax.set_ylim(40.86, 46.8)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    ax.legend()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))


# def get_region_tracks(a_or_c_obj, regions):
# """"""
# a_or_c_iter = a_or_c_obj.iter_on("track")

# for region in unique(regions):  # Loop over regions
# for track in a_or_c_iter:  # Loop over individual eddies (birth to death)
# slc = track[0]
# rgn = regions[slc]
# in_region = rgn == region
# if not in_region.any():
# continue
# else:
# pcntg_in_rgn = rint(100 * (in_region == True).sum() / in_region.size)
# if pcntg_in_rgn >= 50:  # greater equal 50%
# regions[slc] = region  # fill all eddies (birth to death) with region #
## Until AD fixes .add_fields() use `cost_association` as surrogate
# a_or_c_obj.cost_association[:] = regions
# return a_or_c_obj


if __name__ == "__main__":

    plt.close("all")

    ACYC = True

    RHO_files = True #False  # If False we process Z_files

    # Minimum eddy lifetime
    # min_track_lifetime = 15  # days

    # directory = "/marula/emason/BlackSea/"
    #directory = "/Users/emason/Downloads/EDDYCOAST_Files/"
    #directory = "/scratch/ulg/mast/emason/daily_MYP_input_files/EddyToCoast_files/"
    directory = "/scratch/ulg/mast/emason/daily_MYP_input_files/Daily_model_files/"
    # directory = "/Users/emason/Downloads/EDDYCOAST_Files/"

    # files_Z = 'Z_slices/BS_Z_slices_ACYC_track-????_19950[4-9]??.nc'
    # files_RHO = 'RHO_slices/BS_RHO_slices_ACYC_track-????_19950[4-9]??.nc'
    if RHO_files:
        # ncfiles = "RHO_slices/BS_RHO_slices_ACYC_track-????_20150[4-6]??.nc"
        #ncfiles = "BS_RHO_slices_ACYC_track-????_2015????.nc"
        ncfiles = "RHO_slices/BS_RHO_slices_ACYC_track-????_????????.nc"
    else:
        #ncfiles = "BS_Z_slices_ACYC_track-????_201[4-6]????.nc"
        ncfiles = "Z_slices/BS_Z_slices_ACYC_track-*_????????.nc"

    date_units = "days since 1950-01-01 00:00:00"
    d1 = datetime.datetime(2011, 1, 1).date()
    d2 = datetime.datetime(2011, 12, 31).date()
    
    # ds_region_indices = xr.open_dataset('ds_ACYC_50pcnt_region_indices_S_shelf.nc')
    # ds_region_indices = xr.open_dataset('ds_ACYC_50pcnt_region_indices_W_basin.nc')
    # ds_region_indices = xr.open_dataset('ds_ACYC_50pcnt_region_indices_NW_shelf.nc')
    # ds_region_indices = xr.open_dataset('ds_ACYC_50pcnt_region_indices_Central_basin.nc')
    # ds_region_indices = xr.open_dataset("ds_ACYC_50pcnt_region_indices_NE_shelf.nc")

    try:
        mesh_mask = xr.open_dataset(
            "/marula/emason/BlackSea/Sample/mesh_mask_31levels.nc"
        )
    except Exception:
        mesh_mask = xr.open_dataset("./mesh_mask_31levels.nc")

    try:
        region_mask = xr.open_dataset("/home/emason/Downloads/region_O2_smooth.nc")
    except Exception:
        region_mask = xr.open_dataset("./region_O2_smooth.nc")


    print("Opening mfdataset")
    ds = xr.open_mfdataset(
        sorted(glob(directory + ncfiles)),
        parallel=True,
    )
    print("Opened ...")
    try:
        print(ds.rho)
    except Exception:
        print(ds.depth)

    #print(ds)

    cb_regions = dict(
        region_0=("S_shelf"),  # 0
        region_1=("NE_shelf"),  # 1
        region_2=("NW_shelf"),  # 2
        region_3=("W_basin"),  # 3
        region_4=("Central_basin"),  # 4
    )

    for region in cb_regions.values():

        print('Opening region %s' % region)
        ds_region_indices = xr.open_dataset(
            "ds_ACYC_50pcnt_region_indices_%s.nc" % region
        )

        print("Select region indices")
        lookup = ds["track"]
        ds_50pcnt_region = ds.where(
            lookup.isin([ds_region_indices["tracks"]]), drop=True
        )

        # `max_count` is eddy with longest lifetime
        max_count = int(ds_region_indices["counts"].max())
        # unique_tracks = unique(ds_region_indices['tracks'])

        # Get center of eddy
        print("Select eddy centers")
        ds_eddy_center = ds_50pcnt_region.isel(
            norm_eddy_radius=[abs(ds["norm_eddy_radius"]).argmin()]
        )

        first_data = True 

        for count, track in enumerate(sort(unique(ds_50pcnt_region["track"].values))):

            print(count, int(track), max_count, region)

            ds_eddy_interp = ds_eddy_center.where(
                (ds_eddy_center["track"]) == track,
                drop=True,
            ).sortby("n")
            
            #ds_eddy_interp = ds_eddy_interp.chunk({'n': -1})

            ds_eddy_interp.time.attrs["units"] = date_units
            if num2date(ds_eddy_interp["time"].values[0]).date() < d1:
                print("Omitting 2014 eddy:", num2date(ds_eddy_interp["time"].values[0]).date())
                continue
            if num2date(ds_eddy_interp["time"].values[-1]).date() > d2:
                print("Omitting 2019 eddy", num2date(ds_eddy_interp["time"].values[-1]).date())
                continue

            ds_eddy_interp = ds_eddy_interp.drop("norm_eddy_radius")

            the_size = ds_eddy_interp.index.size

            # Normalise the interp:  0 -> 1
            ds_eddy_interp = ds_eddy_interp.assign_coords(
                {"index": linspace(0, 1, the_size)},
            ).chunk({'index': -1})

            the_index = linspace(0, 1, int(max_count))
            if not RHO_files:
                ds_eddy_interp = ds_eddy_interp.interp(
                    index=array(the_index, dtype=float), depth=ds["depth"],
                )
            else:
                ds_eddy_interp = ds_eddy_interp.interp(
                    index=array(the_index, dtype=float), rho=ds["rho"],
                )

            if first_data:
                print('First data')
                ds_eddy_merge = ds_eddy_interp.squeeze().copy()
                first_data = False
                #continue

            ds_eddy_merge = xr.concat([ds_eddy_merge, ds_eddy_interp], dim="count")

        print('Get mean')
        ds_eddy_mean = ds_eddy_merge.mean("count").squeeze()

        if RHO_files:
            vertical_coord = "RHO"
            ds_eddy_mean.transpose("rho", "index").to_netcdf(
                "./Norm2D_V1_%s_%s.nc" % (region, vertical_coord)
            )
        else:
            vertical_coord = "Z"
            ds_eddy_mean.transpose("depth", "index").to_netcdf(
                "./Norm2D_V1_%s_%s.nc" % (region, vertical_coord)
            )
        print('Saved')

    # Reindex eveything
    print("DONE")
