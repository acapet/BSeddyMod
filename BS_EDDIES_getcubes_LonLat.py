# %run make_NewBlackSea_eddy_compo.py
import argparse


# from datetime import date, datetime
import matplotlib.dates as mdates

## See https://matplotlib.org/3.3.0/api/dates_api.html#matplotlib.dates
old_epoch = "0000-12-31T00:00:00"
mdates.set_epoch(old_epoch)

import logging
import os
import subprocess
from glob import glob
from itertools import chain
from math import atan2
from multiprocessing import Pool, cpu_count, sharedctypes

import matplotlib.pyplot as plt
from dateutil import parser

# from Polygon import Polygon
# https://bitbucket.org/jraedler/polygon3/src/master/Polygon/Utils.py
# from Polygon.Utils import convexHull # ,fillHoles, pointList
from dateutil.relativedelta import relativedelta
from gsw import CT_from_pt, Nsquared, SA_from_SP, f, p_from_z, sigma0, z_from_p
from matplotlib import colors
from matplotlib.dates import date2num, datetime, julian2num, num2date
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from netCDF4 import Dataset

# from netCDF4 import num2date as NC4num2date
from numba import njit

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# from numexpr import evaluate
from numpy import roll  # log,
from numpy import (
    abs,
    alltrue,
    arange,
    array,
    asarray,
    ascontiguousarray,
    asfarray,
    atleast_2d,
    bitwise_and,
    ceil,
    concatenate,
    copy,
    ctypeslib,
    diff,
    empty,
    empty_like,
    errstate,
    float32,
    float64,
    float_,
    floor,
    fromiter,
    full,
    full_like,
    hstack,
    int8,
    int16,
    int32,
    int64,
    interp,
    isclose,
    isnan,
    less,
    linspace,
    logical_not,
    logical_or,
    logspace,
    ma,
    maximum,
    median,
    meshgrid,
    nan,
    nanargmax,
    nanargmin,
    nanmean,
    nanmedian,
    nanmin,
    newaxis,
    pi,
    recarray,
    repeat,
    rollaxis,
    sqrt,
    stack,
    take,
    tile,
    unique,
    unravel_index,
    zeros,
    zeros_like,
)
from numpy.ma import is_masked, masked_invalid, masked_where, vstack
from numpy.random import random_sample
from pyproj import Proj
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    NearestNDInterpolator,
    PchipInterpolator,
    RectBivariateSpline,
)
from scipy.ndimage import find_objects, gaussian_filter  # , binary_dilation
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.measurements import label as ndlabel
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from scipy.signal import argrelextrema
from scipy.spatial import Delaunay, cKDTree
from shapely.geometry import asPolygon, shape
from xarray import Dataset as xr_Dataset
from xarray import open_dataset, open_mfdataset
from pandas import date_range

# import progressbar as prog_bar
import py3_eddy_tracker_fortran as hav
from eddy_composite_for_ITE import BaseClass, Composite3D
from eddy_tracker_diagnostics import EddyTrack
from emTools import (
    fillmask_kdtree,
    gaussian_with_nans,
    getncvcm,
    newPosition,
    pcol_2dxy,
    psi2rho,
    rho2u_3d,
    rho2v_3d,
)

# from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator
# from time import sleep


# import motuclient

# from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# See https://matplotlib.org/3.3.0/api/dates_api.html#matplotlib.dates
# set_epoch('0000-12-31')


# Arguments management #                                                                                                                                                                                           
parser = argparse.ArgumentParser()
parser.add_argument("-y","--year", type=int, help="start year")
parser.add_argument("-m","--month", type=int, help="start month (1st)")
parser.add_argument("-n","--ncpu", type=int, help="ncpus")

args = parser.parse_args()
runyear  = args.year
runmonth = args.month
ncpus_arg = args.ncpu

Kelvin = 273.15
# EARTH_R = 6371315.
EARTH_R = 6370997.0  # Same as py-eddy-tracker `grid.py`

logit = False
if logit:
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("numba").disabled = True
    logging.getLogger("numba.byteflow").disabled = True
    logging.getLogger("numba.interpreter").disabled = True
    logging.getLogger("matplotlib").disabled = True
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.colorbar").disabled = True


set_IJ_indices = Composite3D._set_IJ_indices
vorticity = BaseClass.vorticity
get_dx_dy_f = BaseClass.get_dx_dy_f
haversine_dist = BaseClass._haversine_dist
winding_number_poly = hav.pnpoly.wn_pnpoly
distance_vector = hav.haversine.distance_vector
# strptime = datetime.datetime.strptime

interpolator_nn = NearestNDInterpolator
interpolator = CloughTocher2DInterpolator


def compute_scale_and_offset(dmin, dmax, n):
    """http://james.hiebert.name/blog/work/2015/04/18/NetCDF-Scale-Factors.html
    https://stackoverflow.com/questions/57179990/compression-of-arrays-in-netcdf-file
    """
    # stretch/compress data to the available packed range
    scale_factor = (dmax - dmin) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = dmin + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)

def pack_value(unpacked_value, scale_factor, add_offset):
    return floor((unpacked_value - add_offset) / scale_factor)

def unpack_value(packed_value, scale_factor, add_offset):
    return packed_value * scale_factor + add_offset

def create_nc(
    nc,
    xsize,
    ysize,
    zsize,
    BlackSea_z,
    xgrid,
    ygrid,
    xmidi,
    ymidi,
    BlackSea_file,
    eddy_dir,
    savedir,
    mercator_file,
    eddy_tracking_output_file,
    save_file,
    fill_value_32,
    fill_value_64,
    lon_full,
    lat_full,
):

    print("---- Creating output file")
    nc.created = datetime.datetime.utcnow().isoformat()
    nc.lon_min = float64(domain[0])
    nc.lon_max = float64(domain[1])
    nc.lat_min = float64(domain[2])
    nc.lat_max = float64(domain[3])
    nc.z_min = BlackSea_z.min()
    nc.z_max = BlackSea_z.max()
    y, m, d = time_domain[0]
    nc.start_date = date2num(datetime.datetime(y, m, d))
    y, m, d = time_domain[1]
    nc.end_date = date2num(datetime.datetime(y, m, d))
    nc.BlackSea_file = BlackSea_file
    nc.mercator_file = mercator_file
    nc.eddy_dir = eddy_dir
    nc.eddy_tracking_output_file = eddy_tracking_output_file  # CC or AC
    # nc.eddy_file_ccyc = eddy_file_ccyc
    nc.savedir = savedir
    nc.save_file = save_file

    nc.createDimension("time", None)
    nc.createDimension("x", xsize)
    nc.createDimension("y", ysize)
    nc.createDimension("z", zsize)
    nc.createDimension("BS_lon", lon_full.size)
    nc.createDimension("BS_lat", lat_full.size)
    nc.createDimension("four", 4)
    nc.createDimension("fifty", 50)
    nc.createDimension("one", 1)

    nc.createVariable("xmidi", int16, ("one"))
    nc.createVariable("ymidi", int16, ("one"))
    nc.variables["xmidi"][:] = xmidi
    nc.variables["ymidi"][:] = ymidi

    nc.createVariable("x", float32, ("x"))
    nc.variables["x"].description = "Normalised distance along x axis"
    nc.variables["x"].units = "None"

    nc.createVariable("y", float32, ("y"))
    nc.variables["y"].description = "Normalised distance along y axis"
    nc.variables["y"].units = "None"

    nc.createVariable("z", float32, ("z"))
    nc.variables["z"].description = "Depths"
    nc.variables["z"].units = "m"

    nc.variables["x"][:] = xgrid
    nc.variables["y"][:] = ygrid
    nc.variables["z"][:] = BlackSea_z

    nc.createVariable("lon", float32, ("BS_lon"))
    nc.variables["lon"][:] = lon_full
    nc.variables["lon"].description = "Longitude from BlackSea model"
    nc.variables["lon"].units = "degrees"

    nc.createVariable("lat", float32, ("BS_lat"))
    nc.variables["lat"][:] = lat_full
    nc.variables["lat"].description = "Latitude from BlackSea model"
    nc.variables["lat"].units = "degrees"
   
    # Eddy tracker output variables
    nc.createVariable("eddy_date", int64, ("one"))
    nc.createVariable("time", float32, ("time"))
    nc.createVariable("track", int64, ("time"))
    nc.createVariable("n", int32, ("time"))
    nc.createVariable("centlon", float32, ("time"))
    nc.createVariable("centlat", float32, ("time"))
    nc.createVariable("radius", float32, ("time"))
    nc.createVariable("radius_eff", float32, ("time"))
    nc.createVariable("amplitude", float32, ("time"))
    nc.createVariable("cyc", int8, ("time"))
    nc.createVariable("virtual", int8, ("time"))

    # Ordinal
    # nc.createVariable('order', int64, ('time'))
    # nc.variables['order'].description = \
    # ('Eddy count over multiple monthly files')
    # nc.variables['order'].units = "index"

    nc.createVariable("the_tlt_tki", int16, ("time", "z"))
    nc.variables["the_tlt_tki"].description = "Indices for zonal eddy tilt correction"
    nc.variables["the_tlt_tki"].units = "index"

    nc.createVariable("the_tlt_tkj", int16, ("time", "z"))
    nc.variables[
        "the_tlt_tkj"
    ].description = "Indices for meridional eddy tilt correction"
    nc.variables["the_tlt_tkj"].units = "index"

    nc.createVariable("i0i1_j0j1", int16, ("time", "four"))

    nc.createVariable("contour_lon_s", float32, ("time", "fifty"))
    nc.createVariable("contour_lat_s", float32, ("time", "fifty"))
    nc.createVariable("contour_lon_e", float32, ("time", "fifty"))
    nc.createVariable("contour_lat_e", float32, ("time", "fifty"))

    # Create variables at center of eddy (xmidi, ymidi)

    # ---------------------------------------------------------------------------
    xyz_midi_str = "(3D eddy cube)"
    xy_midi_str = "(2D eddy field)"
    ##---------------------------------------------------------------------------
    vname = "VORT"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Vorticity %s" % xyz_midi_str
    nc.variables[vname].units = "s**-1"

    # ---------------------------------------------------------------------------
    vname = "MASK"
    nc.createVariable(
        vname, int16, ("time", "z", "y", "x"), fill_value=fill_value_16, zlib=True
    )
    nc.variables[vname].description = "Mask %s" % xyz_midi_str
    nc.variables[vname].units = "Binary 1s (sea) and 0s (land)"

    # ---------------------------------------------------------------------------
    vname = "TEMP"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Temperature  %s" % xyz_midi_str
    nc.variables[vname].units = "deg C"

    # ---------------------------------------------------------------------------
    vname = "SALT"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Salinity %s" % xyz_midi_str
    nc.variables[vname].units = "PSU"

    # ---------------------------------------------------------------------------
    vname = "RHO"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Density %s" % xyz_midi_str
    nc.variables[vname].units = "kg/m**3"

    # ---------------------------------------------------------------------------
    vname = "TEMP_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Temperature anomaly %s" % xyz_midi_str
    nc.variables[vname].units = "deg C"

    # ---------------------------------------------------------------------------
    vname = "SALT_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Salinity anomaly %s" % xyz_midi_str
    nc.variables[vname].units = "PSU"

    # ---------------------------------------------------------------------------
    vname = "RHO_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Density anomaly %s" % xyz_midi_str
    nc.variables[vname].units = "kg/m**3"

    # ---------------------------------------------------------------------------
    vname = "U"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "U velocity %s" % xyz_midi_str
    nc.variables[vname].units = "m/s"

    # ---------------------------------------------------------------------------
    vname = "V"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "V velocity %s" % xyz_midi_str
    nc.variables[vname].units = "m/s"

    # ---------------------------------------------------------------------------
    vname = "W"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "W velocity %s" % xyz_midi_str
    nc.variables[vname].units = "m/s"

    # ---------------------------------------------------------------------------
    vname = "NPPO"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NPPO  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol O2 m-3 s-1"

    vname = "NPPO_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NPPO_P %s" % xyz_midi_str
    nc.variables[vname].units = "mmol O2 m-3 s-1"

    # ---------------------------------------------------------------------------
    vname = "ZooResp"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ZooResp  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol O2 m-3 s-1"

    vname = "ZooResp_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ZooResp_P %s" % xyz_midi_str
    nc.variables[vname].units = "mmol O2 m-3 s-1"
    # ---------------------------------------------------------------------------
    vname = "DOC"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "DOC %s" % xyz_midi_str
    nc.variables[vname].units = "-"

    vname = "DOC_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "DOC_P %s" % xyz_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "NPPOI"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NPPOI %s" % xy_midi_str
    nc.variables[vname].units = "-"

    vname = "NPPOI_P"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NPPOI_P %s" % xy_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "bac_oxygenconsumptionI"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "bac_oxygenconsumptionI %s" % xy_midi_str
    nc.variables[vname].units = "-"

    vname = "bac_oxygenconsumptionI_P"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "bac_oxygenconsumptionI_P  %s" % xy_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "bac_oxygenconsumption"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "bac_oxygenconsumption %s" % xyz_midi_str
    nc.variables[vname].units = "-"

    vname = "bac_oxygenconsumption_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "bac_oxygenconsumption_P  %s" % xyz_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "POC"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "POC  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "POC_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "POC_P  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "CHL"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "CHL %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "CHL_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "CHL_P  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "PAR"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "PAR %s" % xyz_midi_str
    nc.variables[vname].units = "W/m2"

    vname = "PAR_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "PAR_P  %s" % xyz_midi_str
    nc.variables[vname].units = "W/m2"

    # ---------------------------------------------------------------------------
    vname = "DOX"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "DOX %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "DOX_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "DOX_P  %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "AirSeaOxygenFlux"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "AirSeaOxygenFlux %s" % xy_midi_str
    nc.variables[vname].units = "-"

    vname = "AirSeaOxygenFlux_P"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "AirSeaOxygenFlux_P %s" % xy_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "OXIDATIONBYDOXI"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "OXIDATIONBYDOXI %s" % xy_midi_str
    nc.variables[vname].units = "-"

    vname = "OXIDATIONBYDOXI_P"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "OXIDATIONBYDOXI_P %s" % xy_midi_str
    nc.variables[vname].units = "-"

    # ---------------------------------------------------------------------------
    vname = "OXIDATIONBYDOX"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "OXIDATIONBYDOX %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "OXIDATIONBYDOX_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "OXIDATIONBYDOXP %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "NOS"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NOS %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "NOS_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NOS_P %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "NHS"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NHS %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "NHS_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "NHS_P %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    # ---------------------------------------------------------------------------
    vname = "ODU"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ODU %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"

    vname = "ODU_P"
    nc.createVariable(
        vname, float32, ("time", "z", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ODU_P %s" % xyz_midi_str
    nc.variables[vname].units = "mmol/m3"
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    ### Horizontal sections
    vname = "SSH"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Sea surface height"
    nc.variables[vname].units = "m"
    # ---------------------------------------------------------------------------
    vname = "MLD"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Mixed layer depth"
    nc.variables[vname].units = "m"
    # ---------------------------------------------------------------------------
    vname = "navlon"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Longitude"
    nc.variables[vname].units = "degrees east"
    # ---------------------------------------------------------------------------
    vname = "navlat"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Latitude"
    nc.variables[vname].units = "degrees north"    
    # ---------------------------------------------------------------------------
    vname = "TOPO"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "Topography"
    nc.variables[vname].units = "m"

    # ---------------------------------------------------------------------------
    vname = "ZooRespI"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ZooRespI"
    nc.variables[vname].units = "--"
    
    # ---------------------------------------------------------------------------
    vname = "ZooRespI_P"
    nc.createVariable(
        vname, float32, ("time", "y", "x"), fill_value=fill_value_32, zlib=True
    )
    nc.variables[vname].description = "ZooRespI_P"
    nc.variables[vname].units = "--"

    # ---------------------------------------------------------------------------

    
    return

def non_time_coords(ds):
    """https://github.com/pydata/xarray/issues/1385"""
    return [v for v in ds.data_vars if "time" not in ds[v].dims]

def drop_non_essential_vars_pop(ds):
    """https://github.com/pydata/xarray/issues/1385"""
    return ds.drop(non_time_coords(ds))

def get_pm_pn(lon, lat):
    """"""
    # @njit
    @njit(cache=True, fastmath=True)
    def half_interp(h_one, h_two):
        """
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        """
        h_one = h_one.copy()
        h_one += h_two
        h_one *= 0.5
        return h_one

    lonu = half_interp(lon[:, :-1], lon[:, 1:])
    latu = half_interp(lat[:, :-1], lat[:, 1:])
    lonv = half_interp(lon[:-1], lon[1:])
    latv = half_interp(lat[:-1], lat[1:])

    # Get pm and pn
    pm = zeros_like(lon)
    pm[:, 1:-1] = haversine_dist(lonu[:, :-1], latu[:, :-1], lonu[:, 1:], latu[:, 1:])
    pm[:, 0], pm[:, -1] = pm[:, 1], pm[:, -2]

    pn = zeros_like(lon)
    pn[1:-1] = haversine_dist(lonv[:-1], latv[:-1], lonv[1:], latv[1:])
    pn[0], pn[-1] = pn[1], pn[-2]
    return 1.0 / pm, 1.0 / pn

def ertelpv(u, v, rho, f, pm, pn, dz_r=None, rho0=1027.4):
    """
    Compute Ertel's potential vorticity
    Inputs:
        u on rho-points
        v on rho-points
        rho on rho-points
        z_r
    Outputs
        pv on rho-points
    """
    lbd = rho
    """
    Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)] * dlambda / dz
    """
    # Get 3d f
    # TODO f needs to be 2d; save f, lon and lat in composites ???
    #     For the moment, use f constant...
    # print 'TODO f needs to be 2d'
    # f2d = full_like(ssh.data[0], f)
    pvk = tile(f, (dz_r.size, 1, 1))[:, 1:-1, 1:-1]

    dxi_psi = 0.25 * (pm[:-1, :-1] + pm[:-1, 1:] + pm[1:, :-1] + pm[1:, 1:])
    deta_psi = 0.25 * (pn[:-1, :-1] + pn[:-1, 1:] + pn[1:, :-1] + pn[1:, 1:])

    # Get d(v)/d(xi) at PSI-points
    v1, v2 = rho2v_3d(v)[..., 1:], rho2v_3d(v)[..., :-1]
    dvdxi = evaluate("dxi_psi * (v1 - v2)")

    # Get d(u)/d(eta) at PSI-points.
    u1, u2 = rho2u_3d(u)[:, 1:], rho2u_3d(u)[:, :-1]
    dudeta = evaluate("deta_psi * (u1 - u2)")

    # Compute Ertel potential vorticity <k hat> at horizontal RHO-points and
    #  vertical W-points------------------------------------------
    dvdxi -= dudeta
    omega = zeros_like(dvdxi[:-1, 1:, :-1])
    dvdxi1, dvdxi2, dvdxi3, dvdxi4 = (
        dvdxi[:-1, 1:, :-1],
        dvdxi[:-1, 1:, 1:],
        dvdxi[:-1, :-1, :-1],
        dvdxi[:-1, :-1, 1:],
    )
    dvdxi5, dvdxi6, dvdxi7, dvdxi8 = (
        dvdxi[1:, 1:, :-1],
        dvdxi[1:, 1:, 1:],
        dvdxi[1:, :-1, :-1],
        dvdxi[1:, :-1, 1:],
    )
    omega[:] = evaluate(
        "0.125 * (dvdxi1 + dvdxi2 + dvdxi3 + dvdxi4 + \
                                  dvdxi5 + dvdxi6 + dvdxi7 + dvdxi8)"
    )

    pvk += omega
    pvk *= (
        lbd[1:, 1:-1, 1:-1] - lbd[:-1, 1:-1, 1:-1]
    )  # np.diff(lbd[:, 1:-1, 1:-1], axis=0)
    pvk /= dz_r[:, None, None]

    """------------------------------------------------------------------------
    Ertel potential vorticity, term 2: (dv/dz)*(drho/dx)
    """
    # Compute d(v)/d(z) at horizontal V-points and vertical W-points
    v1, v2, dz_w_at_r = (rho2v_3d(v)[1:], rho2v_3d(v)[:-1], dz_r[:, None, None])
    dvdz = evaluate("(v1 - v2) / dz_w_at_r")

    # Compute d(lambda)/d(xi) at horizontal U-points and vertical RHO-points
    lbd1, lbd2, pm1, pm2 = lbd[..., 1:], lbd[..., :-1], pm[:, 1:], pm[:, :-1]
    dldxi = evaluate("(lbd1 - lbd2) * (0.5 * (pm1 + pm2))")

    # Add in term 2 contribution to Ertel potential vorticity <i hat>
    dvdz1, dvdz2 = dvdz[:, 1:, 1:-1], dvdz[:, :-1, 1:-1]
    dldxi1, dldxi2, dldxi3, dldxi4 = (
        dldxi[1:, 1:-1, :-1],
        dldxi[1:, 1:-1, 1:],
        dldxi[:-1, 1:-1, :-1],
        dldxi[:-1, 1:-1, 1:],
    )
    pvi = evaluate(
        "0.5 * (dvdz1 + dvdz2) * \
                    0.25 * (dldxi1 + dldxi2 + dldxi3 + dldxi4)"
    )

    """------------------------------------------------------------------------
    Ertel potential vorticity, term 3: (du/dz)*(drho/dy)
    """
    # Compute d(u)/d(z) at horizontal U-points and vertical W-points
    u1, u2 = (rho2u_3d(u)[1:], rho2u_3d(u)[:-1])
    dudz = evaluate("(u1 - u2) / dz_w_at_r")

    # Compute d(lambad)/d(eta) at horizontal V-points and vertical RHO-points
    lbd1, lbd2, pn1, pn2 = lbd[:, 1:], lbd[:, :-1], pn[1:], pn[:-1]
    dldeta = evaluate("(lbd1 - lbd2) * (0.5 * (pn1 + pn2))")

    # Add in term 3 contribution to Ertel potential vorticity <j hat>
    dudz1, dudz2 = dudz[:, 1:-1, 1:], dudz[:, 1:-1, :-1]
    dldeta1, dldeta2, dldeta3, dldeta4 = (
        dldeta[1:, :-1, 1:-1],
        dldeta[1:, 1:, 1:-1],
        dldeta[:-1, :-1, 1:-1],
        dldeta[:-1, 1:, 1:-1],
    )
    pvj = evaluate(
        "0.5 * (dudz1 + dudz2) * \
                    0.25 * (dldeta1 + dldeta2 + dldeta3 + dldeta4)"
    )

    """
    Sum potential vorticity components, and divide by rho0
    """
    lbd *= 0
    lbd += nan
    lbd[1:, 1:-1, 1:-1] = evaluate("(-pvi + pvj + pvk) / rho0")
    return ma.masked_where(lbd == nan, lbd)

def fill_nans(x):
    """"""
    x = atleast_2d(x)
    nans = isnan(x)
    if nans.any():
        if nans.all():
            return x
        else:
            try:
                x[:], _ = fillmask_kdtree(x, logical_not(nans))
            except Exception:
                x[:] = nan
            finally:
                return x
    else:
        return x

def get_eddy_recarray(etrk):  # , ncyc):
    """
    Make record array with the eddy properties
    (Copied from eddy_composite_for_ITE.py)
    """
    eddy_observation_list, eddy_dates = [], []
    cntr_observation_list = []

    # Anticyclones
    for (
        centlon,
        centlat,
        radius,
        radius_e,
        date,
        track,
        n,
        amplitude,
        virtual,
        contour_lon_s,
        contour_lat_s,
        contour_lon_e,
        contour_lat_e,
    ) in zip(
        etrk.lons(),
        etrk.lats(),
        etrk.radii(),
        etrk.radii_eff(),
        etrk.dates(),
        etrk.track(),
        etrk.n(),
        etrk.amplitudes(),
        etrk.virtual(),
        etrk.contour_lon_s(),
        etrk.contour_lat_s(),
        etrk.contour_lon_e(),
        etrk.contour_lat_e(),
    ):

        eddy_rec = empty(
            (1),
            dtype=[
                ("centlon", float),
                ("centlat", float),
                ("radius", float),
                ("radius_e", float),
                ("date", float),
                ("track", int),
                ("n", int),
                ("amplitude", float),
                ("virtual", int),
            ],
        )

        cntr_rec = empty(
            (50),
            dtype=[
                ("contour_lon_s", float),
                ("contour_lat_s", float),
                ("contour_lon_e", float),
                ("contour_lat_e", float),
            ],
        )

        eddy_rec = eddy_rec.view(recarray)
        eddy_rec.centlon[:], eddy_rec.centlat[:] = centlon, centlat
        eddy_rec.radius[:], eddy_rec.radius_e[:] = radius, radius_e
        eddy_rec.date[:], eddy_rec.track[:] = date, track
        eddy_rec.n[:], eddy_rec.amplitude[:] = n, amplitude
        eddy_rec.virtual[:] = virtual

        cntr_rec = cntr_rec.view(recarray)
        cntr_rec.contour_lon_s[:] = contour_lon_s
        cntr_rec.contour_lat_s[:] = contour_lat_s
        cntr_rec.contour_lon_e[:] = contour_lon_e
        cntr_rec.contour_lat_e[:] = contour_lat_e

        eddy_observation_list.append(eddy_rec)
        cntr_observation_list.append(cntr_rec)
        eddy_dates.append(date)

    eddy_dates = hstack(eddy_dates)
    eddy_sorted_i = eddy_dates.argsort()
    eddy_unsorted_i = arange(eddy_sorted_i.size)

    return (
        eddy_observation_list,
        eddy_dates,
        eddy_sorted_i,
        eddy_unsorted_i,
        cntr_observation_list,
    )

def get_restart_index_and_clip(sorted_eddy_observation_list, restart_date):
    """
    Find index for restart and trim list accordingly
    (Copied from eddy_composite_for_ITE.py)
    """
    restart_date_i = 0
    while True:
        eddy_rst = sorted_eddy_observation_list[restart_date_i]
        if float(restart_date) == eddy_rst.date:
            break
        restart_date_i += 1
    return sorted_eddy_observation_list[restart_date_i:], restart_date_i

def interp_midi(tri, data, interpolator, xcoords, ycoords, fill_value=nan):
    """"""
    interp = interpolator(tri, data.ravel(), fill_value=fill_value)
    return interp(xcoords, ycoords)  # [::-1]

def interp_midi_mask(tri, data, interpolator, xcoords, ycoords):
    """"""
    interp = interpolator(tri, data.ravel())
    return interp(xcoords, ycoords)  # [::-1]

def MP_process_eddies(eddy_iter):
    """
    `eddy_iter`       : arange type iterable

    `the_eddy_days_i` : array containing discrete indices to each eddy;
                        same size as `eddy_iter`
    """
    logging.info("\n---- eddy_iter %s" % eddy_iter)
    eddy_ind = the_eddy_days_i[eddy_iter]
    logging.info("\n---- eddy_ind %s" % eddy_ind)

    eddy = eddy_observation_list[eddy_ind]
    cntr = cntr_observation_list[eddy_ind]

    the_date_str = num2date(eddy.date[0]).date().isoformat()
    print(
        "\n#### %s #### Lon=%s #### Lat=%s ###############"
        % (the_date_str, eddy.centlon[0], eddy.centlat[0])
    )

    BS_mask3d = model_xarr["BS_mask3d_full"].values
    # nsquared = model_xarr['merc_n2_full'].values

    # Get index slices to eddy in model grid
    ii, jj, _, _ = set_IJ_indices(
        eddy.centlon[0],
        eddy.centlat[0],
        eddy.radius[0],
        BlackSea_lon,
        BlackSea_lat,
        I,
        J,
        nrad=6,
    )
    _i0, _i1 = ii
    _j0, _j1 = jj

    xmidi_par = floor((_i1 - _i0) / 2).astype(int)
    ymidi_par = floor((_j1 - _j0) / 2).astype(int)

    x_slc = slice(_i0, _i1)
    y_slc = slice(_j0, _j1)

    # Lon, lat 2d around eddy
    BlackSea_lon_eddy = BlackSea_lon[y_slc, x_slc]
    BlackSea_lat_eddy = BlackSea_lat[y_slc, x_slc]
    BS_eddy_shape = BlackSea_lon_eddy.shape

    # 3d to eddy
    mask_eddy = BS_mask3d[:, y_slc, x_slc].astype(int).copy()

    # Degrees to km so that we can normalise by radius
    proj = Proj(
        "+proj=aeqd +lat_0=%s +lon_0=%s +R=%s"
        % (eddy.centlat[0], eddy.centlon[0], EARTH_R)
    )

    x_m_2d, y_m_2d = proj(BlackSea_lon_eddy, BlackSea_lat_eddy)

    distances_out = empty(BlackSea_lon_eddy.size, order="F")

    distance_vector(
        full(BlackSea_lon_eddy.size, eddy.centlon[0], order="F"),
        full(BlackSea_lon_eddy.size, eddy.centlat[0], order="F"),
        asfarray(BlackSea_lon_eddy),
        asfarray(BlackSea_lat_eddy),
        distances_out,
    )
    # sasa

    dist_m2d = (
        ascontiguousarray(distances_out).reshape(BS_eddy_shape[::-1]).T
    )  # (x_m_2d ** 2 + y_m_2d ** 2) ** 0.5
    dist_norm_by_rad = dist_m2d / eddy.radius[0]

    mask1 = dist_norm_by_rad <= m1_radius  # implies 1.25 * normalised radius
    mask2 = dist_norm_by_rad <= tree_radius
    mask1_i = mask1.ravel().nonzero()[0]

    # ----------------------------------------------------------------------------
    # Get vorticity and tilt indices
    # ----------------------------------------------------------------------------
    # nsquared_tltd = nsquared[:, y_slc, x_slc].copy()
    the_tlt_tki = empty_like(BlackSea_z, dtype=int)
    the_tlt_tkj = empty_like(BlackSea_z, dtype=int)
    BS_vort_tltd = empty_like(mask_eddy).astype(float64)
    BS_eddy_3d_shape = BS_vort_tltd.shape

    for k in range(zsize):

        BS_vort_tltd[k] = model_xarr["BS_vort_full"].values[k, y_slc, x_slc].copy()
        # Multiply by mask1 to ensure stay within eddy radius
        try:
            BS_vort_tltd_view[:] = BS_vort_tltd[k] * mask1
        except Exception:
            BS_vort_tltd_view = BS_vort_tltd[k] * mask1

        if alltrue(isnan(BS_vort_tltd_view)):
            tlt_i, tlt_j = 0, 0
        else:
            if ncyc == 1:
                tlt_flat = nanargmin(BS_vort_tltd_view)
            else:
                tlt_flat = nanargmax(BS_vort_tltd_view)
        tlt_j, tlt_i = unravel_index(tlt_flat, BS_eddy_shape)

        # Keep record of tilt indices
        the_tlt_tki[k] = tlt_i
        the_tlt_tkj[k] = tlt_j

        tlt_xy = (xmidi_par - tlt_i, ymidi_par - tlt_j)

        BS_vort_tltd[k] = roll(BS_vort_tltd[k], tlt_xy, axis=(1, 0)).copy()

        # nsquared_tltd[k] = roll(nsquared_tltd[k], tlt_xy, axis=(1, 0)).copy()
    # ----------------------------------------------------------------------------
    # Do some masking to isolate eddy
    # ----------------------------------------------------------------------------
    try:
        mask2_3d[:] = tile(mask2 + 1, (zsize, 1, 1)).reshape(BS_eddy_3d_shape)
        BlackSea_z3d[:] = repeat(BlackSea_z, x_m_2d.size).reshape(BS_eddy_3d_shape)
    except Exception:
        mask2_3d = (
            tile(mask2 + 1, (zsize, 1, 1)).reshape(BS_eddy_3d_shape).astype(float)
        )
        BlackSea_z3d = repeat(BlackSea_z, x_m_2d.size).reshape(BS_eddy_3d_shape)

    # Model eddies
    t_eddy    = model_xarr["BS_temp_full"].values[:, y_slc, x_slc].copy()
    tP_eddy   = model_xarr["BS_tempP_full"].values[:, y_slc, x_slc].copy()
    s_eddy    = model_xarr["BS_salt_full"].values[:, y_slc, x_slc].copy()
    sP_eddy   = model_xarr["BS_saltP_full"].values[:, y_slc, x_slc].copy()
    rho_eddy  = model_xarr["BS_rho_full"].values[:, y_slc, x_slc].copy()
    rhoP_eddy = model_xarr["BS_rhoP_full"].values[:, y_slc, x_slc].copy()
    u_eddy    = model_xarr["BS_u_full"].values[:, y_slc, x_slc].copy()
    v_eddy    = model_xarr["BS_v_full"].values[:, y_slc, x_slc].copy()
    w_eddy    = model_xarr["BS_w_full"].values[:, y_slc, x_slc].copy()
    vort_eddy = model_xarr["BS_vort_full"].values[:, y_slc, x_slc].copy()

    ssh_eddy  = model_xarr["BS_ssh_full"].values[y_slc, x_slc].copy()
    mld_eddy  = model_xarr["BS_mld_full"].values[y_slc, x_slc].copy()
    navlon_eddy  = model_xarr["BS_navlon_full"].values[y_slc, x_slc].copy()
    navlat_eddy  = model_xarr["BS_navlat_full"].values[y_slc, x_slc].copy()
    topo_eddy = model_xarr["BS_topo"].values[y_slc, x_slc].copy()

    # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
    bac_oxygenconsumptionI_eddy   = model_xarr["BS_bac_oxygenconsumptionI_full"  ].values[y_slc, x_slc].copy()
    bac_oxygenconsumptionIP_eddy  = model_xarr["BS_bac_oxygenconsumptionIP_full" ].values[y_slc, x_slc].copy()

    ZooRespI_eddy                 = model_xarr["BS_ZooRespI_full"                ].values[y_slc, x_slc].copy()
    ZooRespIP_eddy                = model_xarr["BS_ZooRespIP_full"               ].values[y_slc, x_slc].copy()

    NPPOI_eddy                    = model_xarr["BS_NPPOI_full"                   ].values[y_slc, x_slc].copy()
    NPPOIP_eddy                   = model_xarr["BS_NPPOIP_full"                  ].values[y_slc, x_slc].copy()

    OXIDATIONBYDOXI_eddy          = model_xarr["BS_OXIDATIONBYDOXI_full"         ].values[y_slc, x_slc].copy()
    OXIDATIONBYDOXIP_eddy         = model_xarr["BS_OXIDATIONBYDOXIP_full"        ].values[y_slc, x_slc].copy()

    # BIO 2D PTRC (1 vars : airseaoxygenflux)
    airseaoxygenflux_eddy         = model_xarr["BS_AirSeaOxygenFlux_full"        ].values[y_slc, x_slc].copy()
    airseaoxygenfluxP_eddy        = model_xarr["BS_AirSeaOxygenFluxP_full"       ].values[y_slc, x_slc].copy()

    # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
    NPPO_eddy      = model_xarr["BS_NPPO_full"      ].values[:, y_slc, x_slc].copy()
    NPPOP_eddy     = model_xarr["BS_NPPOP_full"     ].values[:, y_slc, x_slc].copy()
    
    ZooResp_eddy  = model_xarr["BS_ZooResp_full"  ].values[:, y_slc, x_slc].copy()
    ZooRespP_eddy = model_xarr["BS_ZooRespP_full" ].values[:, y_slc, x_slc].copy()
    
    doc_eddy       = model_xarr["BS_DOC_full"       ].values[:, y_slc, x_slc].copy()
    docP_eddy      = model_xarr["BS_DOCP_full"      ].values[:, y_slc, x_slc].copy()
    
    # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
    bac_oxygenconsumption_eddy    = model_xarr["BS_bac_oxygenconsumption_full" ].values[:, y_slc, x_slc].copy()
    bac_oxygenconsumptionP_eddy   = model_xarr["BS_bac_oxygenconsumptionP_full"].values[:, y_slc, x_slc].copy()

    OXIDATIONBYDOX_eddy     = model_xarr["BS_OXIDATIONBYDOX_full"  ].values[:, y_slc, x_slc].copy()
    OXIDATIONBYDOXP_eddy    = model_xarr["BS_OXIDATIONBYDOXP_full"].values[:, y_slc, x_slc].copy()
    
    chl_eddy       = model_xarr["BS_CHL_full" ].values[:, y_slc, x_slc].copy()
    chlP_eddy      = model_xarr["BS_CHLP_full"].values[:, y_slc, x_slc].copy()
    
    dox_eddy       = model_xarr["BS_DOX_full" ].values[:, y_slc, x_slc].copy()
    doxP_eddy      = model_xarr["BS_DOXP_full"].values[:, y_slc, x_slc].copy()
    
    nos_eddy       = model_xarr["BS_NOS_full" ].values[:, y_slc, x_slc].copy()
    nosP_eddy      = model_xarr["BS_NOSP_full"].values[:, y_slc, x_slc].copy()
    
    poc_eddy       = model_xarr["BS_POC_full" ].values[:, y_slc, x_slc].copy()
    pocP_eddy      = model_xarr["BS_POCP_full"].values[:, y_slc, x_slc].copy()
    
    par_eddy       = model_xarr["BS_PAR_full" ].values[:, y_slc, x_slc].copy()
    parP_eddy      = model_xarr["BS_PARP_full"].values[:, y_slc, x_slc].copy()
    
    nhs_eddy       = model_xarr["BS_NHS_full" ].values[:, y_slc, x_slc].copy()
    nhsP_eddy      = model_xarr["BS_NHSP_full"].values[:, y_slc, x_slc].copy()

    odu_eddy       = model_xarr["BS_ODU_full" ].values[:, y_slc, x_slc].copy()
    oduP_eddy      = model_xarr["BS_ODUP_full"].values[:, y_slc, x_slc].copy()

    # Setup for triangulations
    _xxx, _yyy = meshgrid(x_m_2d[ymidi_par], y_m_2d[:, xmidi_par])
    _xxx /= eddy.radius[0]
    _yyy /= eddy.radius[0]

    # logging.info('---- _xx:%s, _zzx:%s, _yy:%s, _zzy:%s, _xxx:%s, _yyy:%s' %
    # (_xx, _zzx, _yy, _zzy, _xxx, _yyy))

    tri_xy = Delaunay(array([_xxx.ravel(), _yyy.ravel()]).T)

    # ---------------------------------------------------------------
    # Surface variables
    # ---------------------------------------------------------------
    ssh_eddy[:]      = fill_nans(ssh_eddy)
    ssh_eddy_interp  = interp_midi(tri_xy, ssh_eddy, interpolator, xgrid, ygrid)

    ## mld_eddy
    mld_eddy[:]      = fill_nans(mld_eddy)
    mld_eddy_interp  = interp_midi(tri_xy, mld_eddy, interpolator, xgrid, ygrid)
    ## navlon_eddy
    navlon_eddy[:]      = fill_nans(navlon_eddy)
    navlon_eddy_interp  = interp_midi(tri_xy, navlon_eddy, interpolator, xgrid, ygrid)

    ## navlat_eddy
    navlat_eddy[:]      = fill_nans(navlat_eddy)
    navlat_eddy_interp  = interp_midi(tri_xy, navlat_eddy, interpolator, xgrid, ygrid)
    ## TOPO
    topo_eddy[:]     = fill_nans(topo_eddy)
    topo_eddy_interp = interp_midi(tri_xy, topo_eddy, interpolator, xgrid, ygrid)

    # ---------------------------------------------------------------
    ## BGC 2D
    # ---------------------------------------------------------------
    # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
#    bac_oxygenconsumptionI_eddy_interp  = zeros_like(mask_eddy_interp[0]).astype(float32)
    bac_oxygenconsumptionI_eddy[:]      = fill_nans  (         bac_oxygenconsumptionI_eddy)
    bac_oxygenconsumptionI_eddy_interp  = interp_midi( tri_xy, bac_oxygenconsumptionI_eddy, interpolator, xgrid, ygrid)

#    bac_oxygenconsumptionIP_eddy_interp = zeros_like(mask_eddy_interp[0]).astype(float32)
    bac_oxygenconsumptionIP_eddy[:]     = fill_nans  (         bac_oxygenconsumptionIP_eddy)
    bac_oxygenconsumptionIP_eddy_interp = interp_midi( tri_xy, bac_oxygenconsumptionIP_eddy, interpolator, xgrid, ygrid)
    # ----
#    ZooRespI_eddy_interp                = zeros_like(mask_eddy_interp[0]).astype(float32)
    ZooRespI_eddy[:]                    = fill_nans  (         ZooRespI_eddy)
    ZooRespI_eddy_interp                = interp_midi( tri_xy, ZooRespI_eddy,   interpolator, xgrid, ygrid)
   
#    ZooRespIP_eddy_interp               = zeros_like(mask_eddy_interp[0]).astype(float32)
    ZooRespIP_eddy[:]                   = fill_nans  (         ZooRespIP_eddy)
    ZooRespIP_eddy_interp               = interp_midi( tri_xy, ZooRespIP_eddy, interpolator, xgrid, ygrid)
    # ----
#    NPPOI_eddy_interp                   = zeros_like(mask_eddy_interp[0]).astype(float32)
    NPPOI_eddy[:]                       = fill_nans  (         NPPOI_eddy)
    NPPOI_eddy_interp                   = interp_midi( tri_xy, NPPOI_eddy, interpolator, xgrid, ygrid)

#    NPPOIP_eddy_interp                  = zeros_like(mask_eddy_interp[0]).astype(float32)
    NPPOIP_eddy[:]                      = fill_nans  (         NPPOIP_eddy)
    NPPOIP_eddy_interp                  = interp_midi( tri_xy, NPPOIP_eddy, interpolator, xgrid, ygrid)
    # ----
#    OXIDATIONBYDOXI_eddy_interp         = zeros_like(mask_eddy_interp[0]).astype(float32)
    OXIDATIONBYDOXI_eddy[:]             = fill_nans  (         OXIDATIONBYDOXI_eddy)
    OXIDATIONBYDOXI_eddy_interp         = interp_midi( tri_xy, OXIDATIONBYDOXI_eddy, interpolator, xgrid, ygrid)

#    OXIDATIONBYDOXIP_eddy_interp        = zeros_like(mask_eddy_interp[0]).astype(float32)
    OXIDATIONBYDOXIP_eddy[:]            = fill_nans  (         OXIDATIONBYDOXIP_eddy)
    OXIDATIONBYDOXIP_eddy_interp        = interp_midi( tri_xy, OXIDATIONBYDOXIP_eddy, interpolator, xgrid, ygrid)
    # ----
    # BIO 2D PTRC (1 vars : airseaoxygenflux)
#    airseaoxygenflux_eddy_interp        = zeros_like(mask_eddy_interp[0]).astype(float32)
    airseaoxygenflux_eddy[:]            = fill_nans  (         airseaoxygenflux_eddy)
    airseaoxygenflux_eddy_interp        = interp_midi( tri_xy, airseaoxygenflux_eddy, interpolator, xgrid, ygrid)

#    airseaoxygenfluxP_eddy_interp       = zeros_like(mask_eddy_interp[0]).astype(float32)
    airseaoxygenfluxP_eddy[:]           = fill_nans  (         airseaoxygenfluxP_eddy)
    airseaoxygenfluxP_eddy_interp       = interp_midi( tri_xy, airseaoxygenfluxP_eddy, interpolator, xgrid, ygrid)

    # ---------------------------------------------------------------
    # 3D variables
    # ---------------------------------------------------------------
    mask_eddy_interp  = empty((BlackSea_z.size, ygrid[:, 0].size, xgrid[0].size)).astype(
        int8
    )
    
    vort_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    temp_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    salt_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    rho_eddy_interp   = zeros_like(mask_eddy_interp).astype(float32)
    u_eddy_interp     = zeros_like(mask_eddy_interp).astype(float32)
    v_eddy_interp     = zeros_like(mask_eddy_interp).astype(float32)
    w_eddy_interp     = zeros_like(mask_eddy_interp).astype(float32)

    tempP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)
    saltP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)
    rhoP_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)

    # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
    NPPO_eddy_interp      = zeros_like(mask_eddy_interp).astype(float32)
    NPPOP_eddy_interp     = zeros_like(mask_eddy_interp).astype(float32)
 
    ZooResp_eddy_interp   = zeros_like(mask_eddy_interp).astype(float32)
    ZooRespP_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)

    doc_eddy_interp       = zeros_like(mask_eddy_interp).astype(float32)
    docP_eddy_interp      = zeros_like(mask_eddy_interp).astype(float32)

    # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
    bac_oxygenconsumption_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    bac_oxygenconsumptionP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    OXIDATIONBYDOX_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    OXIDATIONBYDOXP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    chl_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    chlP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    dox_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    doxP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    nos_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    nosP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    poc_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    pocP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    par_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    parP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    nhs_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    nhsP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)

    odu_eddy_interp  = zeros_like(mask_eddy_interp).astype(float32)
    oduP_eddy_interp = zeros_like(mask_eddy_interp).astype(float32)
       
    # Loop over depth
    for k in range(zsize):

        mask_eddy_interp[k] = interp_midi_mask(
            tri_xy, mask_eddy[k], interpolator_nn, xgrid, ygrid
        )
        # VORT
        vort_eddy[k] = fill_nans(vort_eddy[k])
        vort_eddy_interp[k] = interp_midi(
            tri_xy, vort_eddy[k], interpolator, xgrid, ygrid
        )
        # TEMP
        t_eddy[k] = fill_nans(t_eddy[k])
        temp_eddy_interp[k] = interp_midi(tri_xy, t_eddy[k], interpolator, xgrid, ygrid)
        # TEMP P
        tP_eddy[k] = fill_nans(tP_eddy[k])
        tempP_eddy_interp[k] = interp_midi(
            tri_xy, tP_eddy[k], interpolator, xgrid, ygrid
        )
        # SALT
        s_eddy[k] = fill_nans(s_eddy[k])
        salt_eddy_interp[k] = interp_midi(tri_xy, s_eddy[k], interpolator, xgrid, ygrid)
        # SALT P
        sP_eddy[k] = fill_nans(sP_eddy[k])
        saltP_eddy_interp[k] = interp_midi(
            tri_xy, sP_eddy[k], interpolator, xgrid, ygrid
        )
        # RHO
        rho_eddy[k] = fill_nans(rho_eddy[k])
        rho_eddy_interp[k] = interp_midi(
            tri_xy, rho_eddy[k], interpolator, xgrid, ygrid
        )
        # RHO P
        rhoP_eddy[k] = fill_nans(rhoP_eddy[k])
        rhoP_eddy_interp[k] = interp_midi(
            tri_xy, rhoP_eddy[k], interpolator, xgrid, ygrid
        )
        # U
        u_eddy[k] = fill_nans(u_eddy[k])
        u_eddy_interp[k] = interp_midi(tri_xy, u_eddy[k], interpolator, xgrid, ygrid)
        # V
        v_eddy[k] = fill_nans(v_eddy[k])
        v_eddy_interp[k] = interp_midi(tri_xy, v_eddy[k], interpolator, xgrid, ygrid)
        # W
        w_eddy[k] = fill_nans(w_eddy[k])
        w_eddy_interp[k] = interp_midi(tri_xy, w_eddy[k], interpolator, xgrid, ygrid)

        # ----------------------------------------------------
        dox_eddy[k]         = fill_nans(dox_eddy[k])
        dox_eddy_interp[k]  = interp_midi(tri_xy, dox_eddy[k], interpolator, xgrid, ygrid)
        #
        doxP_eddy[k]        = fill_nans(doxP_eddy[k])
        doxP_eddy_interp[k] = interp_midi(tri_xy, doxP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        nos_eddy[k]         = fill_nans(nos_eddy[k])
        nos_eddy_interp[k]  = interp_midi(tri_xy, nos_eddy[k], interpolator, xgrid, ygrid)
        #
        nosP_eddy[k]        = fill_nans(nosP_eddy[k])
        nosP_eddy_interp[k] = interp_midi(tri_xy, nosP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        poc_eddy[k]         = fill_nans(poc_eddy[k])
        poc_eddy_interp[k]  = interp_midi(tri_xy, poc_eddy[k], interpolator, xgrid, ygrid)
        #
        pocP_eddy[k]        = fill_nans(pocP_eddy[k])
        pocP_eddy_interp[k] = interp_midi(tri_xy, pocP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        par_eddy[k]         = fill_nans(par_eddy[k])
        par_eddy_interp[k]  = interp_midi(tri_xy, par_eddy[k], interpolator, xgrid, ygrid)
        #
        parP_eddy[k]        = fill_nans(parP_eddy[k])
        parP_eddy_interp[k] = interp_midi(tri_xy, parP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        nhs_eddy[k] = fill_nans(nhs_eddy[k])
        nhs_eddy_interp[k] = interp_midi(tri_xy, nhs_eddy[k], interpolator, xgrid, ygrid)
        #
        nhsP_eddy[k] = fill_nans(nhsP_eddy[k])
        nhsP_eddy_interp[k] = interp_midi(tri_xy, nhsP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        odu_eddy[k] = fill_nans(odu_eddy[k])
        odu_eddy_interp[k] = interp_midi(tri_xy, odu_eddy[k], interpolator, xgrid, ygrid)
        #
        oduP_eddy[k] = fill_nans(oduP_eddy[k])
        oduP_eddy_interp[k] = interp_midi(tri_xy, oduP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        NPPO_eddy[k] = fill_nans(NPPO_eddy[k])
        NPPO_eddy_interp[k] = interp_midi(tri_xy, NPPO_eddy[k], interpolator, xgrid, ygrid)
        #
        NPPOP_eddy[k] = fill_nans(NPPOP_eddy[k])
        NPPOP_eddy_interp[k] = interp_midi(tri_xy, NPPOP_eddy[k], interpolator, xgrid, ygrid)
        # ----------------------------------------------------
        chl_eddy[k] = fill_nans(chl_eddy[k])
        chl_eddy_interp[k] = interp_midi(
            tri_xy, chl_eddy[k], interpolator, xgrid, ygrid
        )
        chlP_eddy[k] = fill_nans(chlP_eddy[k])
        chlP_eddy_interp[k] = interp_midi(
            tri_xy, chlP_eddy[k], interpolator, xgrid, ygrid
        )
        # ----------------------------------------------------
        doc_eddy[k] = fill_nans(doc_eddy[k])
        doc_eddy_interp[k] = interp_midi(
            tri_xy, doc_eddy[k], interpolator, xgrid, ygrid
        )
        docP_eddy[k] = fill_nans(docP_eddy[k])
        docP_eddy_interp[k] = interp_midi(
            tri_xy, docP_eddy[k], interpolator, xgrid, ygrid
        )
        # ----------------------------------------------------
        bac_oxygenconsumption_eddy[k] = fill_nans(bac_oxygenconsumption_eddy[k])
        bac_oxygenconsumption_eddy_interp[k] = interp_midi(
                tri_xy, bac_oxygenconsumption_eddy[k], interpolator, xgrid, ygrid
        )
        bac_oxygenconsumptionP_eddy[k] = fill_nans(bac_oxygenconsumptionP_eddy[k])
        bac_oxygenconsumptionP_eddy_interp[k] = interp_midi(
            tri_xy, bac_oxygenconsumptionP_eddy[k], interpolator, xgrid, ygrid
        )
        # ----------------------------------------------------
        OXIDATIONBYDOX_eddy[k] = fill_nans(OXIDATIONBYDOX_eddy[k])
        OXIDATIONBYDOX_eddy_interp[k] = interp_midi(
            tri_xy, OXIDATIONBYDOX_eddy[k], interpolator, xgrid, ygrid
        )
        OXIDATIONBYDOXP_eddy[k] = fill_nans(OXIDATIONBYDOXP_eddy[k])
        OXIDATIONBYDOXP_eddy_interp[k] = interp_midi(
            tri_xy, OXIDATIONBYDOXP_eddy[k], interpolator, xgrid, ygrid
        )
        # ----------------------------------------------------
        ZooResp_eddy[k] = fill_nans(ZooResp_eddy[k])
        ZooResp_eddy_interp[k] = interp_midi(
            tri_xy, ZooResp_eddy[k], interpolator, xgrid, ygrid
        )
        ZooRespP_eddy[k] = fill_nans(ZooRespP_eddy[k])
        ZooRespP_eddy_interp[k] = interp_midi(
                tri_xy, ZooRespP_eddy[k], interpolator, xgrid, ygrid
        )
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Get indices to surface masked points
        mask_k = (mask_eddy_interp[k].ravel() == 0).nonzero()[0]
        if len(mask_k):
            if not k:
                ssh_eddy_interp[:].ravel()[mask_k] = nan
                mld_eddy_interp[:].ravel()[mask_k] = nan
                navlon_eddy_interp[:].ravel()[mask_k] = nan
                navlat_eddy_interp[:].ravel()[mask_k] = nan
                topo_eddy_interp[:].ravel()[mask_k] = nan
                ZooRespI_eddy_interp[:].ravel()[mask_k] = nan
                ZooRespIP_eddy_interp[:].ravel()[mask_k] = nan
                NPPOI_eddy_interp[:].ravel()[mask_k] = nan
                NPPOIP_eddy_interp[:].ravel()[mask_k] = nan
                OXIDATIONBYDOXI_eddy_interp[:].ravel()[mask_k] = nan
                OXIDATIONBYDOXIP_eddy_interp[:].ravel()[mask_k] = nan
                airseaoxygenflux_eddy_interp[:].ravel()[mask_k] = nan
                airseaoxygenfluxP_eddy_interp[:].ravel()[mask_k] = nan
                bac_oxygenconsumptionI_eddy_interp[:].ravel()[mask_k] = nan
                bac_oxygenconsumptionIP_eddy_interp[:].ravel()[mask_k] = nan

            vort_eddy_interp[k].ravel()[mask_k] = nan
            temp_eddy_interp[k].ravel()[mask_k] = nan
            salt_eddy_interp[k].ravel()[mask_k] = nan
            rho_eddy_interp[k].ravel()[mask_k] = nan
            tempP_eddy_interp[k].ravel()[mask_k] = nan
            saltP_eddy_interp[k].ravel()[mask_k] = nan
            rhoP_eddy_interp[k].ravel()[mask_k] = nan
            u_eddy_interp[k].ravel()[mask_k] = nan
            v_eddy_interp[k].ravel()[mask_k] = nan
            w_eddy_interp[k].ravel()[mask_k] = nan
            
            chl_eddy_interp[k].ravel()[mask_k] = nan
            dox_eddy_interp[k].ravel()[mask_k] = nan
            doc_eddy_interp[k].ravel()[mask_k] = nan
            nos_eddy_interp[k].ravel()[mask_k] = nan
            poc_eddy_interp[k].ravel()[mask_k] = nan
            par_eddy_interp[k].ravel()[mask_k] = nan
            nhs_eddy_interp[k].ravel()[mask_k] = nan
            odu_eddy_interp[k].ravel()[mask_k] = nan
            ZooResp_eddy_interp[k].ravel()[mask_k] = nan
            OXIDATIONBYDOX_eddy_interp[k].ravel()[mask_k] = nan
            bac_oxygenconsumption_eddy_interp[k].ravel()[mask_k] = nan
            NPPO_eddy_interp[k].ravel()[mask_k] = nan

            chlP_eddy_interp[k].ravel()[mask_k] = nan
            doxP_eddy_interp[k].ravel()[mask_k] = nan
            docP_eddy_interp[k].ravel()[mask_k] = nan
            nosP_eddy_interp[k].ravel()[mask_k] = nan
            pocP_eddy_interp[k].ravel()[mask_k] = nan
            parP_eddy_interp[k].ravel()[mask_k] = nan
            nhsP_eddy_interp[k].ravel()[mask_k] = nan
            oduP_eddy_interp[k].ravel()[mask_k] = nan
            ZooRespP_eddy_interp[k].ravel()[mask_k] = nan
            OXIDATIONBYDOXP_eddy_interp[k].ravel()[mask_k] = nan
            bac_oxygenconsumptionP_eddy_interp[k].ravel()[mask_k] = nan
            NPPOP_eddy_interp[k].ravel()[mask_k] = nan

    time_ctypes[eddy_iter] = eddy.date[0]
    track_ctypes[eddy_iter] = eddy.track[0]
    n_ctypes[eddy_iter] = eddy.n[0]
    centlon_ctypes[eddy_iter] = eddy.centlon[0]
    centlat_ctypes[eddy_iter] = eddy.centlat[0]
    radius_ctypes[eddy_iter] = eddy.radius[0]
    radius_eff_ctypes[eddy_iter] = eddy.radius_e[0]
    amplitude_ctypes[eddy_iter] = eddy.amplitude[0]
    virtual_ctypes[eddy_iter] = eddy.virtual[0]

    i0i1, j0j1 = eddy_iter * 50, (eddy_iter * 50) + 50

    contour_lon_s_ctypes[i0i1:j0j1] = cntr.contour_lon_s
    contour_lat_s_ctypes[i0i1:j0j1] = cntr.contour_lat_s
    contour_lon_e_ctypes[i0i1:j0j1] = cntr.contour_lon_e
    contour_lat_e_ctypes[i0i1:j0j1] = cntr.contour_lat_e

    i0i1_j0j1_slc = stack((x_slc.start, x_slc.stop, y_slc.start, y_slc.stop))
    i0i1, j0j1 = eddy_iter * 4, (eddy_iter * 4) + 4
    i0i1_j0j1_ctypes[i0i1:j0j1] = i0i1_j0j1_slc

    tilt_slc = slice(eddy_iter, eddy_iter + BlackSea_z.size)
    tilt_i_ctypes[tilt_slc] = the_tlt_tki
    tilt_j_ctypes[tilt_slc] = the_tlt_tkj

    _xyz_size = xgrid.size * zsize
    _xy_size = xgrid.size
    # _yz_size = xgrid[:, 0].size * zsize

    xyz_slc = slice(eddy_iter * _xyz_size, (eddy_iter * _xyz_size) + _xyz_size)
    xy_slc = slice(eddy_iter * _xy_size, (eddy_iter * _xy_size) + _xy_size)

    #-----------------------------------------------------------------------------
    ssh_interp_ctypes[xy_slc] = ssh_eddy_interp.ravel()
    mld_interp_ctypes[xy_slc] = mld_eddy_interp.ravel()
    navlon_interp_ctypes[xy_slc] = navlon_eddy_interp.ravel()
    navlat_interp_ctypes[xy_slc] = navlat_eddy_interp.ravel()
    topo_interp_ctypes[xy_slc] = topo_eddy_interp.ravel()

    # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
    bac_oxygenconsumptionI_interp_ctypes  [xy_slc] = bac_oxygenconsumptionI_eddy_interp.ravel()
    bac_oxygenconsumptionIP_interp_ctypes[xy_slc] = bac_oxygenconsumptionIP_eddy_interp.ravel()

    ZooRespI_interp_ctypes[  xy_slc]        = ZooRespI_eddy_interp.ravel()
    ZooRespIP_interp_ctypes[xy_slc]         = ZooRespIP_eddy_interp.ravel()

    NPPOI_interp_ctypes[  xy_slc]           = NPPOI_eddy_interp.ravel()
    NPPOIP_interp_ctypes[xy_slc]            = NPPOIP_eddy_interp.ravel()

    OXIDATIONBYDOXI_interp_ctypes[  xy_slc] = OXIDATIONBYDOXI_eddy_interp.ravel()
    OXIDATIONBYDOXIP_interp_ctypes[xy_slc]  = OXIDATIONBYDOXIP_eddy_interp.ravel()

    # BIO 2D PTRC (1 vars : airseaoxygenflux)
    airseaoxygenflux_interp_ctypes[ xy_slc] = airseaoxygenflux_eddy_interp.ravel()    
    airseaoxygenfluxP_interp_ctypes[xy_slc] = airseaoxygenfluxP_eddy_interp.ravel()

    # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
    nppo_interp_ctypes[      xyz_slc] = NPPO_eddy_interp.ravel()
    nppoP_interp_ctypes[     xyz_slc] = NPPOP_eddy_interp.ravel()

    ZooResp_interp_ctypes[ xyz_slc]  = ZooResp_eddy_interp.ravel()
    ZooRespP_interp_ctypes[xyz_slc]  = ZooRespP_eddy_interp.ravel()

    doc_interp_ctypes[      xyz_slc]  = doc_eddy_interp.ravel()
    docP_interp_ctypes[     xyz_slc]  = docP_eddy_interp.ravel()

    # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)    
    bac_oxygenconsumption_interp_ctypes[ xyz_slc] = bac_oxygenconsumption_eddy_interp.ravel()
    bac_oxygenconsumptionP_interp_ctypes[xyz_slc] = bac_oxygenconsumptionP_eddy_interp.ravel()

    oxidationbydox_interp_ctypes[xyz_slc]  = OXIDATIONBYDOX_eddy_interp.ravel()
    oxidationbydoxP_interp_ctypes[xyz_slc] = OXIDATIONBYDOXP_eddy_interp.ravel()
    
    chl_interp_ctypes[ xyz_slc] = chl_eddy_interp.ravel()
    chlP_interp_ctypes[xyz_slc] = chlP_eddy_interp.ravel()

    dox_interp_ctypes[ xyz_slc] = dox_eddy_interp.ravel()
    doxP_interp_ctypes[xyz_slc] = doxP_eddy_interp.ravel()

    nos_interp_ctypes[ xyz_slc] = nos_eddy_interp.ravel()
    nosP_interp_ctypes[xyz_slc] = nosP_eddy_interp.ravel()

    poc_interp_ctypes[ xyz_slc] = poc_eddy_interp.ravel()
    pocP_interp_ctypes[xyz_slc] = pocP_eddy_interp.ravel()

    par_interp_ctypes[ xyz_slc] = par_eddy_interp.ravel()
    parP_interp_ctypes[xyz_slc] = parP_eddy_interp.ravel()

    nhs_interp_ctypes[ xyz_slc] = nhs_eddy_interp.ravel()
    nhsP_interp_ctypes[xyz_slc] = nhsP_eddy_interp.ravel()
    
    odu_interp_ctypes[ xyz_slc] = odu_eddy_interp.ravel()
    oduP_interp_ctypes[xyz_slc] = oduP_eddy_interp.ravel()

    # end of bio #
    
    mask_interp_ctypes[xyz_slc] = mask_eddy_interp.ravel()
    
    temp_interp_ctypes[xyz_slc] = temp_eddy_interp.ravel()
    salt_interp_ctypes[xyz_slc] = salt_eddy_interp.ravel()
    rho_interp_ctypes[xyz_slc] = rho_eddy_interp.ravel()
    vort_interp_ctypes[xyz_slc] = vort_eddy_interp.ravel()
    
    tempP_interp_ctypes[xyz_slc] = tempP_eddy_interp.ravel()
    saltP_interp_ctypes[xyz_slc] = saltP_eddy_interp.ravel()
    rhoP_interp_ctypes[xyz_slc] = rhoP_eddy_interp.ravel()
    
    u_interp_ctypes[xyz_slc] = u_eddy_interp.ravel()
    v_interp_ctypes[xyz_slc] = v_eddy_interp.ravel()
    w_interp_ctypes[xyz_slc] = w_eddy_interp.ravel()

    print(
        "Leaving for \n#### %s #### Lon=%s #### Lat=%s ###############"
        % (the_date_str, eddy.centlon[0], eddy.centlat[0])
    )
    return


if __name__ == "__main__":

    # TODO Add MLD (use SSH as template); compression; core depth?; ****DONE
    # TODO Change 'eddy_id' to standard 'track' and 'n' ****DONE
    # TODO Filter `radii` to avoid sudden zooms in radius coordinates ****DONE
    # TODO Add `U_x` and 'V_y' variables ...?
    # TODO Add CMEMS satellite data - SST and CHL ...? ****DONE
    # TODO Add CMEMS altimetric data -  ****DONE
    # TODO Add variable for Virtual obs  ****DONE  ****DONE
    # TODO Add variable for topo median  ****DONE

    use_MP = True
    L_filt_loess = True  # Loess filter for radii

    ACYC = True  # Anticyclones or cyclones

    # BS_dir = '/marula/emason/BlackSea/SSH-U-V_fields/'
    BS_dir = "/scratch/ulg/mast/emason/daily_MYP_input_files/Daily_model_files/"
    BS_dir = "/scratch/ulg/mast/acapet/Compout/"

    T_files = "T_files/"
    U_files = "U_files/"
    V_files = "V_files/"
    W_files = "W_files/"
    P_files = "P_files/"
    # CHL_files = "P_CHL_files/"

    BlackSea_T_files = sorted(glob(BS_dir + T_files + "BlackSea_model_T_????????.nc"))

    # BS_bathy_file = '/marula/emason/BlackSea/bathy_meter.nc'
    BS_bathy_file = "/scratch/ulg/mast/emason/daily_MYP_input_files/bathy_meter.nc"

    # mesh_mask_dir = "/scratch/ulg/mast/acapet/FromLuc/NEMO31/"
    # mesh_mask_dir = "/home/ulg/mast/lvandenb/bsmfc/Nemo3.6/NEMOGCM/CONFIG/BSFS_MYP/GEO/"
    mesh_mask_dir = "/scratch/ulg/mast/emason/daily_MYP_input_files/"

    # eddy_dir = "/home/ulg/mast/emason/BS_surface_fields/"
    eddy_dir = "/scratch/ulg/mast/emason/daily_MYP_input_files/"
    # eddy_dir = "/scratch/ulg/mast/emason/eddyIDs_and_tracks/"

    eddy_tracking_output_file = "Anticyclonic.nc" if ACYC else "Cyclonic.nc"

    # Produced 'output_20.nc' with make_ITE_loess_radius.py
    eddy_filtered_L = (
        "output_loess_ACYC_20_2022version.nc" if ACYC else "output_loess_CYC_20.nc"
    )

    # Topography
    topo_dir = BS_dir
    topo_file = "grid_spec.nc"

    # dists_file = 'dist_to_GSHHG_v2.3.7_1m.nc'

    savedir = BS_dir

    with open_dataset(BlackSea_T_files[0], decode_times=False) as ds:
        lon1d = unique(ds["nav_lon"])  # [1:]
        lat1d = unique(ds["nav_lat"])  # [1:]
    lonmin, lonmax, latmin, latmax = (
        lon1d.min(),
        lon1d.max(),
        lat1d.min(),
        lat1d.max(),
    )

    start_date, end_date = (runyear, runmonth,1), (runyear, 12, 31)

    # assert start_date[-1] == 1, "It's better to start on first day of month"

    the_save_file = (
        "BlackSea_ACYC_composites_%s.nc" if ACYC else "BlackSea_CCYC_composites_%s.nc"
    )

    m1_radius = 1.75  # 1.25
    tree_radius = 2.5  # Initially used 1

    make_figures = False

    vort_space = linspace(-1.5, 1.5, 151)

    spacing = 0.025  # Use this
    xgrid = arange(-4, 4 + spacing, spacing)

    restart = False

    fill_value = float32(1e18)
    fill_value_32 = float32(1e18)
    fill_value_64 = float64(1e18)
    fill_value_16 = int16(1e15)
    fill_value_8 = int8(126)

    ncpu = ncpus_arg #6 # cpu_count() #- 8

    # PRIMES: sigma = desired wavelength * 0.125 / model.resolution
    sigma_lo = 6 * 0.125 / (1 / 12.0)
    sigma_hi = 0.5 * 0.125 / (1 / 12.0)
    gmode = "nearest"

    # ---------------------------------------------------------------------------
    # ---- END user options
    # ---------------------------------------------------------------------------

    plt.close("all")

    ncyc = 1 if ACYC else 0

    # Placeholder
    BlackSea_file = BS_dir + "T_files/BlackSea_model_T_????????.nc"

    if spacing != 0.025:
        print("######## CHECK SPACING ................... %s ########" % spacing)

    assert "%s" in the_save_file, "%s must be in `the_save_file`"

    assert xgrid.size % 2 != 0, "xmidi must be odd number"
    xmidi = abs(xgrid).argmin()
    ymidi = abs(xgrid[1:-1]).argmin()

    xgrid, ygrid = meshgrid(xgrid, xgrid[1:-1])
    # cgrid = (xgrid ** 2 + ygrid ** 2) ** 0.5
    # cgrid_mask = cgrid <= 1.5

    # start_CHL_mean = True

    current_month = -9999

    # Read in eddy tracks and sort in date order
    domain = (lonmin, lonmax, latmin, latmax)
    time_domain = (start_date, end_date)

    if L_filt_loess:
        eddy_filtered_L = BS_dir + eddy_filtered_L
        with Dataset(eddy_filtered_L) as nc:
            loess_L = nc.variables["speed_radius"][:]
    else:
        loess_L = False

    print("---- DONE: loading eddy tracker files")
    etrk_full_period = EddyTrack(
        eddy_dir, eddy_tracking_output_file, time_domain, domain, L_smooth=loess_L
    )

    start_date_str = (
        str(start_date).replace(", ", "-").replace("(", "").replace(")", "")
    )
    end_date_str = str(end_date).replace(", ", "-").replace("(", "").replace(")", "")

    model_dates = date2num(date_range(start=start_date_str, end=end_date_str))

    assert (diff(model_dates) == diff(model_dates)[0]).all(), "bad file somewhere"
    print("---- DONE: dates sorted")

    with Dataset(mesh_mask_dir + "mesh_mask_31levels.nc") as nc:
        BlackSea_z = nc.variables["nav_lev"][:]
        BlackSea_mask3d = nc.variables["fmask"][0]

    BlackSea_lon, BlackSea_lat = meshgrid(lon1d, lat1d)
    print("TTTTTTTTTTTTTT", BlackSea_lon.shape, BlackSea_lat.shape)

    with Dataset(BS_bathy_file) as nc:
        topo = nc.variables["Bathymetry"][:].squeeze()

    size_2d = BlackSea_lon.size
    print("BlackSea_lon.size", BlackSea_lon.size)
    BlackSea_2d_shape = BlackSea_lon.shape
    print("BlackSea_2d_shape", BlackSea_lon.shape)
    J, I = unravel_index(arange(size_2d), BlackSea_2d_shape)

    pm, pn = get_pm_pn(BlackSea_lon, BlackSea_lat)

    # -----------------------------------------------------------------
    xsize = xgrid[0].size
    ysize = ygrid[:, 0].size

    # BlackSea_z = BlackSea_z[z_slc]
    zsize = BlackSea_z.size
    dz_r = BlackSea_z[1:] - BlackSea_z[:-1]
    BlackSea_z_min, BlackSea_z_max = BlackSea_z.min(), BlackSea_z.max()

    xvalues_out, xvalues_depth = meshgrid(xgrid[0], BlackSea_z)
    yvalues_out, yvalues_depth = meshgrid(ygrid[:, 0], BlackSea_z)

    # Loop over eddy observations
    ref_date = -1.0

    print("BlackSea_mask3d.shape", BlackSea_mask3d.shape)
    # Set up model xarray
    data_vars = {
        
        # 3D
        "BS_mask3d_full" : (("z", "y", "x"), BlackSea_mask3d),
        "BS_salt_full"   : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_saltP_full"  : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_temp_full"   : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_tempP_full"  : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_rho_full"    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_rhoP_full"   : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_u_full"      : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_v_full"      : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_w_full"      : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_vort_full"   : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        
        # 2D
        "BS_ssh_full": (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_mld_full": (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_navlon_full": (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_navlat_full": (("y", "x"), BlackSea_mask3d[0].astype(float64)),

        # 3D
        "BS_DOX_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_DOXP_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        
        "BS_NOS_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_NOSP_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        
        "BS_PAR_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_PARP_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_POC_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_POCP_full": (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_CHL_full"                     : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_CHLP_full"                    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_NHS_full"                     : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_NHSP_full"                    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_ODU_full"                     : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_ODUP_full"                    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_bac_oxygenconsumption_full"   : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_bac_oxygenconsumptionP_full"  : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_OXIDATIONBYDOX_full"          : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_OXIDATIONBYDOXP_full"        : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        "BS_NPPO_full"                    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_NPPOP_full"                  : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        
        "BS_ZooResp_full"                : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_ZooRespP_full"               : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        
        "BS_DOC_full"                     : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),
        "BS_DOCP_full"                    : (("z", "y", "x"), BlackSea_mask3d.astype(float64)),

        # 2D
        "BS_ZooRespI_full"                : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_ZooRespIP_full"               : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        
        "BS_bac_oxygenconsumptionI_full"  : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_bac_oxygenconsumptionIP_full" : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        
        "BS_OXIDATIONBYDOXI_full"         : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_OXIDATIONBYDOXIP_full"        : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        
        "BS_NPPOI_full"                   : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_NPPOIP_full"                  : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        
        "BS_AirSeaOxygenFlux_full"        : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        "BS_AirseaOxygenfluxP_full"       : (("y", "x"), BlackSea_mask3d[0].astype(float64)),
        
        "BS_lon2d_full": (("y", "x"), BlackSea_lon),
        "BS_lat2d_full": (("y", "x"), BlackSea_lat),
        "BS_dx"        : (("y", "x"), zeros_like(BlackSea_lon)),
        "BS_dy"        : (("y", "x"), zeros_like(BlackSea_lon)),
        "BS_coriolis"  : (("y", "x"), zeros_like(BlackSea_lon)),
        "BS_topo"      : (("y", "x"), zeros_like(BlackSea_lon)),
        
        # 1D
        "date"     : (("t"), array([0.0])),
        "centlon"  : (("t"), array([0.0])),
        "centlat"  : (("t"), array([0.0])),
        "radius"   : (("t"), array([0.0])),
        "radius_e" : (("t"), array([0.0])),
        "amplitude": (("t"), array([0.0])),
        "virtual"  : (("t"), array([0.0])),
        "track"    : (("t"), array([0])),
        "n"        : (("t"), array([0])),
        "cyc"      : (("t"), array([0])),
    }

    coords = {
        "lon_full": BlackSea_lon[0],  # - 360,
        "lat_full": BlackSea_lat[:, 0],
        "z": BlackSea_z,
        "t": array([0]),
        "one": array([0]),
        "four": linspace(0, 1, 4),
        "fifty": linspace(0, 1, 50),
        "x_compo": xgrid[0],
        "y_compo": ygrid[:, 0],
    }

    model_xarr = xr_Dataset(data_vars=data_vars, coords=coords)

    # -------------------------------------------------------------------------------------

    start = True

    for tind, model_date in enumerate(model_dates):
        if (
            model_date < etrk_full_period.dates().min()
            or model_date > etrk_full_period.dates().max()
        ):
            continue
        else:
            print("#####", model_date, etrk_full_period.dates()[0])

        the_date = num2date(model_date).date()
        the_month = the_date.month
        the_year = the_date.year

        # Create monthly output files
        if current_month != the_month:
            current_month = the_month
            save_file = the_save_file % "".join(
                (str(the_year), str(the_month).zfill(2))
            )
            with Dataset(
                savedir + save_file, "w", clobber=True, format="NETCDF4"
            ) as nc:
                create_nc(
                    nc,
                    xsize,
                    ysize,
                    zsize,
                    BlackSea_z,
                    xgrid[0],
                    ygrid[:, 0],
                    xmidi,
                    ymidi,
                    BS_dir,
                    eddy_dir,
                    savedir,
                    BlackSea_T_files[0],
                    eddy_tracking_output_file,
                    save_file,
                    fill_value_32,
                    fill_value_64,
                    BlackSea_lon[0],
                    BlackSea_lat[:, 0],                 
                )

            # Read in eddy tracks and sort in date order
            start_month_date = (the_year, the_month, num2date(model_date).day)
            # https://code.activestate.com/recipes/476197-first-last-day-of-the-month/
            end_month_date = the_date + relativedelta(day=31)
            end_month_date = (the_year, the_month, end_month_date.day)
            time_month_domain = (start_month_date, end_month_date)

            etrk_data = EddyTrack(
                eddy_dir,
                eddy_tracking_output_file,
                time_month_domain,
                domain,
                L_smooth=loess_L,
            )
            # Sort anticyclones
            (
                eddy_observation_list,
                eddy_dates,
                eddy_sorted_i,
                eddy_unsorted_i,
                cntr_observation_list,
            ) = get_eddy_recarray(etrk_data)

        # Get indices to eddy_observation_list for current day
        # This tells us the time index to save the data once processed
        the_eddy_days_i = hstack(
            [True if d.date == model_date else False for d in eddy_observation_list]
        ).nonzero()[0]
        if not the_eddy_days_i.size:
            continue
        the_eddy_day_i = arange(the_eddy_days_i.size)
        # sasa
        # Make shared arrays to hold results
        time_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        track_ctypes = sharedctypes.RawArray("i", the_eddy_days_i.size)
        n_ctypes = sharedctypes.RawArray("i", the_eddy_days_i.size)
        centlon_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        centlat_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        radius_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        radius_eff_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        amplitude_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)
        virtual_ctypes = sharedctypes.RawArray("i", the_eddy_days_i.size)
        cyc_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size)

        i0i1_j0j1_ctypes = sharedctypes.RawArray("i", the_eddy_days_i.size * 4)
        tilt_i_ctypes = sharedctypes.RawArray(
            "i", the_eddy_days_i.size * BlackSea_z.size
        )
        tilt_j_ctypes = sharedctypes.RawArray(
            "i", the_eddy_days_i.size * BlackSea_z.size
        )
        contour_lon_s_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size * 50)
        contour_lat_s_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size * 50)
        contour_lon_e_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size * 50)
        contour_lat_e_ctypes = sharedctypes.RawArray("d", the_eddy_days_i.size * 50)



        time_ctypes[:] = memoryview(full_like(the_eddy_days_i, fill_value, dtype=float))
        track_ctypes[:] = memoryview(full_like(the_eddy_days_i, fill_value, dtype=int))
        n_ctypes[:] = memoryview(full_like(the_eddy_days_i, fill_value, dtype=int))
        centlon_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=float)
        )
        centlat_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=float)
        )
        radius_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=float)
        )
        radius_eff_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=float)
        )
        amplitude_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=float)
        )
        virtual_ctypes[:] = memoryview(
            full_like(the_eddy_days_i, fill_value, dtype=int)
        )
        cyc_ctypes[:] = memoryview(full_like(the_eddy_days_i, fill_value, dtype=int))

        i0i1_j0j1_ctypes[:] = memoryview(
            full_like(i0i1_j0j1_ctypes, fill_value, dtype=int)
        )
        tilt_i_ctypes[:] = memoryview(full_like(tilt_i_ctypes, fill_value, dtype=int))
        tilt_j_ctypes[:] = memoryview(full_like(tilt_j_ctypes, fill_value, dtype=int))
        contour_lon_s_ctypes[:] = memoryview(
            full_like(contour_lon_s_ctypes, fill_value, dtype=float)
        )
        contour_lat_s_ctypes[:] = memoryview(
            full_like(contour_lat_s_ctypes, fill_value, dtype=float)
        )
        contour_lon_e_ctypes[:] = memoryview(
            full_like(contour_lon_e_ctypes, fill_value, dtype=float)
        )
        contour_lat_e_ctypes[:] = memoryview(
            full_like(contour_lat_e_ctypes, fill_value, dtype=float)
        )

        # Set array sizes
        xy_size = xgrid.size * the_eddy_days_i.size
        xyz_size = xgrid.size * zsize * the_eddy_days_i.size
        # yz_size = ygrid[:, 0].size * zsize * the_eddy_days_i.size
        # xyz_size = xgrid.size * zsize * the_eddy_days_i.size

        print("---- Setting up `sharedctypes` arrays")
        ssh_interp_ctypes   = sharedctypes.RawArray("d", xy_size)
        mld_interp_ctypes   = sharedctypes.RawArray("d", xy_size)
        navlon_interp_ctypes   = sharedctypes.RawArray("d", xy_size)
        navlat_interp_ctypes   = sharedctypes.RawArray("d", xy_size)
        topo_interp_ctypes  = sharedctypes.RawArray("d", xy_size)
        mask_interp_ctypes  = sharedctypes.RawArray("i", xyz_size)
        temp_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        salt_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        vort_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        rho_interp_ctypes   = sharedctypes.RawArray("d", xyz_size)
        tempP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        saltP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        rhoP_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        v_interp_ctypes     = sharedctypes.RawArray("d", xyz_size)
        u_interp_ctypes     = sharedctypes.RawArray("d", xyz_size)
        w_interp_ctypes     = sharedctypes.RawArray("d", xyz_size)


        # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
        bac_oxygenconsumptionI_interp_ctypes = sharedctypes.RawArray("d", xy_size)
        bac_oxygenconsumptionIP_interp_ctypes = sharedctypes.RawArray("d", xy_size)

        ZooRespI_interp_ctypes = sharedctypes.RawArray("d", xy_size)
        ZooRespIP_interp_ctypes = sharedctypes.RawArray("d", xy_size)

        NPPOI_interp_ctypes = sharedctypes.RawArray("d", xy_size)
        NPPOIP_interp_ctypes = sharedctypes.RawArray("d", xy_size)

        OXIDATIONBYDOXI_interp_ctypes = sharedctypes.RawArray("d", xy_size)
        OXIDATIONBYDOXIP_interp_ctypes = sharedctypes.RawArray("d", xy_size)

        # BIO 2D PTRC (1 vars : airseaoxygenflux)
        airseaoxygenflux_interp_ctypes = sharedctypes.RawArray("d", xy_size)
        airseaoxygenfluxP_interp_ctypes = sharedctypes.RawArray("d", xy_size)

        # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
        nppo_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        nppoP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        ZooResp_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        ZooRespP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        doc_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        docP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        
        # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
        bac_oxygenconsumption_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        bac_oxygenconsumptionP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        oxidationbydox_interp_ctypes = sharedctypes.RawArray("d", xyz_size)
        oxidationbydoxP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        chl_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        chlP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        dox_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        doxP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        nos_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        nosP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        poc_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        pocP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        par_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        parP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        nhs_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        nhsP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        odu_interp_ctypes  = sharedctypes.RawArray("d", xyz_size)
        oduP_interp_ctypes = sharedctypes.RawArray("d", xyz_size)

        # end of bio #

        ssh_interp_ctypes[:] = memoryview(
            full_like(ssh_interp_ctypes, fill_value, dtype=float)
        )
        mld_interp_ctypes[:] = memoryview(
            full_like(mld_interp_ctypes, fill_value, dtype=float)
        )
        navlon_interp_ctypes[:] = memoryview(
            full_like(navlon_interp_ctypes, fill_value, dtype=float)
        )
        navlat_interp_ctypes[:] = memoryview(
            full_like(navlat_interp_ctypes, fill_value, dtype=float)
        )        
        topo_interp_ctypes[:] = memoryview(
            full_like(topo_interp_ctypes, fill_value, dtype=float)
        )

        # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
        bac_oxygenconsumptionI_interp_ctypes[:] = memoryview(
            full_like(bac_oxygenconsumptionI_interp_ctypes, fill_value, dtype=float)
        )
        bac_oxygenconsumptionIP_interp_ctypes[:] = memoryview(
            full_like(bac_oxygenconsumptionIP_interp_ctypes, fill_value, dtype=float)
        )

        ZooRespI_interp_ctypes[:] = memoryview(
            full_like(ZooRespI_interp_ctypes, fill_value, dtype=float)
        )
        ZooRespIP_interp_ctypes[:] = memoryview(
            full_like(ZooRespIP_interp_ctypes, fill_value, dtype=float)
        )

        NPPOI_interp_ctypes[:] = memoryview(
            full_like(NPPOI_interp_ctypes, fill_value, dtype=float)
        )
        NPPOIP_interp_ctypes[:] = memoryview(
            full_like(NPPOIP_interp_ctypes, fill_value, dtype=float)
        )

        OXIDATIONBYDOXI_interp_ctypes[:] = memoryview(
            full_like(OXIDATIONBYDOXI_interp_ctypes, fill_value, dtype=float)
        )
        OXIDATIONBYDOXIP_interp_ctypes[:] = memoryview(
            full_like(OXIDATIONBYDOXIP_interp_ctypes, fill_value, dtype=float)
        )

        # BIO 2D PTRC (1 vars : airseaoxygenflux)
        airseaoxygenflux_interp_ctypes[:] = memoryview(
            full_like(airseaoxygenflux_interp_ctypes, fill_value, dtype=float)
        )
        airseaoxygenfluxP_interp_ctypes[:] = memoryview(
            full_like(airseaoxygenfluxP_interp_ctypes, fill_value, dtype=float)
        )

        # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
        nppo_interp_ctypes[:]      = memoryview(full_like(nppo_interp_ctypes, fill_value, dtype=float))
        nppoP_interp_ctypes[:]     = memoryview(full_like(nppoP_interp_ctypes, fill_value, dtype=float))

        ZooResp_interp_ctypes[:]  = memoryview(full_like(ZooResp_interp_ctypes, fill_value, dtype=float))
        ZooRespP_interp_ctypes[:] = memoryview(full_like(ZooRespP_interp_ctypes, fill_value, dtype=float))

        doc_interp_ctypes[:]       = memoryview(full_like(doc_interp_ctypes, fill_value, dtype=float))
        docP_interp_ctypes[:]      = memoryview(full_like(docP_interp_ctypes, fill_value, dtype=float))

        # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
        bac_oxygenconsumption_interp_ctypes[:]  = memoryview(full_like(bac_oxygenconsumption_interp_ctypes , fill_value, dtype=float))
        bac_oxygenconsumptionP_interp_ctypes[:] = memoryview(full_like(bac_oxygenconsumptionP_interp_ctypes, fill_value, dtype=float))
                
        oxidationbydox_interp_ctypes[:]  = memoryview(full_like(oxidationbydox_interp_ctypes , fill_value, dtype=float))
        oxidationbydoxP_interp_ctypes[:] = memoryview(full_like(oxidationbydoxP_interp_ctypes, fill_value, dtype=float))

        chl_interp_ctypes[:]             = memoryview(full_like(chl_interp_ctypes , fill_value, dtype=float))
        chlP_interp_ctypes[:]            = memoryview(full_like(chlP_interp_ctypes, fill_value, dtype=float))
        
        dox_interp_ctypes[:]             = memoryview(full_like(dox_interp_ctypes, fill_value, dtype=float))
        doxP_interp_ctypes[:]            = memoryview(full_like(doxP_interp_ctypes, fill_value, dtype=float))

        nos_interp_ctypes[:]             = memoryview(full_like(nos_interp_ctypes , fill_value, dtype=float))
        nosP_interp_ctypes[:]            = memoryview(full_like(nosP_interp_ctypes, fill_value, dtype=float))

        par_interp_ctypes[:]             = memoryview(full_like(par_interp_ctypes , fill_value, dtype=float))
        parP_interp_ctypes[:]            = memoryview(full_like(parP_interp_ctypes, fill_value, dtype=float))

        poc_interp_ctypes[:]             = memoryview(full_like(poc_interp_ctypes , fill_value, dtype=float))
        pocP_interp_ctypes[:]            = memoryview(full_like(pocP_interp_ctypes, fill_value, dtype=float))

        nhs_interp_ctypes[:]             = memoryview(full_like(nhs_interp_ctypes , fill_value, dtype=float))
        nhsP_interp_ctypes[:]            = memoryview(full_like(nhsP_interp_ctypes, fill_value, dtype=float))
        
        odu_interp_ctypes[:]             = memoryview(full_like(odu_interp_ctypes, fill_value, dtype=float))
        oduP_interp_ctypes[:]            = memoryview(full_like(odu_interp_ctypes, fill_value, dtype=float))
        # end of bio #
        
        mask_interp_ctypes[:] = memoryview(
            full_like(mask_interp_ctypes, fill_value, dtype=int)
        )
        temp_interp_ctypes[:] = memoryview(
            full_like(temp_interp_ctypes, fill_value, dtype=float)
        )
        salt_interp_ctypes[:] = memoryview(
            full_like(salt_interp_ctypes, fill_value, dtype=float)
        )
        vort_interp_ctypes[:] = memoryview(
            full_like(vort_interp_ctypes, fill_value, dtype=float)
        )
        rho_interp_ctypes[:] = memoryview(
            full_like(rho_interp_ctypes, fill_value, dtype=float)
        )
        tempP_interp_ctypes[:] = memoryview(
            full_like(tempP_interp_ctypes, fill_value, dtype=float)
        )
        saltP_interp_ctypes[:] = memoryview(
            full_like(saltP_interp_ctypes, fill_value, dtype=float)
        )
        rhoP_interp_ctypes[:] = memoryview(
            full_like(rhoP_interp_ctypes, fill_value, dtype=float)
        )
        v_interp_ctypes[:] = memoryview(
            full_like(v_interp_ctypes, fill_value, dtype=float)
        )
        u_interp_ctypes[:] = memoryview(
            full_like(u_interp_ctypes, fill_value, dtype=float)
        )
        w_interp_ctypes[:] = memoryview(
            full_like(w_interp_ctypes, fill_value, dtype=float)
        )

        # ----------------------------------------------------------------------------
        # Load in the Black Sea model input data for the current day
        # ----------------------------------------------------------------------------
        BS_date = num2date(model_date).date().isoformat().replace("-", "")
        the_file = BlackSea_file.replace("????????", BS_date)  # model_filenames[tind]
        print("########## %s ---------------------" % the_file)

        with Dataset(the_file) as nc:
            if start:
                start = False
                model_xarr["BS_topo"][:] = topo
                (
                    model_xarr["BS_dx"][:],
                    model_xarr["BS_dy"][:],
                    model_xarr["BS_coriolis"][:],
                ) = get_dx_dy_f(
                    BlackSea_lon[0].astype(float64), BlackSea_lat[:, 0].astype(float64)
                )

            mask = BlackSea_mask3d == 0

            model_xarr["BS_temp_full"][:] = ma.masked_where(
                mask, nc.variables["votemper"][:]
            ).squeeze()

            model_xarr["BS_salt_full"][:] = ma.masked_where(
                mask, nc.variables["vosaline"][:]
            ).squeeze()

            model_xarr["BS_rho_full"][:] = ma.masked_where(
                mask, nc.variables["vorho"][:]
            ).squeeze()

            ssh = nc.variables["sossheig"][:].squeeze()
            model_xarr["BS_ssh_full"][:] = ma.masked_where(mask[0], ssh)

            mld = nc.variables["somld_bs"][:].squeeze()
            model_xarr["BS_mld_full"][:] = ma.masked_where(mask[0], mld)

            navlon = nc.variables["nav_lon"][:].squeeze()
            model_xarr["BS_navlon_full"][:] = ma.masked_where(mask[0], navlon)

            navlat = nc.variables["nav_lat"][:].squeeze()
            model_xarr["BS_navlat_full"][:] = ma.masked_where(mask[0], navlat)
            # model_xarr['BS_pv_full'][:] = zeros_like(nc.variables['uo'][0, z_slc])

        with Dataset(
            the_file.replace("_T_", "_U_").replace("T_files", "U_files")
        ) as nc:
            u = nc.variables["vozocrtx"][:].squeeze()
            model_xarr["BS_u_full"][:] = ma.masked_where(mask, u)

        with Dataset(
            the_file.replace("_T_", "_V_").replace("T_files", "V_files")
        ) as nc:
            v = nc.variables["vomecrty"][:].squeeze()
            model_xarr["BS_v_full"][:] = ma.masked_where(mask, v)

        with Dataset(
            the_file.replace("_T_", "_W_").replace("T_files", "W_files")
        ) as nc:
            w = nc.variables["vovecrtz"][:].squeeze()
            model_xarr["BS_w_full"][:] = ma.masked_where(mask, w)

        with Dataset(
            the_file.replace("_T_", "_ptrc_T_").replace("P_files", "T_files")
        ) as nc:

            nc.set_auto_mask(False)

            # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
            bac_oxygenconsumption = nc.variables["bac_oxygenconsumption"][:].squeeze()
            model_xarr["BS_bac_oxygenconsumption_full"][:] = ma.masked_where(mask, bac_oxygenconsumption)

            oxidationbydox = nc.variables["OXIDATIONBYDOX"][:].squeeze()
            model_xarr["BS_OXIDATIONBYDOX_full"][:] = ma.masked_where(mask, oxidationbydox)
            
            chl = nc.variables["CHL"][:].squeeze()
            model_xarr["BS_CHL_full"][:] = ma.masked_where(mask, chl)

            dox = nc.variables["DOX"][:].squeeze()
            model_xarr["BS_DOX_full"][:] = ma.masked_where(mask, dox)

            nos = nc.variables["NOS"][:].squeeze()
            model_xarr["BS_NOS_full"][:] = ma.masked_where(mask, nos)

            poc = nc.variables["POC"][:].squeeze()
            model_xarr["BS_POC_full"][:] = ma.masked_where(mask, poc)

            par = nc.variables["PAR"][:].squeeze()
            model_xarr["BS_PAR_full"][:] = ma.masked_where(mask, par)

            nhs = nc.variables["NHS"][:].squeeze()
            model_xarr["BS_NHS_full"][:] = ma.masked_where(mask, nhs)

            odu = nc.variables["ODU"][:].squeeze()
            model_xarr["BS_ODU_full"][:] = ma.masked_where(mask, odu)

            # BIO 2D PTRC (1 vars : airseaoxygenflux)
            AirSeaOxygenFlux = nc.variables["AirSeaOxygenFlux"][:].squeeze()
            model_xarr["BS_AirSeaOxygenFlux_full"][:] = ma.masked_where(mask[0], AirSeaOxygenFlux)

        with Dataset(
            the_file.replace("_T_", "_diag_T_").replace("P_files", "T_files")
        ) as nc:  # TODO add the primes (12 July 22)

            nc.set_auto_mask(False)

            # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
            bac_oxygenconsumptionI = nc.variables["bac_oxygenconsumptionI"][:].squeeze()
            model_xarr["BS_bac_oxygenconsumptionI_full"][:] = ma.masked_where(mask[0], bac_oxygenconsumptionI)
            
            ZooRespI = nc.variables["ZooRespI"][:].squeeze()
            model_xarr["BS_ZooRespI_full"][:] = ma.masked_where(mask[0], ZooRespI)

            NPPOI = nc.variables["NPPOI"][:].squeeze()
            model_xarr["BS_NPPOI_full"][:] = ma.masked_where(mask[0], NPPOI)

            OXIDATIONBYDOXI = nc.variables["OXIDATIONBYDOXI"][:].squeeze()
            model_xarr["BS_OXIDATIONBYDOXI_full"][:] = ma.masked_where(mask[0], OXIDATIONBYDOXI)

            # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
            NPPO = nc.variables["NPPO"][:].squeeze()
            model_xarr["BS_NPPO_full"][:] = ma.masked_where(mask, NPPO)

            ZooResp = nc.variables["ZooResp"][:].squeeze()
            model_xarr["BS_ZooResp_full"][:]     = ma.masked_where(mask, ZooResp)

            doc = nc.variables["DOC"][:].squeeze()
            model_xarr["BS_DOC_full"][:]  = ma.masked_where(mask, doc)

        # -------------------------------------------------------------------------
        # Set up list of weights for gap filling of BS input data
        # Note `zsize-1` because bottom layer is all zeros
        k_wghts_list = [
            fillmask_kdtree(
                random_sample(BlackSea_mask3d.shape)[k], BlackSea_mask3d[k]
            )[-1]
            for k in range(zsize - 1)
        ]

        biovar2Ddiaglist=['bac_oxygenconsumptionI', 'ZooRespI', 'NPPOI', 'OXIDATIONBYDOXI']
        biovar2Dptrclist=['AirSeaOxygenFlux']
        biovar3Ddiaglist=['NPPO', 'ZooResp', 'DOC']
        biovar3Dptrclist=['bac_oxygenconsumption', 'OXIDATIONBYDOX', 'CHL', 'DOX', 'NOS', 'POC', 'PAR', 'NHS', 'ODU']

        for k, k_wghts in enumerate(k_wghts_list):  # range(zsize):

            if not k:  # 2d variables
                model_xarr["BS_ssh_full"].values[ssh == 0] = nan
                model_xarr["BS_ssh_full"][:] = fillmask_kdtree(model_xarr["BS_ssh_full"].values, ssh != 0, weights=k_wghts)

                model_xarr["BS_mld_full"].values[mld == 0] = nan
                model_xarr["BS_mld_full"][:] = fillmask_kdtree(model_xarr["BS_mld_full"].values, mld != 0, weights=k_wghts)

                model_xarr["BS_navlon_full"].values[navlon == 0] = nan
                model_xarr["BS_navlon_full"][:] = fillmask_kdtree(model_xarr["BS_navlon_full"].values, navlon != 0, weights=k_wghts)

                model_xarr["BS_navlat_full"].values[navlat == 0] = nan
                model_xarr["BS_navlat_full"][:] = fillmask_kdtree(model_xarr["BS_navlat_full"].values, navlat != 0, weights=k_wghts)

                # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)
                # BIO 2D PTRC (1 vars : airseaoxygenflux)
                for v2d in biovar2Ddiaglist+biovar2Dptrclist:
                    fullstring  = 'BS_'+v2d+'_full'
                    fullstringP = 'BS_'+v2d+'P_full'

                    model_xarr[fullstring ].values                           = fillmask_kdtree(model_xarr[fullstring].values,~BlackSea_mask3d[0].copy(),weights=k_wghts)
                    model_xarr[fullstring ].values[BlackSea_mask3d[0] == 0]  = nan
                    model_xarr[fullstringP]                                  = model_xarr[fullstring].copy()
                    model_xarr[fullstringP]                                 -= gaussian_with_nans(ma.masked_invalid(model_xarr[fullstring].values),sigma_lo,mode="reflect")
                    model_xarr[fullstringP].values[BlackSea_mask3d[0] == 0]  = nan

            model_xarr["BS_u_full"].values[k][u[k] == 0] = nan
            model_xarr["BS_u_full"][k] = fillmask_kdtree(model_xarr["BS_u_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)

            model_xarr["BS_v_full"].values[k][v[k] == 0] = nan
            model_xarr["BS_v_full"][k] = fillmask_kdtree(model_xarr["BS_v_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)

            model_xarr["BS_w_full"].values[k][w[k] == 0] = nan
            model_xarr["BS_w_full"][k] = fillmask_kdtree(model_xarr["BS_w_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)

            model_xarr["BS_vort_full"][k] = vorticity(
                model_xarr["BS_u_full"].values[k],
                model_xarr["BS_v_full"].values[k],
                model_xarr["BS_dx"],
                model_xarr["BS_dy"],
            )
            model_xarr["BS_vort_full"][k] /= model_xarr["BS_coriolis"]

            model_xarr["BS_vort_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            model_xarr["BS_u_full"   ].values[k][BlackSea_mask3d[k] == 0] = nan
            model_xarr["BS_v_full"   ].values[k][BlackSea_mask3d[k] == 0] = nan
            model_xarr["BS_w_full"   ].values[k][BlackSea_mask3d[k] == 0] = nan

            ### PRIMES
            ## 1. Fill the mask
            model_xarr["BS_temp_full"].values[k] = fillmask_kdtree(model_xarr["BS_temp_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts,)
            model_xarr["BS_temp_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            ## 2. Prepare `BS_tempP_full` with temperature data
            model_xarr["BS_tempP_full"][k]       = model_xarr["BS_temp_full"].values[k].copy()
            ## 3. Subtract Gaussian filtered `BS_temp_full` from `BS_temp_full`
            model_xarr["BS_tempP_full"][k]      -= gaussian_with_nans(ma.masked_invalid(model_xarr["BS_temp_full"].values[k]),sigma_lo,mode="reflect")
            model_xarr["BS_tempP_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            ##
            model_xarr["BS_salt_full"].values[k] = fillmask_kdtree(model_xarr["BS_salt_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)
            model_xarr["BS_salt_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            model_xarr["BS_saltP_full"][k]       = model_xarr["BS_salt_full"].values[k].copy()
            model_xarr["BS_saltP_full"][k]      -= gaussian_with_nans(ma.masked_invalid(model_xarr["BS_salt_full"].values[k]),sigma_lo,mode="reflect")
            model_xarr["BS_saltP_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            ##
            # ACcomment : the next line had a missing ".values" if I refer to what was done for other variables.. 
            model_xarr["BS_rho_full"].values[k]   = fillmask_kdtree(model_xarr["BS_rho_full"].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)
            model_xarr["BS_rho_full"].values[k][BlackSea_mask3d[k] == 0] = nan
            model_xarr["BS_rhoP_full"][k]  = model_xarr["BS_rho_full"].values[k].copy()
            model_xarr["BS_rhoP_full"][k] -= gaussian_with_nans(ma.masked_invalid(model_xarr["BS_rho_full"].values[k]),sigma_lo,mode="reflect")
            model_xarr["BS_rhoP_full"].values[k][BlackSea_mask3d[k] == 0] = nan


            for v3d in biovar3Ddiaglist+biovar3Dptrclist:
                fullstring  = 'BS_'+v3d+'_full'
                fullstringP = 'BS_'+v3d+'P_full'

                model_xarr[fullstring ].values[k]                           = fillmask_kdtree(model_xarr[fullstring].values[k],~BlackSea_mask3d[k].copy(),weights=k_wghts)
                model_xarr[fullstring ].values[k][BlackSea_mask3d[k] == 0]  = nan
                model_xarr[fullstringP][k]                                  = model_xarr[fullstring].values[k].copy()
                model_xarr[fullstringP][k]                                 -= gaussian_with_nans(ma.masked_invalid(model_xarr[fullstring].values[k]),sigma_lo,mode="reflect")
                model_xarr[fullstringP].values[k][BlackSea_mask3d[k] == 0]  = nan

        # Prepare for multiprocessor
        # Put variables into shared memory
        # Loop over individual eddies corresponding to current day
        if use_MP:
            print("\nStarting Pool for %s using %s CPUs" %(num2date(model_date).date().isoformat(), ncpu))
            pool = Pool(processes=ncpu)
            pool.map(MP_process_eddies, the_eddy_day_i)
            print("Mapping Done")
            pool.close()
            print("Pool closed")
            pool.join()
            print("Pool joined")
        else:
            print("Starting Serial")
            # for eddy_day_i in the_eddy_day_i:
            #  2834, 2836
            for eddy_day_i in range(112, 115):
                MP_process_eddies(eddy_day_i)

        print("Done (%s, %s)\n--------" % (BS_date, model_date))

        with Dataset(savedir + save_file, "a") as nc:

            # nc.variables['order'][the_eddy_days_i] = eddy_order

            nc.variables["time"][the_eddy_days_i] = time_ctypes[:]
            nc.variables["track"][the_eddy_days_i] = track_ctypes[:]
            nc.variables["n"][the_eddy_days_i] = n_ctypes[:]
            nc.variables["centlon"][the_eddy_days_i] = centlon_ctypes[:]
            nc.variables["centlat"][the_eddy_days_i] = centlat_ctypes[:]
            nc.variables["radius"][the_eddy_days_i] = radius_ctypes[:]
            nc.variables["radius_eff"][the_eddy_days_i] = radius_eff_ctypes[:]
            nc.variables["amplitude"][the_eddy_days_i] = amplitude_ctypes[:]
            nc.variables["cyc"][the_eddy_days_i] = ncyc  # cyc_ctypes[:]
            nc.variables["virtual"][the_eddy_days_i] = virtual_ctypes[:]
            nc.variables["i0i1_j0j1"][the_eddy_days_i] = array(
                i0i1_j0j1_ctypes[:]
            ).reshape(-1, 4)

            nc.variables["contour_lon_s"][the_eddy_days_i] = array(
                contour_lon_s_ctypes[:]
            ).reshape(-1, 50)
            nc.variables["contour_lon_e"][the_eddy_days_i] = array(
                contour_lon_e_ctypes[:]
            ).reshape(-1, 50)
            nc.variables["contour_lat_s"][the_eddy_days_i] = array(
                contour_lat_s_ctypes[:]
            ).reshape(-1, 50)
            nc.variables["contour_lat_e"][the_eddy_days_i] = array(
                contour_lat_e_ctypes[:]
            ).reshape(-1, 50)

            nc.variables["SSH"][the_eddy_days_i] = array(ssh_interp_ctypes[:]).reshape(
                -1, *xgrid.shape
            )
            nc.variables["MLD"][the_eddy_days_i] = array(mld_interp_ctypes[:]).reshape(
                -1, *xgrid.shape
            )

            nc.variables["navlon"][the_eddy_days_i] = array(navlon_interp_ctypes[:]).reshape(
                -1, *xgrid.shape
            )

            nc.variables["navlat"][the_eddy_days_i] = array(navlat_interp_ctypes[:]).reshape(
                -1, *xgrid.shape
            )

            nc.variables["TOPO"][the_eddy_days_i] = array(
                topo_interp_ctypes[:]
            ).reshape(-1, *xgrid.shape)

            # BIO 2D DIAG (4 vars : bac_oxygenconsumptionI, ZooRespI, NPPOI, OXIDATIONBYDOXI)            
            nc.variables["bac_oxygenconsumptionI"  ][the_eddy_days_i] = array(bac_oxygenconsumptionI_interp_ctypes[:] ).reshape(-1, *xgrid.shape)
            nc.variables["bac_oxygenconsumptionI_P"][the_eddy_days_i] = array(bac_oxygenconsumptionIP_interp_ctypes[:]).reshape(-1, *xgrid.shape)

            nc.variables["ZooRespI"  ][the_eddy_days_i] = array(ZooRespI_interp_ctypes[:] ).reshape(-1, *xgrid.shape)
            nc.variables["ZooRespI_P"][the_eddy_days_i] = array(ZooRespIP_interp_ctypes[:]).reshape(-1, *xgrid.shape)

            nc.variables["NPPOI"  ][the_eddy_days_i] = array(NPPOI_interp_ctypes[:] ).reshape(-1, *xgrid.shape)
            nc.variables["NPPOI_P"][the_eddy_days_i] = array(NPPOIP_interp_ctypes[:]).reshape(-1, *xgrid.shape)

            nc.variables["OXIDATIONBYDOXI"  ][the_eddy_days_i] = array(OXIDATIONBYDOXI_interp_ctypes[:] ).reshape(-1, *xgrid.shape)
            nc.variables["OXIDATIONBYDOXI_P"][the_eddy_days_i] = array(OXIDATIONBYDOXIP_interp_ctypes[:]).reshape(-1, *xgrid.shape)

            # BIO 2D PTRC (1 vars : airseaoxygenflux)            
            nc.variables["AirSeaOxygenFlux"  ][the_eddy_days_i] = array(airseaoxygenflux_interp_ctypes[:] ).reshape(-1, *xgrid.shape)
            nc.variables["AirSeaOxygenFlux_P"][the_eddy_days_i] = array(airseaoxygenfluxP_interp_ctypes[:]).reshape(-1, *xgrid.shape)
            
            # BIO 3D DIAG (3 vars : NPPO, ZooResp, DOC)
            nc.variables["NPPO"  ][the_eddy_days_i] = array(nppo_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["NPPO_P"][the_eddy_days_i] = array(nppoP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)
            
            nc.variables["ZooResp"  ][the_eddy_days_i] = array(ZooResp_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["ZooResp_P"][the_eddy_days_i] = array(ZooRespP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["DOC"  ][the_eddy_days_i] = array(doc_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["DOC_P"][the_eddy_days_i] = array(docP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            # BIO 3D PTRC (9 vars :  bac_oxygenconsumption, OXIDATIONBYDOX, CHL, DOX, NOS, POC, PAR, NHS, ODU)
            nc.variables["bac_oxygenconsumption"  ][the_eddy_days_i] = array(bac_oxygenconsumption_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["bac_oxygenconsumption_P"][the_eddy_days_i] = array(bac_oxygenconsumptionP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["OXIDATIONBYDOX"  ][the_eddy_days_i] = array(oxidationbydox_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["OXIDATIONBYDOX_P"][the_eddy_days_i] = array(oxidationbydoxP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["CHL"  ][the_eddy_days_i] = array(chl_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["CHL_P"][the_eddy_days_i] = array(chlP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["DOX"  ][the_eddy_days_i] = array(dox_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["DOX_P"][the_eddy_days_i] = array(doxP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["NOS"  ][the_eddy_days_i] = array(nos_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["NOS_P"][the_eddy_days_i] = array(nosP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["POC"  ][the_eddy_days_i] = array(poc_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["POC_P"][the_eddy_days_i] = array(pocP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["PAR"  ][the_eddy_days_i] = array(par_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["PAR_P"][the_eddy_days_i] = array(parP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["NHS"  ][the_eddy_days_i] = array(nhs_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["NHS_P"][the_eddy_days_i] = array(nhsP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            nc.variables["ODU"  ][the_eddy_days_i] = array(odu_interp_ctypes[:] ).reshape(-1, zsize, *xgrid.shape)
            nc.variables["ODU_P"][the_eddy_days_i] = array(oduP_interp_ctypes[:]).reshape(-1, zsize, *xgrid.shape)

            ## End of bio ##
            
            nc.variables["MASK"][the_eddy_days_i] = array(
                mask_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["TEMP"][the_eddy_days_i] = array(
                temp_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["SALT"][the_eddy_days_i] = array(
                salt_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["RHO"][the_eddy_days_i] = array(rho_interp_ctypes[:]).reshape(
                -1, zsize, *xgrid.shape
            )

            nc.variables["VORT"][the_eddy_days_i] = array(
                vort_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["TEMP_P"][the_eddy_days_i] = array(
                tempP_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["SALT_P"][the_eddy_days_i] = array(
                saltP_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["RHO_P"][the_eddy_days_i] = array(
                rhoP_interp_ctypes[:]
            ).reshape(-1, zsize, *xgrid.shape)

            nc.variables["U"][the_eddy_days_i] = array(u_interp_ctypes[:]).reshape(
                -1, zsize, *xgrid.shape
            )

            nc.variables["V"][the_eddy_days_i] = array(v_interp_ctypes[:]).reshape(
                -1, zsize, *xgrid.shape
            )

            nc.variables["W"][the_eddy_days_i] = array(w_interp_ctypes[:]).reshape(
                -1, zsize, *xgrid.shape
            )

