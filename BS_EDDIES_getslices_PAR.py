# %run bs_eddies_getslices_PAR.py

# from datetime import date, datetime
import matplotlib.dates as mdates
# See https://matplotlib.org/3.3.0/api/dates_api.html#matplotlib.dates
#old_epoch = '0000-12-31T00:00:00'
#mdates.set_epoch(old_epoch)

import argparse

from copy import copy
from glob import glob
from multiprocessing import Pool

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib import cm
from matplotlib.dates import date2num, datetime, num2date
from netCDF4 import Dataset
from pandas import to_datetime
from numpy import (array, atleast_3d, ceil, float32, float_, floor, hstack,
                   int32, isnan, linspace, logical_not, ma, meshgrid, nanmax,
                   nanmin, newaxis, poly1d, polyfit, rint, zeros, zeros_like)
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import map_coordinates
from xgcm import Grid

from emTools import fillmask_kdtree, getncvcm
from gradient04 import gradient
from BS_EDDIES_getcubes import get_pm_pn


parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year" , type=str, help="start year")
parser.add_argument("-m", "--month", type=str, help="start month (1st)")
parser.add_argument("-n", "--ncpu" , type=int, help="ncpus")

args = parser.parse_args()
runyear  = args.year
runmonth = args.month
ncpus_arg = args.ncpu

EARTH_R = 6370997.0  # Same as py-eddy-tracker `grid.py`

fill_value_32 = float32(1e18)


def compute_scale_and_offset(dmin, dmax, n):
    # Stretch/compress data to the available packed range
    scale_factor = (dmax - dmin) / (2 ** n - 1)
    # Translate the range to be symmetric about zero
    add_offset = dmin + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)

def pack_value(unpacked_value, scale_factor, add_offset):
    return floor((unpacked_value - add_offset) / scale_factor)

def unpack_value(packed_value, scale_factor, add_offset):
    return packed_value * scale_factor + add_offset

def interp_to_2D(datain, x, y, x_ext, y_ext):
    """"""
    return RectBivariateSpline(y, x, datain).ev(y_ext, x_ext)

def transform(grid, var_on_rho, var_on_rho_str, target_rho_levels):
    """"""
    # print(var_on_rho)
    return grid.transform(
        var_on_rho[var_on_rho_str],
        "Z",
        target_rho_levels,
        target_data=var_on_rho["rho"],
        method="linear",
    )

def extended(ax, pt, x, y, **args):
    # https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array
    # https://stackoverflow.com/questions/19310735/extending-a-line-segment-in-matplotlib

    xlim = (-pt, pt)  # ax.get_xlim()
    ylim = (-pt, pt)  # ax.get_ylim()

    x_ext = linspace(xlim[0], xlim[1], 1001)
    p = polyfit(x, y, deg=1)
    y_ext = poly1d(p)(x_ext)
    y_ext = ma.masked_outside(y_ext, -pt, pt)
    x_ext = x_ext[y_ext.mask == False]
    y_ext = y_ext[y_ext.mask == False].data
    ax.plot(x_ext, y_ext, **args)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax, x_ext, y_ext


# https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

#def processaday():


if __name__ == "__main__":

    plt.close("all")
    make_figure = False  # True
    ACYC = True  # False for CCYC
    pt = 3.5  # pt is point on normalised grid
    z_thresh = 210.0  #225.0

    print(runyear)
    start_date, end_date = str(runyear)+str(runmonth), str(runyear)+'12'

    print(start_date)
    print(end_date)
    directory = "/scratch/ulg/mast/acapet/Compout/cubes/"

    BS_compo_files = (
        "BlackSea_ACYC_composites_"+str(runyear)+"??.nc"
        if ACYC
        else "BlackSea_CCYC_composites_"+str(runyear)+"??.nc"
    )

    BS_compo_files = sorted(glob(directory + BS_compo_files))

    print(BS_compo_files)
    
    biovar2Ddiaglist=['bac_oxygenconsumptionI', 'ZooRespI', 'NPPOI', 'OXIDATIONBYDOXI']
    biovar2Dptrclist=['AirSeaOxygenFlux']
    biovar3Ddiaglist=['NPPO', 'ZooResp', 'DOC']
    biovar3Dptrclist=['bac_oxygenconsumption', 'OXIDATIONBYDOX', 'CHL', 'DOX', 'NOS', 'POC', 'PAR', 'NHS', 'ODU']

    #TODO : RHO range ?? 
    target_rho_levels = linspace(
        9, 19, 101
        #floor(ds_daily.rho.min()), ceil(ds_daily.rho.max()), 101
    )
    savedir = directory
    
    # ---------------------------------------------------------------------------
    # ---- END user options
    # ---------------------------------------------------------------------------

    index_count = 0

    start_i, stop_i = [i for i, f in enumerate(BS_compo_files) if start_date in f or end_date in f]
    BS_compo_files = BS_compo_files[start_i:stop_i + 1]
    
    savefile_z = "Z_slices/BS_Z_slices_%s_track-%s_%s.nc"  # CYC TRACK YYYYMMDD
    savefile_rho = "RHO_slices/BS_RHO_slices_%s_track-%s_%s.nc"  # CYC TRACK YYYYMMDD
    region_mask = xr.open_dataset('/home/ulg/mast/acapet/Evan/region_O2_smooth.nc')
    depth_mask = xr.open_dataset('/scratch/ulg/mast/emason/daily_MYP_input_files/bathy_meter.nc')

    fig = plt.figure(1)

    for i, BS_file in enumerate(BS_compo_files):

        print("####", BS_file)
        ds = xr.open_dataset(BS_file)

        if not i:
            x, y = meshgrid(ds.x.values, ds.y.values)
            cx, cy = 0, 0

            mask = zeros(ds.MASK.shape[1:]).astype("int")

            vort  = ma.zeros((mask.shape))
            rho   = ma.zeros((mask.shape))
            rhoP  = ma.zeros((mask.shape))
            temp  = ma.zeros((mask.shape))
            tempP = ma.zeros((mask.shape))
            salt  = ma.zeros((mask.shape))
            saltP = ma.zeros((mask.shape))
            w     = ma.zeros((mask.shape))
            u     = ma.zeros((mask.shape))
            v     = ma.zeros((mask.shape))
            
            # Here we prepare the variables for storing 3D cubes info.
            biodic={}
            for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                biodic[vvv]={'varname':vvv,
                             'cubearray'   : ma.zeros((mask.shape)),
                             'cubearrayP'  : ma.zeros((mask.shape))
                             }

            # Here we prepare the variables for storing 2D cubes info.
            for vvv in biovar2Ddiaglist+biovar2Dptrclist:
                biodic[vvv]={'varname':vvv,
                             'cubearray'   : ma.zeros((mask.shape[1:])),
                             'cubearrayP'  : ma.zeros((mask.shape[1:]))
                             }

        for tind, the_day in enumerate(ds.time): # This loop is on time AND tracks. ds.time isn't monotonous; it contains times for each obs of each track. 

            yyyymmdd_str = num2date(the_day).strftime("%Y%m%d")
            print("------", tind, yyyymmdd_str)

            mld = ds["MLD"].values[tind]
            lon = ds["navlon"].values[tind]
            lat = ds["navlat"].values[tind]                        
            ssh = ds["SSH"].values[tind]
            topo = ds["TOPO"].values[tind]

            # bio loop 2D, filling cube arrays with data from nc
            for vvv in biovar2Ddiaglist + biovar2Dptrclist:
                biodic[vvv]['cubearray']  = ds[vvv].values[tind] 
                biodic[vvv]['cubearrayP'] = ds[vvv+'_P'].values[tind] 

            # Exclude too shallow regions
            if nanmax(topo) < z_thresh:
                print("I'm out")
                continue
                
            centlon = ds["centlon"][tind]
            centlat = ds["centlat"][tind]
            
            ri, rj = (abs(depth_mask['lon'] - centlon).argmin(),
                      abs(depth_mask['lat'] - centlat).argmin())
            region = region_mask['region'][rj, ri]
            
            radius     = ds["radius"][tind]
            radius_eff = ds["radius_eff"][tind]
            amplitude  = ds["amplitude"][tind]
            track      = ds["track"][tind]
            n          = ds["n"][tind]
            time       = ds["time"][tind]

            #if True:
            #NOTE I made 4 subplots for a figure for the paper, but really not necesarry for this script ...
            #fig, ax = plt.subplots(2, 2, figsize=(9, 8))
            #fig, ax = plt.subplots(1, figsize=(9, 8))
            #fig.suptitle(yyyymmdd_str)

            ax = fig.add_subplot(111)

            topo200 = ax.contour( x, y, topo, [200] )
            tnearest = topo200.find_nearest_contour(cx,cy, indices=None, pixel=False)

            print('tnearest : ') 
            print(tnearest)
            
#            x_short = linspace(0, tnearest[3])
#            y_short = linspace(0, tnearest[4])

#            _, x_ext, y_ext = extended(
#                ax,
#                pt,
#                x_short,
#                y_short,
#                color="lightsteelblue",
#                lw=2,
#                label="extended",
#            )
            # x_ext = linspace(x_ext.squeeze()[0], x_ext.squeeze()[-1], ds.x.size)
            # y_ext = linspace(y_ext.squeeze()[0], y_ext.squeeze()[-1], ds.x.size)

            # Get the coordinates of two points of intersetion between 
            #   the line including the segment between center and nearest coastline, 
            #   and a centered circle of radius pt
            intpoint1, intpoint2 = circle_line_segment_intersection((0,0), pt, (0,0), (tnearest[3],tnearest[4]))
    
            x_ext = linspace(intpoint1[0], intpoint2[0], ds.x.size)
            y_ext = linspace(intpoint1[1], intpoint2[1], ds.x.size)

            ax.clear()
            plt.clf()

            # Z LOOP <BEG>
            for zi, z in enumerate(ds.z):

                mask[zi] = ds["MASK"][tind, zi]

                if (mask[zi] == 0).all():  # no data
                    continue

                msk = float_(mask[zi]).copy()

                if not msk.all():
                    if msk.sum() <= 10:  # Fix if only a few data points at level `z` Not sure if 10 is too small
                        continue 
                    
                    vort[zi]  = ds["VORT"][tind, zi]
                    rho[zi]   = ds["RHO"][tind, zi]
                    rhoP[zi]  = ds["RHO_P"][tind, zi]
                    temp[zi]  = ds["TEMP"][tind, zi]
                    tempP[zi] = ds["TEMP_P"][tind, zi]
                    salt[zi]  = ds["SALT"][tind, zi]
                    saltP[zi] = ds["SALT_P"][tind, zi]
                    w[zi]     = ds["W"][tind, zi]
                    u[zi]     = ds["U"][tind, zi]
                    v[zi]     = ds["V"][tind, zi]

                    # bio loop 3D Storing data from the cubes netcdf
                    for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                        biodic[vvv]['cubearray'][zi]  = ds[vvv][tind, zi] 
                        biodic[vvv]['cubearrayP'][zi] = ds[vvv+'_P'][tind, zi] 

                    vort[zi], wgt = fillmask_kdtree(vort[zi] , msk, k=4, weights=None, fill_value=fill_value_32)
                    rho[zi]       = fillmask_kdtree(rho[zi]  , msk, weights=wgt)
                    rhoP[zi]      = fillmask_kdtree(rhoP[zi] , msk, weights=wgt)
                    temp[zi]      = fillmask_kdtree(temp[zi] , msk, weights=wgt)
                    tempP[zi]     = fillmask_kdtree(tempP[zi], msk, weights=wgt)
                    salt[zi]      = fillmask_kdtree(salt[zi] , msk, weights=wgt)
                    saltP[zi]     = fillmask_kdtree(saltP[zi], msk, weights=wgt)
                    w[zi]         = fillmask_kdtree(w[zi]    , msk, weights=wgt)
                    u[zi]         = fillmask_kdtree(u[zi]    , msk, weights=wgt)
                    v[zi]         = fillmask_kdtree(v[zi]    , msk, weights=wgt)

                    # bio loop 3D : Filling in the cube arrays
                    for vvv in biovar3Ddiaglist + biovar3Dptrclist:                        
                        biodic[vvv]['cubearray'][zi]   = fillmask_kdtree(biodic[vvv]['cubearray'][zi]  , msk, weights=wgt) 
                        biodic[vvv]['cubearrayP'][zi]  = fillmask_kdtree(biodic[vvv]['cubearrayP'][zi]  , msk, weights=wgt) 

                if not zi:
                    mask2d  = zeros((x_ext.size, ds.z.size))
                    vort2d  = ma.zeros((x_ext.size, ds.z.size))
                    rho2d   = ma.zeros((x_ext.size, ds.z.size))
                    rho2dP  = ma.zeros((x_ext.size, ds.z.size))
                    temp2d  = ma.zeros((x_ext.size, ds.z.size))
                    temp2dP = ma.zeros((x_ext.size, ds.z.size))
                    salt2d  = ma.zeros((x_ext.size, ds.z.size))
                    salt2dP = ma.zeros((x_ext.size, ds.z.size))
                    w2d     = ma.zeros((x_ext.size, ds.z.size))
                    u2d     = ma.zeros((x_ext.size, ds.z.size))
                    v2d     = ma.zeros((x_ext.size, ds.z.size))

                    #bio loop 3D : initiate zero for slice arrays
                    for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                        biodic[vvv].update({'slicearray' : ma.zeros((x_ext.size, ds.z.size)), 
                                            'slicearrayP': ma.zeros((x_ext.size, ds.z.size))})        

                    if not msk.all():
                        mld[:]  = fillmask_kdtree(mld, msk, weights=wgt)
                        lon[:]  = fillmask_kdtree(lon, msk, weights=wgt)
                        lat[:]  = fillmask_kdtree(lat, msk, weights=wgt)
                        topo[:] = fillmask_kdtree(topo, msk, weights=wgt)
                        ssh[:]  = fillmask_kdtree(ssh, msk, weights=wgt)
                        # bio loop 2D, filling masked points
                        for vvv in biovar2Ddiaglist + biovar2Dptrclist:
                            biodic[vvv]['cubearray'][:]  = fillmask_kdtree(biodic[vvv]['cubearray' ], msk, weights=wgt)
                            biodic[vvv]['cubearrayP'][:] = fillmask_kdtree(biodic[vvv]['cubearrayP'], msk, weights=wgt)

                    mld1d   = interp_to_2D(mld, x[0], y[:, 0], x_ext, y_ext)
                    lon1d   = interp_to_2D(lon, x[0], y[:, 0], x_ext, y_ext)
                    lat1d   = interp_to_2D(lat, x[0], y[:, 0], x_ext, y_ext)
                    
                    topo1d  = interp_to_2D(topo, x[0], y[:, 0], x_ext, y_ext)
                    ssh1d   = interp_to_2D(ssh, x[0], y[:, 0], x_ext, y_ext) 
                    # bio loop 2D, interp on slice
                    for vvv in biovar2Ddiaglist + biovar2Dptrclist:
                        biodic[vvv].update({'slicearray'  : interp_to_2D(biodic[vvv]['cubearray' ], x[0], y[:, 0], x_ext, y_ext),
                                            'slicearrayP' : interp_to_2D(biodic[vvv]['cubearrayP'], x[0], y[:, 0], x_ext, y_ext)})

                mask2d[:, zi]  = interp_to_2D(mask[zi] , x[0], y[:, 0], x_ext, y_ext)
                vort2d[:, zi]  = interp_to_2D(vort[zi] , x[0], y[:, 0], x_ext, y_ext)
                rho2d[:, zi]   = interp_to_2D(rho[zi]  , x[0], y[:, 0], x_ext, y_ext)
                rho2dP[:, zi]  = interp_to_2D(rhoP[zi] , x[0], y[:, 0], x_ext, y_ext)
                temp2d[:, zi]  = interp_to_2D(temp[zi] , x[0], y[:, 0], x_ext, y_ext)
                temp2dP[:, zi] = interp_to_2D(tempP[zi], x[0], y[:, 0], x_ext, y_ext)
                salt2d[:, zi]  = interp_to_2D(salt[zi] , x[0], y[:, 0], x_ext, y_ext)
                salt2dP[:, zi] = interp_to_2D(saltP[zi], x[0], y[:, 0], x_ext, y_ext)
                w2d[:, zi]     = interp_to_2D(w[zi]    , x[0], y[:, 0], x_ext, y_ext)
                u2d[:, zi]     = interp_to_2D(u[zi]    , x[0], y[:, 0], x_ext, y_ext)
                v2d[:, zi]     = interp_to_2D(v[zi]    , x[0], y[:, 0], x_ext, y_ext)

                # bio loop 3D : Filling slice arrays with data interpollated from cube arrays
                for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                    biodic[vvv]['slicearray'][:, zi]  =  interp_to_2D(biodic[vvv]['cubearray' ][zi], x[0], y[:, 0], x_ext, y_ext)
                    biodic[vvv]['slicearrayP'][:, zi] =  interp_to_2D(biodic[vvv]['cubearrayP'][zi], x[0], y[:, 0], x_ext, y_ext)
                
                # END OF Z LOOP

            mask2d[:] = abs(rint(mask2d))

            vort2d[:]  = ma.masked_where(mask2d == 0, ma.masked_array(vort2d , fill_value=fill_value_32))
            rho2d[:]   = ma.masked_where(mask2d == 0, ma.masked_array(rho2d  , fill_value=fill_value_32))
            rho2dP[:]  = ma.masked_where(mask2d == 0, ma.masked_array(rho2dP , fill_value=fill_value_32))
            temp2d[:]  = ma.masked_where(mask2d == 0, ma.masked_array(temp2d , fill_value=fill_value_32))
            temp2dP[:] = ma.masked_where(mask2d == 0, ma.masked_array(temp2dP, fill_value=fill_value_32))
            salt2d[:]  = ma.masked_where(mask2d == 0, ma.masked_array(salt2d , fill_value=fill_value_32))
            salt2dP[:] = ma.masked_where(mask2d == 0, ma.masked_array(salt2dP, fill_value=fill_value_32))
            w2d[:]     = ma.masked_where(mask2d == 0, ma.masked_array(w2d    , fill_value=fill_value_32))
            u2d[:]     = ma.masked_where(mask2d == 0, ma.masked_array(u2d    , fill_value=fill_value_32))
            v2d[:]     = ma.masked_where(mask2d == 0, ma.masked_array(v2d    , fill_value=fill_value_32))

            # bio loop 3D : Masking slice arrays
            for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                biodic[vvv]['slicearray'][:]  =  ma.masked_where(mask2d == 0, ma.masked_array(biodic[vvv]['slicearray'],  fill_value=fill_value_32))
                biodic[vvv]['slicearrayP'][:] =  ma.masked_where(mask2d == 0, ma.masked_array(biodic[vvv]['slicearrayP'], fill_value=fill_value_32))           
            
            if not tind:
                cyc = "ACYC" if ACYC else "CCYC"
            # Debug
            '''cyc = "ACYC" if ACYC else "CCYC"
            print('Remove once debugged')'''

            norm_eddy_radius, depth = (linspace(-pt, pt, x_ext.size), ds.z)

            dax = ["index", "norm_eddy_radius", "depth"]
            # print('---- set Dataset %s' % yyyymmdd)

            data_vars_dic = {
                    "vort"       : (dax, vort2d[newaxis]),
                    "rho"        : (dax, rho2d[newaxis]),
                    "rhoP"       : (dax, rho2dP[newaxis]),
                    "temp"       : (dax, temp2d[newaxis]),
                    "tempP"      : (dax, temp2dP[newaxis]),
                    "salt"       : (dax, salt2d[newaxis]),
                    "saltP"      : (dax, salt2dP[newaxis]),
                    "w"          : (dax, w2d[newaxis]),
                    "u"          : (dax, u2d[newaxis]),
                    "v"          : (dax, v2d[newaxis]),

                    "mld"        : (("index", "norm_eddy_radius"), [mld1d]),
                    "lon"        : (("index", "norm_eddy_radius"), [lon1d]),
                    "lat"        : (("index", "norm_eddy_radius"), [lat1d]),
                    
                    "topo"       : (("index", "norm_eddy_radius"), [topo1d]),
                    "ssh"        : (("index", "norm_eddy_radius"), [ssh1d]),
                    "time"       : ('index', [time.data]),
                    "centlon"    : ('index', [centlon.data]),
                    "centlat"    : ('index', [centlat.data]),
                    "radius"     : ('index', [radius.data]),
                    "radius_eff" : ('index', [radius_eff.data]),
                    "amplitude"  : ('index', [amplitude.data]),
                    "track"      : ('index', [track.data]),
                    "n"          : ('index', [n.data]),
                    #index = (('index'), ['index']),
                    "region"     : ('index', [region.data])
            }

            # bio loop 3D, adding variables in the xarray built for netcdf output.
            for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                data_vars_dic.update({ vvv      : (dax, biodic[vvv]['slicearray' ][newaxis]),
                                       vvv+'_P' : (dax, biodic[vvv]['slicearrayP'][newaxis])})

            for vvv in biovar2Ddiaglist + biovar2Dptrclist:
                data_vars_dic.update({ vvv      : (("index", "norm_eddy_radius"), [biodic[vvv]['slicearray' ]]),
                                       vvv+'_P' : (("index", "norm_eddy_radius"), [biodic[vvv]['slicearrayP']])})


            ds_daily = xr.Dataset(
                data_vars=data_vars_dic,
               coords=dict(
                    index=array([index_count]),
                    #time=(["time"], yyyymmdd),
                    #time=array([yyyymmdd]),
                    #time = (('index'), time),
                    depth=(["depth"], depth.data),
                    norm_eddy_radius=(["norm_eddy_radius"], norm_eddy_radius),
                    xcoord=(["norm_eddy_radius"], x_ext.astype(float32)),
                    ycoord=(["norm_eddy_radius"], y_ext.astype(float32)),
                )
            )

            ## SAVING TO NETCDF
            ds_daily.transpose("index", "depth", "norm_eddy_radius").to_netcdf(
                savedir + savefile_z % (cyc, str(track.values).zfill(4), yyyymmdd_str),
                unlimited_dims="index",
            )
            
            grid = Grid(
                ds_daily,
                coords={"Z": {"center": "depth"}, "Y": {"center": "norm_eddy_radius"}},
                periodic=False,
            )

            ds_daily["rho"] -= 1000

            ### Linear Interpolation with xgcm
            print("--- Transform to rho coords")
            vort_on_rho = transform(grid, ds_daily, "vort", target_rho_levels)
            rhoP_on_rho = transform(grid, ds_daily, "rhoP", target_rho_levels)
            temp_on_rho = transform(grid, ds_daily, "temp", target_rho_levels)
            tempP_on_rho = transform(grid, ds_daily, "tempP", target_rho_levels)
            salt_on_rho = transform(grid, ds_daily, "salt", target_rho_levels)
            saltP_on_rho = transform(grid, ds_daily, "saltP", target_rho_levels)
            w_on_rho = transform(grid, ds_daily, "w", target_rho_levels)
            u_on_rho = transform(grid, ds_daily, "u", target_rho_levels)
            v_on_rho = transform(grid, ds_daily, "v", target_rho_levels)

            # bio loop 3D -> rho levels
            for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                biodic[vvv].update({'onrho'  : transform(grid, ds_daily, vvv , target_rho_levels),
                                    'onrhoP' : transform(grid, ds_daily, vvv+'_P' , target_rho_levels)})

            listof_onrhos = [vort_on_rho,
                rhoP_on_rho,
                temp_on_rho,
                tempP_on_rho,
                salt_on_rho,
                saltP_on_rho,
                w_on_rho,
                u_on_rho,
                v_on_rho]

            for vvv in biovar3Ddiaglist + biovar3Dptrclist:
                listof_onrhos.append(biodic[vvv]['onrho'])
                listof_onrhos.append(biodic[vvv]['onrhoP'])

            for vvv in ['time','centlon','centlat','radius','radius_eff','amplitude','track','n','region']:
                listof_onrhos.append(ds_daily[vvv])

            # Merge the transforms
            ds_daily_on_rho = xr.merge(listof_onrhos)

            
            # ds_daily_on_rho['time']       = (('index'), ds_daily['time'])
            # ds_daily_on_rho['centlon']    = (('index'), ds_daily['centlon'])
            # ds_daily_on_rho['centlat']    = (('index'), ds_daily['centlat'])
            # ds_daily_on_rho['radius']     = (('index'), ds_daily['radius'])
            # ds_daily_on_rho['radius_eff'] = (('index'), ds_daily['radius_eff'])
            # ds_daily_on_rho['amplitude']  = (('index'), ds_daily['amplitude'])
            # ds_daily_on_rho['track']      = (('index'), ds_daily['track'])
            # ds_daily_on_rho['n']          = (('index'), ds_daily['n'])
            # ds_daily_on_rho['index']      = (('index'), ds_daily['index'])
            # ds_daily_on_rho['region']     = (('index'), ds_daily['region'])
            #ds_daily_on_rho["index"] = [the_day]

            ds_daily_on_rho.transpose("index", "rho", "norm_eddy_radius").to_netcdf(
                savedir + savefile_rho % (cyc, str(track.values).zfill(4), yyyymmdd_str),
                unlimited_dims="index",
            )
            
            ds_daily_on_rho.close()
            ds_daily.close()
            ds.close()
            
            index_count += 1




'''
            if make_figure:
                DOX_plot = dox_on_rho.plot(
                    ax=ax[0, 1], x="norm_eddy_radius", yincrease=False, #cmap=cmap_banded
                )
                NPP_plot = npp_on_rho.plot(
                    ax=ax[1, 0], x="norm_eddy_radius", yincrease=False, #cmap=cmap_banded
                )
                VORT_plot = vort_on_rho.plot(
                    ax=ax[1, 1], x="norm_eddy_radius", yincrease=False, #cmap=cmap_banded
                )

                DOX_plot.set_clim(0, 400)
                NPP_plot.set_clim(nanmin(npp_on_rho), nanmax(npp_on_rho))
                VORT_plot.set_clim(-1, 1)

                fig.savefig("rho_levels.png", dpi=300, bbox_inches="tight")

                plt.show()           
'''
