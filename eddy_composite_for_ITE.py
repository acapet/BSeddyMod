# %run eddy_composite_for_ITE.py

from glob import glob
import time
import matplotlib.dates as dt
import matplotlib.pyplot as plt
from pyproj import Proj
#import numexpr as ne
from numpy import array, empty, empty_like, zeros_like, meshgrid, \
                  sin, cos, deg2rad, pi, arctan2, hypot, recarray, ma, \
                  float64, abs, unravel_index, r_, sqrt, int, \
                  asarray, where, nonzero, floor, diff, float32, int16, \
                  int32, hstack, arange, logical_or, int64, ctypeslib, \
                  concatenate, logical_not, unique, int8, fromiter, \
                  linspace, maximum, logical_xor, float_
import scipy.interpolate as scint
import scipy.io as io
#import scipy.spatial as spa
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree, Delaunay, distance
#import progressbar as prog_bar
from dateutil import parser
#from scipy.constants import convert_temperature as K2C
import os.path
import re
import xarray as xr
from eddy_composite import Composite, EddyTrack
from emTools import getncvcm, fillmask_kdtree, nearest, newPosition, fillmask_fast_kdtree
from multiprocessing import Pool, sharedctypes, cpu_count
#from eddy_composite_for_cmems import get_eddy_recarray, get_restart_index_and_clip
#import gradient04 as grad4
from gradient04 import gradient
#from itertools import chain


EARTH_R = 6371315.
FILL_VALUE = -32767.0


def str_compare(str1, str2):
    return str1 in str2 and str2 in str1


def compute_scale_and_offset(min, max, n):
    # stretch/compress data to the available packed range
    scale_factor = (max - min) / (2 ** n - 1)
    # translate the range to be symmetric about zero
    add_offset = min + 2 ** (n - 1) * scale_factor
    return (scale_factor, add_offset)


def pack_value(unpacked_value, scale_factor, add_offset):
    return floor((unpacked_value - add_offset) / scale_factor)

def unpack_value(packed_value, scale_factor, add_offset):
    return packed_value * scale_factor + add_offset


def mp_mkcompo(v_or_k):
    fill_value = FILL_VALUE
    compo.make_composite_by_vars(
        mdsbmodel, v_or_k, fill_value)





def create_nc(savefile, x_arr, time_domain,
              domain, medsub_variables,
              fill_value, the_model, nc_scaling,
              add_model_topo=False,
              add_SRTM_topo=False):
    """
    """
    #fill_value *= -1
    mdl_type = the_model.model
    with Dataset(savefile, 'w', clobber=True, format='NETCDF4') as nc:

        nc.created = dt.datetime.datetime.utcnow().isoformat()
        nc.lonmin = float64(domain[0])
        nc.lonmax = float64(domain[1])
        nc.latmin = float64(domain[2])
        nc.latmax = float64(domain[3])
        y, m, d = time_domain[0]
        nc.start_date = dt.date2num(dt.datetime.datetime(y, m, d))
        y, m, d = time_domain[1]
        nc.end_date = dt.date2num(dt.datetime.datetime(y, m, d))

        N, M, L = x_arr.mask3d_out.squeeze().shape

        nc.createDimension('x', L)
        nc.createDimension('y', M)
        nc.createDimension('z', N)
        nc.createDimension('time', None)
        nc.createDimension('one', 1)

        #nc.createVariable('eddy_i', int64, ('one'))
        nc.createVariable('eddy_date', int64, ('one'))

        nc.createVariable('time', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('eddy_id', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('centlon', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('centlat', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('radius', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('radius_eff', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('amplitude', float32, ('time'),
                          fill_value=fill_value, zlib=True)
        #nc.createVariable('eddy_bearing', float32, ('time'),
                          #fill_value=fill_value, zlib=True)
        nc.createVariable('cyc', int16, ('time'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('x', float32, ('x'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('y', float32, ('y'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('z', float32, ('z'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('the_tlt_tki', int16, ('time', 'z'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('the_tlt_tkj', int16, ('time', 'z'),
                          fill_value=fill_value, zlib=True)
        nc.createVariable('x_ITE_core_zrange', float32, ('time'),
                            fill_value=fill_value, zlib=True)
        nc.createVariable('y_ITE_core_zrange', float32, ('time'),
                            fill_value=fill_value, zlib=True)
        nc.createVariable('i_min_n2', int16, ('time'),
                            fill_value=fill_value, zlib=True)
        nc.createVariable('j_min_n2', int16, ('time'),
                            fill_value=fill_value, zlib=True)
        nc.createVariable('k_min_n2', int16, ('time'),
                            fill_value=fill_value, zlib=True)

        if add_model_topo:
            nc.createVariable('topo_model', float32, ('time', 'y', 'x'),
                              fill_value=fill_value, zlib=True)

        if add_SRTM_topo:
            nc.createVariable('topo_SRTM', float32, ('time', 'y', 'x'),
                              fill_value=fill_value, zlib=True)

        for medsub_variable in medsub_variables:
            if 'ssh' in medsub_variable or 'mld' in medsub_variable:
                nc.createVariable(medsub_variable, float32,
                                  ('time', 'y', 'x'),
                                  fill_value=fill_value, zlib=True)
            elif 'mask3d' in medsub_variable:
                nc.createVariable(medsub_variable, int8,
                                  ('time', 'z', 'y', 'x'),
                                  fill_value=fill_value, zlib=True)
            else:
                nc.createVariable(medsub_variable, float32,
                                  ('time', 'z', 'y', 'x'),
                                  fill_value=fill_value, zlib=True)
        #nc.createVariable('temp', int16, ('time', 'z', 'y', 'x'),
                          #fill_value=fill_value, zlib=True)
        #nc.createVariable('salt', int16, ('time', 'z', 'y', 'x'),
                          #fill_value=fill_value, zlib=True)
        #nc.createVariable('u', int32, ('time', 'z', 'y', 'x'),
                          #fill_value=fill_value, zlib=True)
        #nc.createVariable('v', int32, ('time', 'z', 'y', 'x'),
                          #fill_value=fill_value, zlib=True)
        #nc.createVariable('vort', int32, ('time', 'z', 'y', 'x'),
                          #fill_value=fill_value, zlib=True)
        #nc.createVariable('model_eddy_bearing', float32, ('time', 'z'),
                          #fill_value=fill_value, zlib=True)

        # Set variable attributes
        nc.variables['x'].description = 'Normalised distance along x axis'
        nc.variables['x'].units = "m"
        #----------------------------------------------------------------------
        nc.variables['y'].description = 'Normalised distance along y axis'
        nc.variables['y'].units = "m"
        #----------------------------------------------------------------------
        nc.variables['z'].description = 'Depths'
        nc.variables['z'].units = "m"
        #----------------------------------------------------------------------
        nc.variables['the_tlt_tki'].description = 'Indices for zonal eddy tilt correction'
        nc.variables['the_tlt_tki'].units = "index"
        #----------------------------------------------------------------------
        nc.variables['the_tlt_tkj'].description = 'Indices for meridional eddy tilt correction'
        nc.variables['the_tlt_tkj'].units = "index"
        #----------------------------------------------------------------------
        nc.variables['i_min_n2'].description = 'Zonal index to tilted eddy N2 min'
        nc.variables['i_min_n2'].units = "index"
        #----------------------------------------------------------------------
        nc.variables['j_min_n2'].description = 'Meridional index to tilted eddy N2 min'
        nc.variables['j_min_n2'].units = "index"
        #----------------------------------------------------------------------
        nc.variables['k_min_n2'].description = 'Vertical index to tilted eddy N2 min'
        nc.variables['k_min_n2'].units = "index"
        #----------------------------------------------------------------------
        nc.variables['x_ITE_core_zrange'].description = ''.join(("Depth range ",
            "of zonal ITE N2 minimum; not necessarily centered around 'z_ITE_core'"))
        nc.variables['x_ITE_core_zrange'].units = 'm'
        #----------------------------------------------------------------------
        nc.variables['y_ITE_core_zrange'].description = ''.join(("Depth range ",
            "of meridional ITE N2 minimum; not necessarily centered around 'z_ITE_core'"))
        nc.variables['y_ITE_core_zrange'].units = 'm'
        #----------------------------------------------------------------------

        #----------------------------------------------------------------------
        if 'temp' in medsub_variables:
            nc.variables['temp'].description = 'Temperature from %s' % mdl_type
            nc.variables['temp'].units = "deg C"
        #sf, ao = compute_scale_and_offset(*nc_scaling['temp'])
        #nc.variables['temp'].scale_factor = sf
        #nc.variables['temp'].add_offset = ao
        #----------------------------------------------------------------------
        print ('FIXME')
        if 'salt' in medsub_variables:
            nc.variables['salt'].description = 'Salinity from %s' % mdl_type
            nc.variables['salt'].units = "dimensionless"
        #sf, ao = compute_scale_and_offset(*nc_scaling['salt'])
        #nc.variables['salt'].scale_factor = sf
        #nc.variables['salt'].add_offset = ao
        ##----------------------------------------------------------------------
        if 'u' in medsub_variables:
            nc.variables['u'].description = 'zonal velocity from %s' % mdl_type
            nc.variables['u'].units = "m/s"
        #sf, ao = compute_scale_and_offset(*nc_scaling['u'])
        #nc.variables['u'].scale_factor = sf
        #nc.variables['u'].add_offset = ao
        ##----------------------------------------------------------------------
        if 'v' in medsub_variables:
            nc.variables['v'].description = 'meridional velocity from %s' % mdl_type
            nc.variables['v'].units = "m/s"
        #sf, ao = compute_scale_and_offset(*nc_scaling['v'])
        #nc.variables['v'].scale_factor = sf
        #nc.variables['v'].add_offset = ao
        ##----------------------------------------------------------------------
        if 'vort' in medsub_variables:
            nc.variables['vort'].description = 'vertical vorticity from %s u, v' % mdl_type
            nc.variables['vort'].units = "1/s"
        #sf, ao = compute_scale_and_offset(*nc_scaling['vort'])
        #nc.variables['vort'].scale_factor = sf
        #nc.variables['vort'].add_offset = ao
        ##----------------------------------------------------------------------
        if 'mld' in medsub_variables:
            nc.variables['mld'].description = 'Mixed layer depth from %s' % mdl_type
            nc.variables['mld'].units = "m"
            if 'MFS' in mdl_type:
                nc.variables['mld'].standard_name = "ocean_mixed_layer_thickness_defined_by_sigma_theta"
                nc.variables['mld'].long_name = "Ocean mixed layer thickness defined by density (as in de Boyer Montegut,2004)"
        ##----------------------------------------------------------------------
        #nc.variables['model_eddy_bearing'].description = 'direction of propagation at each vertical level'
        #nc.variables['model_eddy_bearing'].units = "degrees, 0 is east, 90 is north"
        #----------------------------------------------------------------------



def get_eddy_recarray(etrk_cyc, etrk_acyc):
    '''
    Make record array with the eddy properties
    '''
    eddy_observation_list, eddy_dates = [], []

    # Cyclones
    for centlon, centlat, radius, radius_e, date, bearing, eddy_id, amplitude in zip(
                                                    etrk_cyc.lons(),
                                                    etrk_cyc.lats(),
                                                    etrk_cyc.radii(),
                                                    etrk_cyc.radii_eff(),
                                                    etrk_cyc.dates(),
                                                    etrk_cyc.bearings(),
                                                    etrk_cyc.eddy_ids(),
                                                    etrk_cyc.amplitudes()):
        eddy_rec = empty((1), dtype=[('centlon', float),
                                     ('centlat', float),
                                     ('radius', float),
                                     ('radius_e', float),
                                     ('date', float),
                                     ('bearing', float),
                                     ('eddy_id', int),
                                     ('amplitude', float),
                                     ('cyc', int)])
        eddy_rec = eddy_rec.view(recarray)
        eddy_rec.centlon[:] = centlon
        eddy_rec.centlat[:] = centlat
        eddy_rec.radius[:] = radius
        eddy_rec.radius_e[:] = radius_e
        eddy_rec.date[:] = date
        eddy_rec.bearing[:] = bearing
        eddy_rec.eddy_id[:] = eddy_id
        eddy_rec.amplitude[:] = amplitude
        eddy_rec.cyc[:] = -1

        eddy_observation_list.append(eddy_rec)
        eddy_dates.append(date)

    # Anticyclones
    for centlon, centlat, radius, radius_e, date, bearing, eddy_id, amplitude in zip(
                                                    etrk_acyc.lons(),
                                                    etrk_acyc.lats(),
                                                    etrk_acyc.radii(),
                                                    etrk_acyc.radii_eff(),
                                                    etrk_acyc.dates(),
                                                    etrk_acyc.bearings(),
                                                    etrk_acyc.eddy_ids(),
                                                    etrk_acyc.amplitudes()):
        eddy_rec = empty((1), dtype=[('centlon', float),
                                    ('centlat', float),
                                    ('radius', float),
                                    ('radius_e', float),
                                    ('date', float),
                                    ('bearing', float),
                                    ('eddy_id', int),
                                    ('amplitude', float),
                                    ('cyc', int)])
        eddy_rec = eddy_rec.view(recarray)
        eddy_rec.centlon[:] = centlon
        eddy_rec.centlat[:] = centlat
        eddy_rec.radius[:] = radius
        eddy_rec.radius_e[:] = radius_e
        eddy_rec.date[:] = date
        eddy_rec.bearing[:] = bearing
        eddy_rec.eddy_id[:] = eddy_id
        eddy_rec.amplitude[:] = amplitude
        eddy_rec.cyc[:] = 1

        eddy_observation_list.append(eddy_rec)
        eddy_dates.append(date)

    eddy_dates = hstack(eddy_dates)
    eddy_sorted_i = eddy_dates.argsort()
    eddy_unsorted_i = arange(eddy_sorted_i.size)

    return (eddy_observation_list, eddy_dates,
            eddy_sorted_i, eddy_unsorted_i)



def get_restart_index_and_clip(sorted_eddy_observation_list,
                               restart_date):
    '''Find index for restart and trim list accordingly
    '''
    restart_date_i = 0
    while True:
        eddy_rst = sorted_eddy_observation_list[restart_date_i]
        if float(restart_date) == eddy_rst.date:
            break
        restart_date_i += 1
    return sorted_eddy_observation_list[restart_date_i:], restart_date_i









class BaseClass(object):
    """
    """
    __slots__ = (
        '_lon',
        '_lat',
        'xslc',
        'yslc',
        )


    def __init__(self):
        """
        """
        self._lon = None
        self._lat = None

    #def kdt(self, lon, lat, limits, k=4, n_jobs=-1):
        #"""
        #Make kde tree for indices if domain crosses zero meridian
        #"""
        #ppoints = array([lon.ravel(), lat.ravel()]).T
        #ptree = cKDTree(ppoints)
        #pindices = ptree.query(limits, k=k, n_jobs=n_jobs)[1]
        #iind = array([], dtype=int)
        #for pind in pindices.ravel():
            #_, i = unravel_index(pind, lon.shape)
            #iind = r_[iind, i]
            ##jind = np.r_[jind, j]
        #return iind  #, jind

    def set_limit_indices(self):
        """
        """
        self.xslc = slice(abs(self._lon - self.limits[0]).argmin(),
                          abs(self._lon - self.limits[1]).argmin())
        self.yslc = slice(abs(self._lat - self.limits[2]).argmin(),
                          abs(self._lat - self.limits[3]).argmin())

        return
        #if self.zero_crossing:
            #"""
            #Used for a zero crossing, e.g., across Agulhas region
            #"""
            #def half_limits(lon, lat):
                #return array([array([lon.min(), lon.max(),
                                     #lon.max(), lon.min()]),
                              #array([lat.min(), lat.min(),
                                     #lat.max(), lat.max()])]).T
            ## Get bounds for right part of grid
            #lat = self._lat[self.yslc] # [self._lon >= 360 + self.limits[0]]
            ##lon = self._lon[self._lon >= 360 + self.limits[0]]
            #lon = self._lon[self.xslc]

            ##print 'sssssssssssssss',lon, lat, self.limits[0]

            #limits = half_limits(lon, lat)
            ##print self._lon, self._lat, limits
            #iind = self.kdt(self._lon, self._lat, limits)
            #i1 = iind.min()
            ## Get bounds for left part of grid
            ##lat = self._lat[self._lon <= self.limits[1]]
            ##lon = self._lon[self._lon <= self.limits[1]]
            #limits = half_limits(lon, lat)
            #iind = self.kdt(self._lon, self._lat, limits)
            #i0 = iind.max()
            #self.xslc = slice(i0, i1)


    @staticmethod
    def _haversine_dist(lon1, lat1, lon2, lat2):
        """
        Haversine formula to calculate distance between two lon/lat points
        Uses mean earth radius in metres (from ROMS scalars.h) = 6371315.0
        Input:
            lon1, lat1, lon2, lat2
        Return:
            distance (m)
        """
        lon1, lat1, lon2, lat2 = (lon1.copy(), lat1.copy(),
                                  lon2.copy(), lat2.copy())
        dlat = deg2rad(lat2 - lat1)
        dlon = deg2rad(lon2 - lon1)
        deg2rad(lat1, out=lat1)
        deg2rad(lat2, out=lat2)
        a = (sin(0.5 * dlon)) ** 2
        a *= cos(lat1) * cos(lat2)
        a += (sin(0.5 * dlat)) ** 2
        c = 2. * arctan2(sqrt(a), sqrt(1. - a))
        return EARTH_R * c  # return the distance


    @staticmethod
    def vorticity(u, v, dx, dy):
        '''
        Returns vorticity calculated using 'gradient'
        Boundary condition has better extrapolation
        u and v at rho points
        '''
        def vort(u, v, dx ,dy):
            uy, _ = gradient(u, dy, dx)
            _, vx = gradient(v, dy, dx)
            return vx - uy
        xi = zeros_like(u)
        try:
            for k in range(u.shape[0]):
                xi[k] = vort(u[k], v[k], dx, dy)
        except:
            xi[:] = vort(u, v, dx, dy)
        return xi


    @staticmethod
    def _half_interp(h_one, h_two):
        """
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        """
        h_one = h_one.copy()
        h_one += h_two
        h_one *= 0.5
        return h_one


    @staticmethod
    def get_dx_dy_f(lon, lat):
        half_interp = BaseClass._half_interp
        haversine_dist = BaseClass._haversine_dist

        lon, lat = meshgrid(lon.copy(), lat.copy())
        lonu = half_interp(lon[:, :-1], lon[:, 1:])
        latu = half_interp(lat[:, :-1], lat[:, 1:])
        lonv = half_interp(lon[:-1], lon[1:])
        latv = half_interp(lat[:-1], lat[1:])

        # Get pm and pn
        pm = zeros_like(lon)
        pm[:, 1:-1] = haversine_dist(lonu[:, :-1], latu[:, :-1],
                                     lonu[:, 1:], latu[:, 1:])
        pm[:, 0] = pm[:, 1]
        pm[:, -1] = pm[:, -2]

        pn = zeros_like(lon)
        pn[1:-1] = haversine_dist(lonv[:-1], latv[:-1],
                                  lonv[1:], latv[1:])
        pn[0] = pn[1]
        pn[-1] = pn[-2]

        f = sin(deg2rad(lat))
        f *= 4.
        f *= pi
        f /= (23 * 3600 + 56 * 60 + 4.1) #86400.
        return pm, pn, f


    @staticmethod
    def get_eddy_bearing(model_xarr, bearings, tree=None):
        """
        """
        r2d = 180. / pi
        if tree is None:  # get points within effective radius
            points = array([model_xarr.X.ravel(),
                            model_xarr.Y.ravel()]).T
            tree = cKDTree(points)
        #dist = (model_xarr.radius_e /
        #        model_xarr.radius)
        dist = 1.
        thetree = tree.query_ball_point([0, 0], dist, n_jobs=-1)
        #print dist
        for k in range(len(bearings)):
            #print '--', model_xarr.u_in.squeeze()[k].shape
            u = model_xarr.u_out.squeeze()[k].flat[thetree]
            v = model_xarr.v_out.squeeze()[k].flat[thetree]
            #u -= u.mean()
            #v -= v.mean()
            bearings[k] = arctan2(v.mean(), u.mean())
            bearings[k] *= r2d
            if bearings[k] < 0.:
                bearings[k] += 360.
            if 0 and k == 6: # test figure k=6 for 100 m
                uu = model_xarr.u_out.squeeze(1)[k]
                vv = model_xarr.v_out.squeeze(1)[k]
                spd = hypot(uu, vv)
                print ('eddy.bearing', eddy.bearing)
                plt.figure()
                plt.title('u:%s, v:%s, bearing:%s' % (u, v, bearings[k]))
                plt.pcolormesh(model_xarr.X.squeeze(1)[k],
                               model_xarr.Y.squeeze(1)[k],
                               spd)
                plt.plot(array([0, u]) * 10,
                         array([0, v]) * 10, 'w', lw=2)
                plt.axis('image')
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.show()
        return bearings



class MedSubModel(BaseClass):
    """

    """
    __slots__ = (
        'medsub_file',
        'medsub_dir',
        'medsub_filenames',
        'medsub_variables',
        'model',
        'vnm_lon',
        'vnm_lat',
        'vnm_init',
        'vnm_depth',
        'vnm_time',
        'rplc_temp',
        'rplc_salt',
        'rplc_u',
        'rplc_v',
        'rplc_ssh',
        'rplc_mld',
        'jday',
        'limits',
        'zero_crossing',
        'fill_value',
        'xsize',
        'ysize',
        'zsize',
        '_var',
        'production_dates',
        'two_d',
        '_lon',
        '_lat',
        '_fill_value',
        '_depth',
        '_dates',
        '_date_units',
        'base_date',
        'ti',
        'zslc',
        'depth',
        '_mask3d',
        '_mld',
        '_ssh',
        '_temp',
        '_salt',
        '_u',
        '_v',
        '_vort',
        'i_kdt',
        'j_kdt',
        'inds_kdt',
        )



    def __init__(self, medsub_file, medsub_filenames,
                 medsub_variables, limits, depth,
                 two_d=True, production_dates=None,
                 fill_value=9999):
        """

        """
        super(MedSubModel, self).__init__()
        self.medsub_file = medsub_file
        self.medsub_dir = ''.join(medsub_file.rpartition('/')[:-1])
        self.medsub_filenames = medsub_filenames
        self.medsub_variables = medsub_variables
        self.model = medsub_filenames['model']
        # vnm for 'v'ariable 'n'a'm'e
        self.vnm_lon = medsub_filenames['lon']
        self.vnm_lat = medsub_filenames['lat']
        self.vnm_init = medsub_filenames['temp']
        self.vnm_depth = medsub_filenames['depth']
        self.vnm_time = medsub_filenames['time']

        self.rplc_temp = None
        self.rplc_salt = None
        self.rplc_u = None
        self.rplc_v = None
        self.rplc_ssh = None
        self.rplc_mld = None

        self.i_kdt = None
        self.j_kdt = None
        self.inds_kdt = None


        self.jday = None
        offset = 3  #0.1 # degrees
        self.limits = (limits[0] - offset,
                       limits[1] + offset,
                       limits[2] - offset,
                       limits[3] + offset)
        self.zero_crossing = False
        #self.xslc = slice(None, None)
        #self.yslc = slice(None, None)
        self.init(depth)
        print ('check me below')
        self.fill_value = fill_value
        self.xsize = self.xslc.stop - self.xslc.start
        self.ysize = self.yslc.stop - self.yslc.start
        self.zsize = self._depth.size
        if isinstance(depth, tuple):
            zspan = self.zslc.stop - self.zslc.start
        else:
            zspan = self.zslc.stop
        self._var = ma.empty((zspan, self.ysize, self.xsize))
        for medsub_variable in medsub_variables:
            if ('ssh' in medsub_variable or
                'sla' in medsub_variable or
                'mld' in medsub_variable):
                setattr(self, '_%s' % medsub_variable,
                        empty_like(self._var[0]))
            elif 'mask3d' in medsub_variable:
                setattr(self, '_%s' % medsub_variable,
                        empty_like(self._var, dtype=int8))
            else:
                setattr(self, '_%s' % medsub_variable,
                        empty_like(self._var))
        self.production_dates = production_dates
        self.two_d = two_d


    def init(self, depth):
        """
        """
        #print 'AAAAA', self.medsub_file
        with Dataset(glob(self.medsub_file)[0]) as nc:
            self._lon = nc.variables[self.vnm_lon][:]
            self._lat = nc.variables[self.vnm_lat][:]
            if 'MFS' in self.model:
                self._fill_value = nc.variables[self.vnm_init[0]]._FillValue
            else:
                self._fill_value = nc.variables[self.vnm_init]._FillValue
            self._depth = nc.variables[self.vnm_depth][:]
            self._dates = nc.variables[self.vnm_time][:]
            self._date_units = nc.variables[self.vnm_time].units

        #sasa
        if self.limits[0] < 0 and self.limits[1] >= 0:
            self.zero_crossing = True
        #print 'commented for GLO, needs work'
        #elif self.limits[0] < 0 and self.limits[1] <= 0:
            #self._lon -= 360.

        self.set_limit_indices()
        # Set time index
        self.ti = 0
        # Set depth info
        if isinstance(depth, tuple):
            zitop = abs(abs(depth[1]) - self._depth).argmin() + 1
            zibot = abs(abs(depth[0]) - self._depth).argmin()
        else:
            zitop = abs(abs(depth) - self._depth).argmin() + 1
            zibot = None
        self.zslc = slice(zibot, zitop)
        self.depth = self._depth[self.zslc]
        # Set both datetime and jday
        self._set_date()

    def _set_date(self, medsub_file=None):
        """
        Set date and Julian day (jday for comparison with py-eddy-tracker data)
        """
        if medsub_file is not None:
            self.medsub_file = medsub_file
        self.base_date = dt.date2num(parser.parse(
            self._date_units.split(' ')[2:4][0]))
        self.jday = dt.num2julian(self.base_date)

    def set_date_and_index(self, medsub_file, the_date):
        self._set_date(medsub_file)

    def lon(self):
        return self._lon[self.xslc]

    def lat(self):
        return self._lat[self.yslc]


    def fillmask(self, varname):
        '''
        Fill mask after reading data with self.set_variable
        '''
        var = empty_like(self._var)

        if self.i_kdt is None:
            self.i_kdt, self.j_kdt = unravel_index(
                arange(self._var.data[0].size), self._var.data[0].shape)
            self.inds_kdt = array([self.i_kdt.ravel(),
                                   self.j_kdt.ravel()]).T

        if ('ssh' not in varname and
            'mld' not in varname):

            for k in range(var.shape[0]):
                try:
                    #var[k], _ = fillmask_kdtree(self._var.data[k],
                                               #~self._var.mask[k])
                    var[k], _ = fillmask_fast_kdtree(
                        self._var.data[0], ~self._var.mask[0],
                        self.i_kdt, self.j_kdt, self.inds_kdt)
                except: # if no good values
                    var[k] = self.fill_value
            setattr(self, '_%s' % varname, var)

        else:
            var[0], _ = fillmask_kdtree(
                self._var.data[0], ~self._var.mask[0],
                self.i_kdt, self.j_kdt, self.inds_kdt)
            #i, j = unravel_index(arange(self._var.data[0].size), self._var.data[0].shape)
            #inds = array([i.ravel(), j.ravel()]).T
            #var[0], _ = fillmask_fast_kdtree(self._var.data[0],
                                       #~self._var.mask[0],
                                       #i, j, inds)
            setattr(self, '_%s' % varname, var[0])


    def set_variable(self, varname, ymd):
        """
        """
        vnm_var = self.medsub_filenames[varname]
        ymd = ''.join((ymd[0], ymd[1].zfill(2), ymd[2].zfill(2)))

        #if 'MFS' in self.model:
            ## Update production date in filename when necessary
            #production_date = self.medsub_file.rsplit('-b')[-1].split('_an')[0]

            #if str_compare('salt', varname):
                #medsub_file = self.medsub_file.replace(self.rplc_salt[1],
                                                       #self.rplc_salt[0])
                #vnm_var = vnm_var[0]
            #elif str_compare('u', varname):
                #medsub_file = self.medsub_file.replace(self.rplc_u[1],
                                                       #self.rplc_u[0])
                #vnm_var = vnm_var[0]
            #elif str_compare('v', varname):
                #medsub_file = self.medsub_file.replace(self.rplc_v[1],
                                                       #self.rplc_v[0])
                #vnm_var = vnm_var[0]
            #elif str_compare('ssh', varname):
                #medsub_file = self.medsub_file.replace(self.rplc_ssh[1],
                                                       #self.rplc_ssh[0])
                #vnm_var = vnm_var[0]
            #elif str_compare('mld', varname):
                #medsub_file = self.medsub_file.replace(self.rplc_mld[1],
                                                       #self.rplc_mld[0])
                #vnm_var = vnm_var[0]
            #elif str_compare('temp', varname):
                #medsub_file = self.medsub_file
                #vnm_var = vnm_var[0]
            #elif str_compare('mask3d', varname):
                #medsub_file = self.medsub_file

        #else:
        medsub_file = self.medsub_file

        #print 'START', medsub_file
        #if self.production_dates is not None:
            #for new_production_date in self.production_dates:
                #medsub_file_tmp = re.sub('-b%s_an' % production_date,
                                        #' -b%s_an' % new_production_date,
                                        #medsub_file)
                #if os.path.isfile(medsub_file_tmp % ymd):
                    #medsub_file = medsub_file_tmp
                    #continue

        z, y, x = self.zslc, self.yslc, self.xslc
        if ('ssh' not in varname and
            'mld' not in varname):
            #print 'dddddd', medsub_file, varname
            self._var[:] = self._read_var(medsub_file, varname,
                                          vnm_var, ymd, z, y, x)
        else:
            self._var[0] = self._read_var(medsub_file, varname,
                                          vnm_var, ymd, None, y, x)


    @staticmethod
    def _read_var(medsub_file, varname,
                  vnm_var, ymd, zslc, yslc, xslc):
        """
        """
        #print 'ooooooooo', medsub_file
        try:
            medsub_file = medsub_file % ymd
        except:
            medsub_file = medsub_file % (ymd, ymd)
        medsub_file = glob(medsub_file)[0]
        ##print '-----Reading MedSub file:----', medsub_file #% ymd
        tind = int(ymd[2])-1
        #print '---vnm_var--------', vnm_var
        #print medsub_file, ymd
        with Dataset(medsub_file) as nc:
            if ('ssh' not in varname and
                'mld' not in varname):
                return nc.variables[vnm_var][
                    tind, zslc, yslc, xslc].squeeze()
            else:
                return nc.variables[vnm_var][
                        tind, yslc, xslc].squeeze()



    @staticmethod
    def get_vort(u, v, dx, dy):
        """
        """
        return BaseClass.vorticity(u, v, dx, dy)





class Composite3D(Composite):
    """

    """

    __slots__ = (
        'nlev',
        'variables',
        'model_fill_value',
        'fill_value',
        'resolution',
        'X3d',
        'Xshp',
        'Yshp',
        'thevar',
        'tri',
        'dmask',
        'mask2d',
        'J',
        'I',
        'J0',
        'I0',
        'J1',
        'I1',
        'topo_lon',
        'topo_lat',
        'topo',
        'topo_out',
        'xtopoproj',
        'ytopoproj',
        'I_topo',
        'J_topo',
        'srtm_topo_lon',
        'srtm_topo_lat',
        'srtm_topo',
        'srtm_topo_out',
        'srtm_xtopoproj',
        'srtm_ytopoproj',
        'I_srtmtopo',
        'J_srtmtopo',
        'mask3d_out',
        'zos_out',
        'ssh_out',
        'mld_out',
        'temp_out',
        'salt_out',
        'u_out',
        'v_out',
        'vort_out',
        'I0_topo',
        'I1_topo',
        'J0_topo',
        'J1_topo',
        )



    def __init__(self, lon2d, lat2d, nlev, variables,
                 model_fill_value, fill_value=9999,
                 max_eddy_radius=4., resolution=0.2,
                 topo=None, srtm_topo=None):
        """

        """
        super(Composite3D, self).__init__(lon2d, lat2d,
                                          max_eddy_radius=max_eddy_radius,
                                          resolution=resolution,
                                          fill_value=fill_value)
        self.variables = variables
        self.nlev = nlev
        ysize, xsize = lon2d.shape
        #self.data_temp = empty((nlev, ysize, xsize))
        #print('**********************', fill_value)
        y, x = self.X.shape
        self.X3d = ma.empty((nlev, y, x))
        self.Xshp, self.Yshp = self.X.shape[1], self.Y.shape[0]
        self.fill_value = fill_value
        self.model_fill_value = model_fill_value
        self.thevar = None
        for variable in variables:
            if variable in ('zos', 'sossheig', 'ssh', 'mld', 'chl', 'sst') or nlev <= 1:
                setattr(self, '%s_out' % variable, zeros_like(self.X))
            else:
                setattr(self, '%s_out' % variable, zeros_like(self.X3d))

        self.tri = None
        self.dmask = None
        self.mask2d = None

        # Indices for subsetting in self.set_projection
        self.J, self.I = unravel_index(
            arange(self.lon2d.size), self.lon2d.shape)
        self.I0, self.I1 = None, None
        self.J0, self.J1 = None, None

        if topo is not None:
            self.topo_lon = topo[0]
            self.topo_lat = topo[1]
            self.topo = topo[2]
            # In variables
            self.xtopoproj = empty_like(self.topo_lon)
            self.ytopoproj = empty_like(self.topo_lat)
            self.I_topo, self.J_topo = None, None
            # Out variables
            self.topo_out = empty_like(self.X)
            #self.X_proj_topo = empty_like(self.X)
            #self.Y_proj_topo = empty_like(self.X)

        if srtm_topo is not None:
            self.srtm_topo_lon = srtm_topo[0]
            self.srtm_topo_lat = srtm_topo[1]
            self.srtm_topo = srtm_topo[2]
            # In variables
            self.xsrtmtopoproj = empty_like(self.srtm_topo_lon)
            self.ysrtmtopoproj = empty_like(self.srtm_topo_lat)
            self.I_srtmtopo, self.J_srtmtopo = None, None
            # Out variables
            self.srtm_topo_out = empty_like(self.X)
            #self.X_proj_srtm_topo = empty_like(self.X)
            #self.Y_proj_srtm_topo = empty_like(self.X)


    def set_projection(self, radius):
        """
        Make a projection and ensure that (0, 0) corresponds
        to (centlon, centlat) of eddy.
        """
        lon2d = self.lon2d - 360
        lat2d = self.lat2d
        self.proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                         % (self.centlat, self.centlon))
        self.xproj, self.yproj = self.proj(lon2d, lat2d)

        ii, jj, x, y = self._set_IJ_indices(
            self.centlon, self.centlat, radius, lon2d, lat2d, self.I, self.J)

        x, y = meshgrid(linspace(x[0], x[1], self.X.shape[1]),
                        linspace(y[0], y[1], self.Y.shape[0]))
        self.X[:], self.Y[:] = self.proj(x, y)
        print('WARNING: I changed _ii and _jj to ii and jj on 13June19, maybe not correct to this...')
        self.I0, self.I1 = ii
        self.J0, self.J1 = jj


    @staticmethod
    def _set_IJ_indices(centlon, centlat, radius,
                        lon2d, lat2d, I, J, nrad=10):
        '''Generic function for corner indices to bounding box
        into parent 2d domain around eddy
        '''
        #nrad = 10
        # Loop over angles to build local grid around
        # eddy center
        xx, yy, iii, jjj = [], [], [], []

        for i, ang in enumerate([270, 180, 90, 0]):
            # radius * 4 for usual size of compo grid
            x, y = newPosition(
                centlon, centlat, ang, radius*4)
            #print('x============ y=============',x, y)
            # radius * 10 should be enough for interpolation bounds...
            ii, jj = newPosition(
                centlon, centlat, ang, radius*nrad)
            #print ii, jj, centlon, centlat, ang, radius*nrad
            index = sqrt((lon2d - ii) ** 2
                       + (lat2d - jj) ** 2).argmin()
            #print '-------', index, I[index], J[index]
            if i in (0, 2):
                #print('AAAAA', x, xx)
                #xx.append(x[0])
                xx.append(x)
                #print
                iii.append(I[index])
            else:
                #yy.append(y[0])
                yy.append(y)
                jjj.append(J[index])
        #print 'aa', iii, jjj, xx, yy
        return iii, jjj, xx, yy



    def set_topo_projection(self, radius):
        """
        Make a projection and ensure that (0, 0) corresponds
        to (centlon, centlat) of eddy.
        """
        if self.I_topo is None:
            self.J_topo, self.I_topo = unravel_index(
                arange(self.topo_lon.size), self.topo_lon.shape)

        lon2d, lat2d = self.topo_lon, self.topo_lat

        self.proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                         % (self.centlat, self.centlon))
        self.xtopoproj[:], self.ytopoproj[:] = self.proj(lon2d, lat2d)

        ii, jj, xx, yy = self._set_IJ_indices(
            self.centlon, self.centlat, radius, lon2d, lat2d, self.I_topo, self.J_topo)

        self.I0_topo, self.I1_topo = ii
        self.J0_topo, self.J1_topo = jj

        xx, yy = meshgrid(linspace(xx[0], xx[1], self.Xshp),
                          linspace(yy[0], yy[1], self.Yshp))

        self.X[:], self.Y[:] = self.proj(xx, yy)



    def make_composite_topo(self, fill_value):
        """
        """
        inter_LC = scint.CloughTocher2DInterpolator

        J0, J1, I0, I1 = (self.J0_topo, self.J1_topo,
                          self.I0_topo, self.I1_topo)

        tri = Delaunay(
            array([self.xtopoproj[J0:J1, I0:I1].ravel(),
                   self.ytopoproj[J0:J1, I0:I1].ravel()]).T
            )

        topo = self.topo[J0:J1, I0:I1]
        mask = logical_xor(topo == 0, topo == fill_value)
        if mask.any():
            topo[:], _ = fillmask_kdtree(topo, float_(~mask))
            #fig=plt.figure(1000)
            #ax = fig.add_subplot(131)
            #ax.matshow(topo, cmap=plt.cm.gist_ncar)
            #ax.set_clim(0, 5000)
            #ax = fig.add_subplot(132)
            #ax.matshow(mask, cmap=plt.cm.gist_ncar)

        intobj = inter_LC(tri,
                          topo.ravel())#,
                          #fill_value=0)
        self.topo_out[:] = intobj(self.X, self.Y)



    def set_srtm_topo_projection(self, radius):
        """
        Make a projection and ensure that (0, 0) corresponds
        to (centlon, centlat) of eddy.
        """
        needs_updating
        self.xsrtmtopoproj[:], self.ysrtmtopoproj[:] = self.proj(
            self.srtm_topo_lon, self.srtm_topo_lat
            )
        self.xsrtmtopoproj -= self.xoff
        self.ysrtmtopoproj -= self.yoff
        self.xsrtmtopoproj /= radius
        self.ysrtmtopoproj /= radius



    def make_composite_srtm_topo(self, fill_value):
        """
        """
        needs_updating
        inter_LC = scint.CloughTocher2DInterpolator
        intobj = inter_LC(self.srtmtopotri,
                          self.srtm_topo[self.srtmmask],
                          fill_value=fill_value)
        self.srtm_topo_out[:] = intobj(self.X, self.Y)



    def _compo_interp(self, medsub_variable, d0, d1, intobj):
        '''
        '''
        getattr(
            self, '%s_out' % medsub_variable)[d0:d1] = intobj(self.X,
                                                              self.Y).ravel()


    def make_composite_by_vars(self, mdsbmodel, medsub_variable, fill_value):
        """
        Override Composite.make_composite
        """
        inter_LC = scint.CloughTocher2DInterpolator
        intern_LC = scint.NearestNDInterpolator
        #Delaunay_LC = Delaunay
        xsize = self.X.size
        J0, J1, I0, I1 = self.J0, self.J1, self.I0, self.I1
        tri = self.tri
        #print('--------------IN %s' % medsub_variable)
        if ('ssh' in medsub_variable or
            'mld' in medsub_variable or
            'sst' in medsub_variable or
            'chl' in medsub_variable):

            d0, d1 = 0, xsize
            intobj = inter_LC(
                tri, getattr(mdsbmodel, '_%s' %
                    medsub_variable).data[J0:J1, I0:I1].ravel())
            self._compo_interp(medsub_variable, d0, d1, intobj)

        else:  # 'mask3d' in medsub_variable:
            for k in range(zspan):
                d0 = k * xsize
                d1 = d0 + xsize
                if 'mask3d' in medsub_variable:
                    mask2d = getattr(mdsbmodel, '_%s' %
                        medsub_variable).data[k, J0:J1, I0:I1].ravel()
                    intobj = intern_LC(tri, mask2d.astype(int))
                        #self.tri, mask2d.data[J0:J1, I0:I1].astype(int))
                else:
                    intobj = inter_LC(
                        tri,
                        getattr(mdsbmodel, '_%s' %
                                medsub_variable).data[k, J0:J1, I0:I1].ravel())
                self._compo_interp(medsub_variable, d0, d1, intobj)
        #print('--------------OUT %s' % medsub_variable)





    #def make_composite_by_depths(self, mdsbmodel, k, fill_value):
        #"""
        #Override Composite.make_composite
        #"""
        #inter_LC = scint.CloughTocher2DInterpolator
        ##Delaunay_LC = Delaunay

        #d0 = k * self.X.size
        #d1 = d0 + self.X.size

        #for medsub_variable in mdsbmodel.medsub_variables:
            ##print '------', medsub_variable
            #intobj = inter_LC(self.tri,
                              #getattr(mdsbmodel, '_%s' % medsub_variable).data[k].ravel())
            #getattr(self, '%s_out' % medsub_variable)[d0:d1] = intobj(self.X,
                                                                    #self.Y).ravel()



    @staticmethod
    def check_mask_constant(mask1, mask2, varname):
        if distance.hamming(mask1.ravel(),
                            mask2.ravel()):
            print ('Alert, change of mask detected for *%s*' % varname)
            return True




    @staticmethod
    def vorticity(u, v, dx, dy):
        '''
        Returns vorticity calculated using 'gradient'
        Boundary condition has better extrapolation
        u and v at rho points
        '''
        def vort(u, v, dx ,dy):
            #uy, _ = gradient(u, dy, dx)
            #_, vx = gradient(v, dy, dx)
            _, uy = gradient(u, dx, dy)
            vx, _ = gradient(v, dx, dy)
            return vx - uy
        xi = zeros_like(u)
        try:
            for k in range(u.shape[0]):
                xi[k] = vort(u[k], v[k], dx, dy)
        except:
            xi[:] = vort(u, v, dx, dy)
        return xi


def get_mercator_data():

    params = ()
    python = '/home/emason/VENVP2/bin/python'
    motu = '/home/emason/Dropbox/MOTU_client/motu-client-python-master/src/python/motu-client.py'
    user = 'emason'
    pwd = 'Nge4*yywM743'
    m = 'http://nrtcmems.mercator-ocean.fr/motu-web/Motu'
    s = 'GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS'
    d = 'global-analysis-forecast-phy-001-024'
    x, X, y, Y, z, Z = -15, 1, 5, 55, 0.49, 5727.91
    t, T = "2008-01-01", "2008-01-02"
    o = '/home/emason/toto/'
    f = 'savefile.nc'


    data_call = '/home/emason/VENVP2/bin/python /home/emason/Dropbox/MOTU_client/motu-client-python-master/src/python/motu-client.py -u %s -p %s -m http://nrtcmems.mercator-ocean.fr/motu-web/Motu -s GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS -d global-analysis-forecast-phy-001-024 -x -15. -X 1.0 -y 5. -Y 55. -z  0.49 -Z 5727.91 -v thetao -t "2008-01-01" -T "2008-01-02" -o /home/emason/toto/ -f current_date.nc' % ('emason', 'Nge4*yywM743')
    subprocess.check_call(data_call, shell=True)




if __name__ == '__main__':

    plt.close('all')
    #-------------------------------------------------------------------


    cmems_model = 'GLO'

    restart = False

    add_model_topo = True
    add_SRTM_topo = False


    # Choose start and end dates
    #start_date, end_date = (2012, 1, 1), (2012, 12, 30)  # ITE
    start_date, end_date = (2007, 1, 1), (2007, 12, 30)  # ITE

    # Bounding region
    #lonmin, lonmax, latmin, latmax = (-75., -58., 24., 32.) # ITE work (Sargasso test)
    #lonmin, lonmax, latmin, latmax = (-80., -29., 20., 45.) # ITE work (larger test)
    lonmin, lonmax, latmin, latmax = (-80., -5., 20., 45.) # ITE work (2007 test)

    # This can be an integer or a tuple (dmin, dmax)
    depth = (0, 1500) #2500 # choose a lowest depth (m)
    #depth = (0, 2) #2500 # choose a lowest depth (m)


    #savedir = '/marula/emason/aviso_eddy_tracking/MedSub_models/'
    #savedir = '/marula/emason/aviso_eddy_tracking/ITE_work/NATL_GLO/toto/'
    savedir = '/marula/emason/aviso_eddy_tracking/ITE_work/NATL_GLO/2007/'
    #savedir = '/marula/emason/aviso_eddy_tracking/ITE_work/NATL_GLO/'


    if 'GLO' in cmems_model:
        medsub_variables = ['temp', 'salt', 'ssh', 'mld', 'vort']
        #medsub_variables = ['ssh', 'temp', 'salt', 'vort']
        compo_resolution = 0.08
        #eddy_dir = '/data_cmems/eddy_tracking/GLO/'
        #eddy_dir = '/data_cmems/eddy_tracking/GFM_new/'
        #eddy_dir = '/marula/emason/aviso_eddy_tracking/MedSub_models/GLOBAL_ANALYSIS_FORECAST_PHY_001_024-eddies/'
        #eddy_dir = '/marula/emason/aviso_eddy_tracking/ITE_work/NATL_GLO/'
        eddy_dir = '/marula/emason/aviso_eddy_tracking/ITE_work/NATL_GLO/2007/'
        #model_dir = '/marula/emason/data/Copernicus/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/'
        #model_dir = '/marula/emason/data/Copernicus/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/NATLAN_2012/'
        model_dir = '/marula/emason/data/Copernicus/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/NATLAN_2012/2007/'
        #model_dir = '/marula/emason/data/Copernicus/GLOBAL_ANALYSIS_FORECAST_PHY_001_024/NATLAN_2012/'
        grid_file = ('grid_spec.nc', 'grid_x_T', 'grid_y_T', 'depth_t')
        medsub_filenames = dict(model=cmems_model,
                                filenames='GLOB_MERC_PHY_001_024_????????.nc',
                                ssh='zos',
                                temp='thetao',
                                salt='so',
                                u='uo',
                                v='vo',
                                mld='mlotst',
                                lon='longitude',
                                lat='latitude',
                                depth='depth',
                                time='time',
                                mask3d='thetao')

    else:
        raise Exception('%s not implemented' % cmems_model)


    eddy_file_cyc = 'Cyclonic.nc'
    eddy_file_acyc = 'Anticyclonic.nc'


    #savefile = 'my_%s_composites_NATL_small.nc' % cmems_model
    #savefile = 'my_%s_composites_NATL_to_be_renamed.nc' % cmems_model
    savefile = 'my_%s_composites_NATL_fix_mask3d.nc' % cmems_model



    srtm_dir = '/marula/emason/data/topo/SRTM30_PLUS_V11/'


    nc_scaling = dict(temp=(1., 36., 16),
                      salt=(30., 40., 16),
                      u=(-3., 3., 32),
                      v=(-3., 3., 32),
                      vort=(-3., 3., 32))

    fill_value = 9999
    norm_cutoff = 5
    topo_pad = 5


    #-------------------------------------------------------------------
    if 'vort' in medsub_variables and 'u' not in medsub_variables:
        medsub_variables.append('u')
    if 'vort' in medsub_variables and 'v' not in medsub_variables:
        medsub_variables.append('v')
    # Remove duplicates, just in case ...
    medsub_variables = list(set(medsub_variables))
    # Ensure 'vort' is last if is defined
    medsub_variables = sorted(medsub_variables)

    # Prepend to ensure it's first
    medsub_variables.insert(0, 'mask3d')


    #-------------------------------------------------------------------
    if add_model_topo:
        print ('------ Reading model topography')
        with Dataset(model_dir + grid_file[0]) as nc:
            topo_lon = nc.variables[grid_file[1]][:]
            topo_lat = nc.variables[grid_file[2]][:]
            topo = nc.variables[grid_file[3]][:]
        if topo_lon.min() >= 0:
            topo_lon -= 180
            s_i = topo_lon.size / 2
            topo = concatenate((topo[:, s_i:],
                                topo[:,: s_i]), axis=1)
        if topo_lon.squeeze().ndim == 1:
            topo_i0, topo_i1 = (abs(topo_lon - (lonmin - topo_pad)).argmin(),
                                abs(topo_lon - (lonmax + topo_pad)).argmin())
            topo_j0, topo_j1 = (abs(topo_lat - (latmin - topo_pad)).argmin(),
                                abs(topo_lat - (latmax + topo_pad)).argmin())
            topo_lon, topo_lat = meshgrid(topo_lon[topo_i0:topo_i1],
                                          topo_lat[topo_j0:topo_j1])

        topo = topo[topo_j0:topo_j1, topo_i0:topo_i1]

        #topo = topo#.flatten()
        #topo_lon = topo_lon.flatten()
        #topo_lat = topo_lat.flatten()


    if add_SRTM_topo:
        print ('------ Reading SRTM topography')
        srtm_files = glob(srtm_dir + 'w*.nc')
        thesrtmfiles = []
        for srtm_file in srtm_files:
            with Dataset(srtm_file) as nc:
                srtm_lon = nc.variables['x'][:]
                srtm_lat = nc.variables['y'][:]
            srtm_xsize = srtm_lon.size
            srtm_ysize = srtm_lat.size
            if not (srtm_lon >= lonmin).any():
                continue
            if not (srtm_lon <= lonmax).any():
                continue
            if not (srtm_lat >= latmin).any():
                continue
            if not (srtm_lat <= latmax).any():
                continue
            thesrtmfiles.append([srtm_file, srtm_lon.mean(), srtm_lat.mean()])

        '''No time right now make auto selection of files for concatenating'''
        #for thesrtmfile in thesrtmfiles:
        with Dataset(thesrtmfiles[0][0]) as nc:
            srtm_lon = nc.variables['x'][:]
            srtm_lat = nc.variables['y'][:]
            topo_i0, topo_i1 = (abs(srtm_lon - (lonmin - topo_pad)).argmin(),
                                abs(srtm_lon - (lonmax + topo_pad)).argmin())
            topo_j0, topo_j1 = (abs(srtm_lat - (latmin - topo_pad)).argmin(),
                                abs(srtm_lat - (latmax + topo_pad)).argmin())
            srtm_topo = nc.variables['z'][topo_j0:topo_j1, topo_i0:topo_i1]
        srtm_lon, srtm_lat = (srtm_lon[topo_i0:topo_i1],
                              srtm_lat[topo_j0:topo_j1])

        with Dataset(thesrtmfiles[1][0]) as nc:
            srtm_lon = nc.variables['x'][:]
            srtm_lat_tmp = nc.variables['y'][:]
            topo_i0, topo_i1 = (abs(srtm_lon - (lonmin - topo_pad)).argmin(),
                                abs(srtm_lon - (lonmax + topo_pad)).argmin())
            topo_j0, topo_j1 = (abs(srtm_lat_tmp - (latmin - topo_pad)).argmin(),
                                abs(srtm_lat_tmp - (latmax + topo_pad)).argmin())
            srtm_topo = concatenate((srtm_topo,
                                     nc.variables['z'][topo_j0:topo_j1, topo_i0:topo_i1]),
                                     axis=0).flatten()
        srtm_topo *= -1
        srtm_lon, srtm_lat= (srtm_lon[topo_i0:topo_i1],
                             hstack((srtm_lat, srtm_lat_tmp[topo_j0:topo_j1])))
        srtm_lon, srtm_lat = meshgrid(srtm_lon, srtm_lat)
        srtm_lon = srtm_lon.flatten()
        srtm_lat = srtm_lat.flatten()





    #aaaaaaaa


    #-------------------------------------------------------------------
    print ('------ Working on *py-eddy-tracker* nc files at %s' % eddy_dir)

    etrk_cyc = EddyTrack(eddy_dir, eddy_file_cyc,
                         (start_date, end_date),
                         (lonmin, lonmax, latmin, latmax))

    etrk_acyc = EddyTrack(eddy_dir, eddy_file_acyc,
                         (start_date, end_date),
                         (lonmin, lonmax, latmin, latmax))

    #etrk_acyc.lons()
    #etrk_cyc.lons()
    #etrk_acyc._lon -= 180.
    #etrk_cyc._lon -= 180.

    # Cyclonic and anticyclones
    (eddy_observation_list, eddy_dates,
     eddy_sorted_i, eddy_unsorted_i) = get_eddy_recarray(etrk_cyc, etrk_acyc)


    #-------------------------------------------------------------------
    print ('------ Working on *MODEL* files at %s' % model_dir)

    thesavefile = savedir + savefile

    ncpu = min(cpu_count(), len(medsub_variables))

    if 'GLO' in cmems_model:
        medsub_files = sorted(glob(model_dir +
                                   medsub_filenames['filenames']))
        production_dates = None



    # For cases where all the records are in a single file (e.g., GLO)
    if len(medsub_files) == 1:
        # make list repeating filename *time* times
        with Dataset(medsub_files[0]) as nc:
            model_dates = nc.variables[medsub_filenames['time']][:]
            if 'GLO' in cmems_model:
                numdates = model_dates.size
                medsub_files = medsub_files * numdates
    else:
        model_dates = []
        for medsub_file in medsub_files:
            with Dataset(medsub_file) as nc:
                try:
                    ttmp = int(nc.variables[medsub_filenames['time']][:])
                except:
                    ttmp = int(nc.variables[medsub_filenames['time']][:].data)
                model_dates.append(ttmp)

    assert (diff(model_dates) == diff(model_dates)[0]).all(), 'bad file somewhere'


    limits = (lonmin, lonmax, latmin, latmax)

    medsub_ini_file = medsub_files[0]



    mdsbmodel = MedSubModel(medsub_ini_file,
                            medsub_filenames,
                            medsub_variables,
                            limits,
                            depth,
                            two_d=True,
                            fill_value=fill_value,
                            production_dates=production_dates)



    #else:
    medsub_file_dt = mdsbmodel.medsub_file.rpartition('/')[-1].rpartition(
        '_')[-1].rpartition('.')[0]
    mdsbmodel.medsub_file = mdsbmodel.medsub_file.replace(
        medsub_file_dt, '%s')
    tunit = 1.

    model_dates = floor(
        array([(mdsbmodel.base_date + (c / tunit)) for c in model_dates]))

    mdsbmodel.lon()
    mdsbmodel._lon += 360

    dx, dy, coriolis = mdsbmodel.__class__.get_dx_dy_f(mdsbmodel.lon(),
                                                       mdsbmodel.lat())
    mdsbmodel_lon, mdsbmodel_lat = meshgrid(mdsbmodel.lon(),
                                            mdsbmodel.lat())


    # -------Compo----------------------------------------------------
    if isinstance(depth, tuple):
        zspan = mdsbmodel.zslc.stop - mdsbmodel.zslc.start
    else:
        zspan = mdsbmodel.zslc.stop

    if add_model_topo:
        model_topo = (topo_lon, topo_lat, topo)
    else:
        model_topo = None

    if add_SRTM_topo:
        srtm_topo = (srtm_lon, srtm_lat, srtm_topo)
    else:
        srtm_topo = None



    compo = Composite3D(mdsbmodel_lon, mdsbmodel_lat,
                        zspan, medsub_variables, mdsbmodel._fill_value,
                        fill_value=mdsbmodel.fill_value,
                        resolution=compo_resolution,
                        topo=model_topo,
                        srtm_topo=srtm_topo)

    model_shp = mdsbmodel._var.shape
    compo_2dshp = compo.X.shape
    compo_3dshp = compo.X3d.shape

    ## prepare for bearings comnputations
    #bearings = empty(model_shp[0])
    #points = array([compo.X.ravel(),
                    #compo.Y.ravel()]).T
    #tree = cKDTree(points)

    for medsub_variable in medsub_variables:

        if ('ssh' not in medsub_variable and
            'mld' not in medsub_variable):
            setattr(compo, '%s_out' % medsub_variable,
                    sharedctypes.RawArray('d', compo.X3d.size))
            memoryview(getattr(compo, '%s_out' % medsub_variable))[:] = array(
                zeros_like(compo.X3d, dtype=float64))
        elif 'mask3d' in medsub_variable:
            setattr(compo, '%s_out' % medsub_variable,
                    sharedctypes.RawArray('i', compo.X.size))
            memoryview(getattr(compo, '%s_out' % medsub_variable))[:] = array(
                zeros_like(compo.X, dtype=int8))
        else:
            setattr(compo, '%s_out' % medsub_variable,
                    sharedctypes.RawArray('d', compo.X.size))
            memoryview(getattr(compo, '%s_out' % medsub_variable))[:] = array(
                zeros_like(compo.X, dtype=float64))


    ## Set up dtypes
    #the_dtypes = [(''.join((medsub_variable, '_in')), float,
                   #(model_shp if ('ssh' not in medsub_variable and
                                  #'mld' not in medsub_variable) else model_shp[1:]))
                    #for medsub_variable in medsub_variables]



    print ('------ matching eddy properties to MedSubModel')
    sorted_eddy_observation_list = [eddy_observation_list[i] for i in eddy_sorted_i]
    if not restart:
        eddy_i = 0
    else:
        with Dataset(thesavefile) as nc:
            restart_date = nc.variables['eddy_date'][:]
        sorted_eddy_observation_list, eddy_i = get_restart_index_and_clip(
            sorted_eddy_observation_list, restart_date)


    # Loop over eddy observations
    ref_date = -1.

    # Set up xarray
    data_vars={'salt': (('z', 'y', 'x'), mdsbmodel._salt),
               'temp': (('z', 'y', 'x'), mdsbmodel._temp),
               'u': (('z', 'y', 'x'), mdsbmodel._u),
               'v': (('z', 'y', 'x'), mdsbmodel._v),
               'vort': (('z', 'y', 'x'), mdsbmodel._vort),
               'ssh': (('y', 'x'), mdsbmodel._ssh),
               'X': (('y_compo', 'x_compo'), compo.X),
               'Y': (('y_compo', 'x_compo'), compo.Y),
               'date': (('t'), array([0.])),
               'centlon': (('t'), array([0.])),
               'centlat': (('t'), array([0.])),
               'radius': (('t'), array([0.])),
               'radius_e': (('t'), array([0.])),
               'amplitude': (('t'), array([0.])),
               'eddy_id': (('t'), array([0])),
               'cyc': (('t'), array([0]))}
    coords={'x':mdsbmodel_lon[0],
            'y':mdsbmodel_lat[:,0],
            'z':mdsbmodel.depth.data,
            't':array([0]),
            'x_compo':compo.X[0],
            'y_compo':compo.Y[:,0]}
    model_xarr = xr.Dataset(data_vars=data_vars,
                            coords=coords)



    # Loop over eddies...
    for eddy in sorted_eddy_observation_list:
        #print '********', eddy.date
        if eddy.date in model_dates:

            the_date = dt.num2date(eddy.date)[0]
            ymd = (str(the_date.year),
                   str(the_date.month).zfill(2),
                   str(the_date.day))

            # Update required MedSub variables
            if eddy.date != ref_date:  # only read when date changes
                print ('------ updating MedSub variables for %s' % dt.num2date(
                    eddy.date)[0].isoformat())
                for medsub_variable in medsub_variables:
                    if 'vort' in medsub_variable:
                        for k in range(zspan):
                            mdsbmodel._vort[k] = mdsbmodel.get_vort(
                                mdsbmodel._u[k],
                                mdsbmodel._v[k],
                                dx, dy)
                        mdsbmodel._vort /= coriolis
                    elif 'mask3d' in medsub_variable:  # add 3D mask (reads *salt.mask*)
                        mdsbmodel.set_variable(medsub_variable, ymd)
                        mdsbmodel._mask3d[:] = logical_not(mdsbmodel._var.mask).astype(int8)
                    else:
                        medsub_file_dt = mdsbmodel.medsub_file.rpartition('/')[-1].partition('_')[0]
                        mdsbmodel.set_variable(medsub_variable, ymd)
                        sasa
                        mdsbmodel.fillmask(medsub_variable)
                        if 'mld' in medsub_variable:
                            mdsbmodel._mld[:] = gaussian_filter(mdsbmodel._mld, 0.25)
                        #ssss
                    #print '------------ medsub_variable', medsub_variable

                ref_date = eddy.date.copy()

            #sasa
            model_xarr['date'][0] = eddy.date[0]
            model_xarr['centlon'][0] = eddy.centlon[0]
            model_xarr['centlat'][0] = eddy.centlat[0]
            model_xarr['radius'][0] = eddy.radius[0]
            model_xarr['radius_e'][0] = eddy.radius_e[0]
            model_xarr['amplitude'][0] = eddy.amplitude[0]
            model_xarr['eddy_id'][0] = eddy.eddy_id[0]
            model_xarr['cyc'][0] = eddy.cyc[0]

            #print eddy.eddy_id[0]

            try:  # to get model data
                for medsub_variable in medsub_variables:
                    compovar = '%s_out' % medsub_variable
                    #print 'oooooooooooooooooooo', medsub_variable
                    if ('ssh' in medsub_variable or
                        'mld' in medsub_variable):
                        model_xarr[compovar] = (('y_compo', 'x_compo'), compo.X)
                        setattr(compo, '%s_out' % medsub_variable,
                                sharedctypes.RawArray('d', compo.X.size))
                        memoryview(getattr(
                            compo, '%s_out' % medsub_variable))[:] = array(
                                    zeros_like(compo.X, dtype=float64)
                                    )
                    elif 'mask3d' in  medsub_variable:
                        model_xarr[compovar] = (('z', 'y_compo', 'x_compo'), compo.X3d)
                        setattr(compo, '%s_out' % medsub_variable,
                                sharedctypes.RawArray('i', compo.X3d.size))
                        memoryview(getattr(
                            compo, '%s_out' % medsub_variable))[:] = array(
                                    zeros_like(compo.X3d, dtype=int32)
                                    )
                    else:
                        model_xarr[compovar] = (('z', 'y_compo', 'x_compo'), compo.X3d)
                        setattr(compo, '%s_out' % medsub_variable,
                                sharedctypes.RawArray('d', compo.X3d.size))
                        memoryview(getattr(
                            compo, '%s_out' % medsub_variable))[:] = array(
                                    zeros_like(compo.X3d, dtype=float64)
                                    )
            except:
                pass #raise Exception('Something wrong here')


            compo.centlon[:] = eddy.centlon
            compo.centlat[:] = eddy.centlat
            compo.set_projection(model_xarr.radius.data[0])

            #dmask = ~logical_or(
                #ma.masked_greater(abs(compo.xproj), norm_cutoff).mask,
                #ma.masked_greater(abs(compo.yproj), norm_cutoff).mask)
            #compo.dmask = dmask
            #ilist = compo.ilist
            #compo.tri = Delaunay(array([compo.xproj,
                                        #compo.yproj]).T)

            compo.tri = Delaunay(
                array([compo.xproj[compo.J0:compo.J1, compo.I0:compo.I1].ravel(),
                       compo.yproj[compo.J0:compo.J1, compo.I0:compo.I1].ravel()]).T)

            #print('starting Pool with %s cores' % ncpu)
            pool = Pool(processes=ncpu)
            #pool.imap_unordered(mp_mkcompo, medsub_variables)
            pool.map(mp_mkcompo, medsub_variables)
            pool.close()
            pool.join()
            #print('Pooling done')
            sasa

            for medsub_variable in medsub_variables:
                compovar = '%s_out' % medsub_variable
                #print compovar
                try:
                    #print '1'
                    model_xarr[compovar] = (('z', 'y_compo', 'x_compo'), array([
                        getattr(compo, compovar)]).reshape(compo_3dshp))
                except:
                    #print '2'
                    model_xarr[compovar] = (('y_compo', 'x_compo'), array([
                        getattr(compo, compovar)]).reshape(compo_2dshp))


            if add_model_topo:  #NOTE this could be added to variables???

                compo.set_topo_projection(model_xarr.radius.data[0])

                #j0, j1, i0, i1 = (compo.J0_topo, compo.J1_topo,
                                  #compo.I0_topo, compo.I1_topo)

                #compo.topotri = Delaunay(
                    #array([compo.xtopoproj[j0:j1, i0:i1].ravel(),
                           #compo.ytopoproj[j0:j1, i0:i1].ravel()]).T)
                compo.make_composite_topo(fill_value)
                #topo_iout = compo.topo_out < mdsbmodel.depth.min()
                #compo.topo_out[topo_iout] = fill_value
                #compo.topo_out = ma.masked_equal(compo.topo_out, fill_value)
                #sasa

            if add_SRTM_topo:

                compo.set_srtm_topo_projection(model_xarr.radius.data[0])

                #j0, j1, i0, i1 = (compo.J0_srtm_topo, compo.J1_srtm_topo,
                                  #compo.I0_srtm_topo, compo.I1_srtm_topo)

                #compo.srtmtopotri = Delaunay(
                    #array([compo.xsrtmtopoproj[j0:j1, i0:i].ravel(),
                           #compo.ysrtmtopoproj[j0:j1, i0:i].ravel()]).T)

                compo.make_composite_srtm_topo(fill_value)
                topo_iout = compo.srtm_topo_out < mdsbmodel.depth.min()
                compo.srtm_topo_out[topo_iout] = fill_value
                compo.srtm_topo_out = ma.masked_equal(compo.srtm_topo_out,
                                                      fill_value)
            #aaaaaa
            #print '-----------DDDDDDDDDD'
            # Save to nc
            if eddy_i == 0:
                print('#### Creating new file %s' % (thesavefile))
                #model_xarr.X[:] = compo.X
                #model_xarr.Y[:] = compo.Y
                create_nc(thesavefile, model_xarr, (start_date, end_date),
                          (lonmin, lonmax, latmin, latmax), medsub_variables,
                           mdsbmodel.fill_value, mdsbmodel, nc_scaling,
                           add_model_topo=add_model_topo,
                           add_SRTM_topo=add_SRTM_topo)
                print('#### File created')


            with Dataset(thesavefile, 'a') as nc:

                i = eddy_sorted_i[eddy_i]

                for medsub_variable in medsub_variables:

                    if medsub_variable in medsub_variables[0]:
                        nc.variables['time'][i] = model_xarr.date
                        nc.variables['centlon'][i] = model_xarr.centlon
                        nc.variables['centlat'][i] = model_xarr.centlat
                        nc.variables['radius'][i] = model_xarr.radius
                        nc.variables['radius_eff'][i] = model_xarr.radius_e
                        nc.variables['amplitude'][i] = model_xarr.amplitude
                        #nc.variables['eddy_bearing'][eddy_i] = model_xarr.bearing
                        nc.variables['eddy_id'][i] = model_xarr.eddy_id
                        nc.variables['cyc'][i] = model_xarr.cyc

                    if 'mask3d' not in medsub_variable:
                        nc.variables[medsub_variable][i] = ctypeslib.array(
                            getattr(model_xarr, ''.join((
                                medsub_variable, '_out')))).squeeze()
                    else:  # mask3d
                        nc.variables[medsub_variable][i] = ctypeslib.array(
                            getattr(model_xarr, ''.join((
                                medsub_variable, '_out')))).squeeze().round()


                if add_model_topo:
                    nc.variables['topo_model'][i] = compo.topo_out

                if add_SRTM_topo:
                    nc.variables['topo_SRTM'][i] = compo.srtm_topo_out

                if eddy_i == 0:
                    #y, x = model_xarr.X.shape
                    nc.variables['x'][:] = model_xarr.X[0]
                    nc.variables['y'][:] = model_xarr.Y[:, 0]
                    nc.variables['z'][:] = model_xarr.z

                #nc.variables['eddy_i'][:] = eddy_i
                nc.variables['eddy_date'][:] = int(floor(eddy.date))
                #print eddy.date, int(floor(eddy.date))

            #print '-----------EEEEEEEEEEE'

            eddy_i += 1
            #sasa

    print('\nDone')
