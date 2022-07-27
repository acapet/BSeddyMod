# -*- coding: utf-8 -*-
"""
What's here:

1. emMaxVar(velx, vely) - principal axis of a scatter plot wrt the horizontal
2. distLonLat(lon1, lat1, lon2, lat2) - Haversine formula to calculate distance between points
3. distLonLatRhumb(lon1, lat1, lon2, lat2) - Rhumb line distance between 2 points
4. bearingCalc(pt1, pt2) - Calculate the bearing of point 2 from point 1.
5. newPosition(lonin, latin, angle, distance) - Given the inputs return
   the lon, lat of the new position...
6. newPositionRhumb(lon, lat, angle, distance) - Given the inputs return
   the lon, lat of the new position using Rhumb lines...
7. saveFig(figName, dpi, format) - save a figure given inputs (dpi normally 300)
10. get_KE(u, v, mask=None): Calculate kinetic energy
11. u2rho_2d(uFieldIn):
12. v2rho_2d(uFieldIn):
13. getPmPnData(file): returns pm, pn, dx, dy and mask from a roms file
14. get_time_mean(var): returns the mean over a 3-D time series
15. get_time_mean_and_dev(var): returns the deviation from the mean AND the mean itself
    over a 3-D time series
16. get_EKE(u, v, dx, dy, dt, mask): retruns the intgrated over time EKE for a 2-D field



"""
import time
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import numpy as np
from numpy import where, vstack
#import griddata as GD
import os
import matplotlib.transforms as BB
import matplotlib.path       as PP
#from scipy.sandbox.delaunay import *
#from scipy.sandbox import delaunay
from matplotlib import tri
#import matplotlib.tri as mpl_tri
#from mpl_toolkits.basemap import interp
from scipy import io
#from matplotlib.cbook import Bunch
#import matplotlib.delaunay as dy
import scipy.ndimage as nd
import scipy.spatial as sp
import matplotlib.mlab as mlab
from scipy.interpolate import Rbf,griddata
import scipy.interpolate.interpnd as interpnd
import numexpr as ne
import datetime
from numba import jit, njit

#import _iso # this is from octant

def emMaxVar(velx, vely):
    '''
    This program calculates the angle forming
    the principal axis of a scatter plot wrt the horizontal,
    and calculates the components in the new system coordinates.
    Uses Emery-Thompson pg 326
    Returns the projected components and the angle in degrees wrt to E
    '''

    # get rid of nans
    velx1 = velx; vely1 = vely
    ind   = plt.find(np.isnan(velx))
    velx1[ind] = []; vely1[ind] = []

    # Covariance matrix
    c = np.cov(velx1, vely1)

    # Calculate angle, following 4.3.23b
    ang = 0.5 * np.arctan2(2. * c[0,1], (c[0,0]-c[1,1]))
    angulo = ang * 180. / np.pi + 180. # angle wrt E, indicates current direction
    angdeg = 270. - ang * 180. / np.pi

    # Calculate the velocity components in the new axes
    nvelx =  velx * np.cos(ang) + vely * np.sin(ang)
    nvely = -velx * np.sin(ang) + vely * np.cos(ang)
    return nvelx, nvely, angdeg



def distLonLat(lon1, lat1, lon2, lat2):
    '''
    Haversine formula to calculate distance between one point and another
      --uses mean earth radius in metres (from scalars.h) = 6371315.0
    '''
    radius   = 6371315.0 # Mean earth radius in metres (from scalars.h)
    lon1     = np.deg2rad(lon1); lon2 = np.deg2rad(lon2)
    lat1     = np.deg2rad(lat1); lat2 = np.deg2rad(lat2)
    dlat     = lat2 - lat1
    dlon     = lon2 - lon1
    a        = np.power(np.sin(0.5*dlat),2) + \
                   np.cos(lat1) * np.cos(lat2) * np.power(np.sin(0.5*dlon),2)
    c        = 2. * np.arctan2(np.sqrt(a), np.sqrt(1. - a))
    distance = radius * c
    # get the bearing
    bearing  = np.arctan2(np.sin(dlon) * np.cos(lat2), np.cos(lat1) * \
                   np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    bearing  = np.rad2deg(bearing)
    return distance, bearing



def distLonLatRhumb(lon1, lat1, lon2, lat2):
    '''
    Rhumb line distance between 2 points
    There are some caveats that I haven't covered here!!
       check: http://www.movable-type.co.uk/scripts/LatLong.html
    '''
    radius   = 6371315.0 # Mean earth radius in metres (from scalars.h)
    lon1     = np.deg2rad(lon1); lon2 = np.deg2rad(lon2)
    lat1     = np.deg2rad(lat1); lat2 = np.deg2rad(lat2)
    dPhi     = np.log(np.tan(0.5*lat2 + 0.25*np.pi) / np.tan(0.5*lat1 + 0.25*np.pi))
    dlat     = lat2 - lat1
    dlon     = lon2 - lon1
    q        = dlat / dPhi
    distance = np.sqrt(np.power(dlat,2) + (np.power(q,2) * np.power(dlon,2))) * radius
    bearing  = np.arctan2(dlon, dPhi)
    bearing  = np.rad2deg(bearing)
    return distance, bearing



def newPosition(lon, lat, angle, distance):
    '''
    Given the inputs (base lon, base lat, angle [deg], distance [m]) return
    the lon, lat of the new position...
    '''
    lon    = np.deg2rad(lon)
    lat    = np.deg2rad(lat)
    angle  = np.deg2rad(angle)
    radius = 6371315.0 # Mean earth radius in metres
    d_R    = distance / radius # angular distance
    lat1   = np.arcsin(np.sin(lat) * np.cos(d_R) + np.cos(lat) * \
                      np.sin(d_R) * np.cos(angle))
    lon1   = lon + np.arctan2(np.sin(angle) * np.sin(d_R) * np.cos(lat), \
                             np.cos(d_R) - np.sin(lat) * np.sin(lat1))
    lat    = np.rad2deg(lat1)
    lon    = np.rad2deg(lon1)
    return lon, lat



def newPositionRhumb(lon, lat, angle, distance):
    # Given the inputs (base lon, base lat, angle, distance) return
    # the lon, lat of the new position using Rhumb lines...
    # There are some caveats that I haven't covered here!!
    #   check: http://www.movable-type.co.uk/scripts/LatLong.html
    lon    = np.deg2rad(lon)
    lat    = np.deg2rad(lat)
    angle  = np.deg2rad(angle)
    radius = 6371315.0 # Mean earth radius in metres
    d_R    = distance / radius # angular distance
    lat1   = lat + d_R * np.cos(angle)
    dPhi   = np.log(np.tan(0.5*lat1 + 0.25*np.pi) / np.tan(0.5*lat + 0.25*np.pi))
    q      = (lat1 - lat) / dPhi
    dlon   = d_R * np.sin(angle) / q
    lon1   = (lon + dlon + np.pi)%(2. * np.pi) - np.pi
    lat    = np.rad2deg(lat1)
    lon    = np.rad2deg(lon1)
    return lon, lat


'''
def saveFig(path, figName, dpi=None, format=None):
    # Save a figure
    figName = path + figName
    plt.savefig(figName, dpi=dpi, format=format)
    print 'Figure', figName, 'saved...'
    return
'''


def gaussian_with_nans(u, sigma, mode='reflect'):
    '''
    http://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    '''
    assert np.ma.any(u.mask), 'u must be a masked array'
    mask = np.flatnonzero(u.mask)
    #u.flat[mask] = np.nan

    v = u.data.copy()
    v.flat[mask] = 0
    v[:] = nd.gaussian_filter(v, sigma=sigma, mode=mode)

    w = np.ones_like(u.data)
    w.flat[mask] = 0
    w[:] = nd.gaussian_filter(w, sigma=sigma, mode=mode)

    with np.errstate(invalid='ignore'):
        out = v / w

    anynans = np.isnan(out)
    if anynans.sum():
        out[anynans] = 0
        #print 'its w'
    #if np.isnan(v).sum():
        #print 'its v'
    #out = v / w
    #if np.isnan(out).sum():
        #print 'its out %s, %s' % (v[np.isnan(out)], w[np.isnan(out)])

    return out





def do_basemap(M, ax, par=True, mer=True, fcc='#D8D8BF'):
    lw = 0.5
    if np.logical_or(np.diff([M.lonmin, M.lonmax]) > 100,
                     np.diff([M.latmin, M.latmax]) > 100):
        stride = (30, 60)
        lw = 0.1
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 50,
                     np.diff([M.latmin, M.latmax]) > 50):
        stride = (10, 10)
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 30,
                       np.diff([M.latmin, M.latmax]) > 30):
        stride = (5, 5)
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 15,
                       np.diff([M.latmin, M.latmax]) > 15):
        stride = (3, 3)
    else:
        stride = (2, 2)
    if par: labels = [1, 0, 0, 0]
    else:   labels = [0, 0, 0, 0]
    M.drawparallels(np.arange(-720, 720., stride[0]), labels=labels,
                    linewidth=lw, size=8, ax=ax, zorder=12)
    if mer: labels = [0, 0, 0, 1]
    else:   labels = [0, 0, 0, 0]
    M.drawmeridians(np.arange(-720, 720., stride[1]), labels=labels,
                    linewidth=lw, size=8, ax=ax, zorder=12)
    #M.fillcontinents('#D6C6A9', ax=ax, zorder=11)
    M.fillcontinents(fcc, ax=ax, zorder=11)
    M.drawcoastlines(ax=ax, linewidth=lw, zorder=12)
    return










def get_KE(u, v, mask=None):
    if mask is not None:
        """ The ROMS mask uses 0s for the mask, whilst
        numpy mask uses 1s. So we convert ROMS to numpy
        style by subtracting 1 """
        mask = mask - 1
    ke = 0.5 * (np.power(u,2) + np.power(v,2))
    ke = MA.masked_array(ke, mask=mask)
    return ke



def enclose_box(lonmin, lonmax, latmin, latmax, lon, lat):
    '''
    Return i,j of corner points in roms grid for given
    unrotated lon, lat coords
    imin,imax,jmin,jmax = enclose_box(lonmin,lonmax,latmin,latmax,lon,lat)

    Note, warnings of out of grid should not be a problem, but maybe they are...
    '''
    if np.diff([lonmin, lonmax]) < 1.:
        lonmin -= 1.
        lonmax += 1.
    if np.diff([latmin, latmax]) < 1.:
        latmin -= 1.
        latmax += 1.
    Ai, Aj = nearest(lonmin, latmin, lon, lat)
    Bi, Bj = nearest(lonmax, latmax, lon, lat)
    Ci, Cj = nearest(lonmin, latmax, lon, lat)
    Di, Dj = nearest(lonmax, latmin, lon, lat)
    imin = np.min([Ai, Bi, Ci, Di])
    jmin = np.min([Aj, Bj, Cj, Dj])
    imax = np.max([Ai, Bi, Ci, Di])
    jmax = np.max([Aj, Bj, Cj, Dj])
    #print imin
    #while np.any(lon[jmin:jmax,imin:imax][:,0] > lonmin):
        #imin -= 1
        #if imin < 0:
            #print 'imin out of grid'
            #imin = 0
            #break
    ##print jmin,jmax,imin,imax
    #while np.any(lon[jmin:jmax,imin:imax][:,-1] < lonmax):
        #imax += 1
        #if imax > lon.shape[1]:
            #print 'imax out of grid'
            #imax = lon.shape[1]
            #break
    #while np.any(lat[jmin:jmax,imin:imax][0] > latmin):
        #jmin -= 1
        #if jmin < 0:
            #print 'jmin out of grid'
            #jmin = 0
            #break
    #while np.any(lat[jmin:jmax,imin:imax][-1] < latmax):
        #jmax += 1
        #if jmax > lon.shape[0]:
            #print 'jmax out of grid'
            #jmax = lon.shape[0]
            #break
    return imin, imax, jmin, jmax



def u2rho_2d(uu_in):
    '''
    Convert a 2D field at u points to a field at rho points
    Checked against Jeroen's u2rho.m
    '''
    def uu2ur(uu_in, Mp, Lp):
        L = Lp - 1
        Lm = L  - 1
        u_out = np.zeros((Mp, Lp))
        u_out[:, 1:L] = 0.5 * (uu_in[:, 0:Lm] + \
                               uu_in[:, 1:L])
        u_out[:, 0]   = u_out[:, 1]
        u_out[:, L]   = u_out[:, Lm]
        return (np.squeeze(u_out))
    # First check to see if has time dimension
    if uu_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = uu_in.shape
        u_out      = uu2ur(uu_in, Mshp, Lshp+1)
    else:
        # Has time dimension
        time, Mshp, Lshp = uu_in.shape
        u_out            = np.zeros((time, Mshp, Lshp+1))
        for t in np.arange(time):
            u_out[t] = uu2ur(uu_in[t], Mshp, Lshp+1)
    return u_out



def v2rho_2d(vv_in):
    # Convert a 2D field at v points to a field at rho points
    def vv2vr(vv_in, Mp, Lp):
        M             = Mp - 1
        Mm            = M  - 1
        v_out         = np.zeros((Mp, Lp))
        v_out[1:M, :] = 0.5 * (vv_in[0:Mm, :] + \
                               vv_in[1:M, :])
        v_out[0, :]   = v_out[1, :]
        v_out[M, :]   = v_out[Mm, :]
        return (np.squeeze(v_out))
    # First check to see if has time dimension
    if vv_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = vv_in.shape
        v_out      = vv2vr(vv_in, Mshp+1, Lshp)
    else:
        # Has time dimension
        time, Mshp, Lshp = vv_in.shape
        v_out            = np.zeros((time, Mshp+1, Lshp))
        for t in np.arange(time):
            v_out[t] = vv2vr(vv_in[t], Mshp+1, Lshp)
    return v_out



def u2rho_3d(uu_in):
    # Convert a 3D field at u points to a field at rho points
    # Calls u2rho_2d
    def levloop(uu_in):
        Nlevs, Mshp, Lshp = uu_in.shape
        u_out             = np.zeros((Nlevs, Mshp, Lshp+1))
        for Nlev in np.arange(Nlevs):
             u_out[Nlev] = u2rho_2d(uu_in[Nlev])
        return u_out
    # First check to see if has time dimension
    if uu_in.ndim < 4:
        # No time dimension
        u_out = levloop(uu_in)
    else:
        # Has time dimension
        time, Nlevs, Mshp, Lshp = uu_in.shape
        u_out                   = np.zeros((time, Nlevs, Mshp, Lshp+1))
        for t in np.arange(time):
            u_out[t] = levloop(uu_in[t])
    return u_out



def v2rho_3d(vv_in):
    # Convert a 3D field at v points to a field at rho points
    # Calls v2rho_2d
    def levloop(vv_in):
        Nlevs, Mshp, Lshp = vv_in.shape
        v_out             = np.zeros((Nlevs, Mshp+1, Lshp))
        for Nlev in range(Nlevs):
             v_out[Nlev] = v2rho_2d(vv_in[Nlev])
        return v_out
    # First check to see if has time dimension
    if vv_in.ndim < 4:
        # No time dimension
        v_out = levloop(vv_in)
    else:
        # Has time dimension
        time, Nlevs, Mshp, Lshp = vv_in.shape
        v_out                   = np.zeros((time, Nlevs, Mshp+1, Lshp))
        for t in np.arange(time):
            v_out[t] = levloop(vv_in[t])
    return v_out




def rho2u_2d(urho_in):
    # Convert a 2D field at rho points to a field at u points
    def uvr2uu(urho_in, Lp):
        L     = Lp - 1
        u_out = 0.5 * (urho_in[:, 0:L] + urho_in[:, 1:Lp])
        return np.squeeze(u_out)
    # First check to see if has time dimension
    if urho_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = urho_in.shape
        u_out  = uvr2uu(urho_in, Lshp)
    else:
        # Has time dimension
        time, Mshp, Lshp = urho_in.shape
        u_out            = np.zeros((time, Mshp, Lshp-1))
        for t in np.arange(time):
            u_out[t] = uvr2uu(urho_in[t], Lshp)
    return u_out




def rho2v_2d(vrho_in):
    # Convert a 2D field at rho points to a field at v points
    def uvr2vv(vrho_in, Mp):
        M     = Mp - 1
        v_out = 0.5 * (vrho_in[0:M, :] + vrho_in[1:Mp, :])
        return np.squeeze(v_out)
    # First check to see if has time dimension
    if vrho_in.ndim < 3:
        # No time dimension
        Mshp, Lshp = vrho_in.shape
        v_out  = uvr2vv(vrho_in, Mshp)
    else:
        # Has time dimension
        time, Mshp, Lshp = vrho_in.shape
        v_out            = np.zeros((time, Mshp-1, Lshp))
        for t in np.arange(time):
            v_out[t] = uvr2vv(vrho_in[t], Mshp)
    return v_out



def rho2u_3d(urho_in):
    # Convert a 3D field at rho points to a field at u points
    # Calls rho2u_2d
    def levloop(urho_in):
        Nlevs, Mshp, Lshp = urho_in.shape
        rho_out             = np.zeros((Nlevs, Mshp, Lshp-1))
        for Nlev in np.arange(Nlevs):
             rho_out[Nlev] = rho2u_2d(urho_in[Nlev])
        return rho_out
    # First check to see if has time dimension
    if urho_in.ndim < 4:
        # No time dimension
        rho_out = levloop(urho_in)
    else:
        # Has time dimension
        time, Nlevs, Mshp, Lshp = urho_in.shape
        rho_out                 = np.zeros((time, Nlevs, Mshp, Lshp+1))
        for t in np.arange(time):
            rho_out[t] = levloop(urho_in[t])
    return rho_out



def rho2v_3d(vrho_in):
    # Convert a 3D field at rho points to a field at v points
    # Calls rho2v_2d
    def levloop(vrho_in):
        Nlevs, Mshp, Lshp = vrho_in.shape
        rho_out             = np.zeros((Nlevs, Mshp-1, Lshp))
        for Nlev in np.arange(Nlevs):
             rho_out[Nlev] = rho2v_2d(vrho_in[Nlev])
        return rho_out
    # First check to see if has time dimension
    if vrho_in.ndim < 4:
        # No time dimension
        rho_out = levloop(vrho_in)
    else:
        # Has time dimension
        time, Nlevs, Mshp, Lshp = vrho_in.shape
        rho_out                 = np.zeros((time, Nlevs, Mshp+1, Lshp))
        for t in np.arange(time):
            rho_out[t] = levloop(vrho_in[t])
    return rho_out


def rho2w(rho_in,zeta=np.array(1)):
    '''
    Take 3d variable at (vertical) rho points
    and translate to w points.
    Keyword zeta to be used ONLY for zr
    '''
    if zeta.size==1:
        zeta = rho_in[0][np.newaxis]
    return np.concatenate(
        (zeta,
         0.5 * (rho_in[:-1] + rho_in[1:]),
         rho_in[-1][np.newaxis]),axis=0)



def psi2rho(var_psi):
    # Convert a psi field to rho points
    M, L    = var_psi.shape
    Mp      = M + 1
    Lp      = L + 1
    Mm      = M - 1
    Lm      = L - 1
    var_rho = np.zeros((Mp, Lp))
    var_rho[1:M, 1:L] = 0.25*(var_psi[0:Mm, 0:Lm] + var_psi[0:Mm, 1:L] + \
                              var_psi[1:M,  0:Lm] + var_psi[1:M,  1:L])
    var_rho[0, :]     = var_rho[1,  :]
    var_rho[M, :]     = var_rho[Mm, :]
    var_rho[:, 0]     = var_rho[:,  1]
    var_rho[:, L]     = var_rho[:, Lm]
    return var_rho



def uvp_mask(rho_mask):
    '''
    Get mask at u, v, psi points
    '''
    Mp, Lp = rho_mask.shape
    M = Mp - 1
    L = Lp - 1
    u_mask   = rho_mask[:,:L] * rho_mask[:,1:Lp]
    v_mask   = rho_mask[:M]   * rho_mask[1:Mp]
    psi_mask =   u_mask[:M]   *   u_mask[1:Mp]
    return u_mask, v_mask, psi_mask



def get_f_pm_pn(lon, lat):
    '''
    f,pm,pn = get_f_pm_pn(lon, lat)

    Returns f, pm and pn given 2d lon and lat fields
    '''
    Lp, Mp = lon.shape
    dx = np.zeros((Lp, Mp))
    dy = np.zeros((Lp, Mp))
    lonu = rho2u_2d(lon)
    latu = rho2u_2d(lat)
    lonv = rho2v_2d(lon)
    latv = rho2v_2d(lat)
    dx[:, 1:Mp-1] = distLonLat(lonu[:, 0:Mp-2], latu[:, 0:Mp-2], \
                               lonu[:, 1:Mp-1], latu[:, 1:Mp-1])[0]
    dx[:, 0]      = dx[:, 1]
    dx[:, Mp-1]   = dx[:, Mp-2]
    dy[1:Lp-1, :] = distLonLat(lonv[0:Lp-2, :], latv[0:Lp-2, :], \
                               lonv[1:Lp-1, :], latv[1:Lp-1, :])[0]
    dy[0, :]      = dy[1, :]
    dy[Lp-1, :]   = dy[Lp-2, :]
    pm = 1. / dx
    pn = 1. / dy
    f = 4. * np.pi * np.sin(np.deg2rad(lat)) / 86400.
    #aaaaaaaaaa
    return f, pm, pn



def vorticity(ubar, vbar, pm, pn):
    # Returns vorticity, can deal with time dimension
    # ubar (vbar) should be in u (v) points
    # ... use rho2u_2d/3d (rho2v_2d/3d)
    def vort(ubar, vbar, pm, pn, Mp, Lp):
        M      = Mp - 1
        L      = Lp - 1
        Lm     = L  - 1
        Mm     = M  - 1
        xi     = np.zeros((M, L))
        mn_p   = np.zeros((M, L))
        uom    = np.zeros((M, Lp))
        von    = np.zeros((Mp, L))
        uom    = 2 * ubar / (pm[:, 0:L] + pm[:, 1:Lp])
        von    = 2 * vbar / (pn[0:M, :] + pn[1:Mp, :])
        mn     = pm * pn
        mn_p   = (mn[0:M,  0:L]  + mn[0:M,  1:Lp] +
                  mn[1:Mp, 1:Lp] + mn[1:Mp, 0:L]) / 4
        vort   = mn_p * (von[:, 1:Lp] - von[:, 0:L] - \
                         uom[1:Mp, :] + uom[0:M, :])
        # At the moment vort is size L by M; convert to Lp by Mp
        vort   = psi2rho(vort)
        return vort
    Mp, Lp = pm.shape
    # Check to see if has time dimension
    if ubar.ndim < 3:
        # No time dimension
        xi = vort(ubar, vbar, pm, pn, Mp, Lp)
    else:
        # Has time dimension
        time = ubar.shape[0]
        xi   = np.zeros((time, Mp, Lp))
        for t in range(time):
            xi[t, :] = vort(ubar[t, :], vbar[t, :], pm, pn, Mp, Lp)
    return xi


def del_scalar(scalar, dx, dy):
    '''
    '''
    dSdx, dSdy = np.gradient(scalar, dx, dy)
    return dSdx + dSdy


def vorticity2(u, v, pm, pn, three_d=False):
    '''
    Returns vorticity calculated using 'gradient'
    Boundary condition has better extrapolation
      u and v at rho points
      three_d==True if 3-d velocity arrays
    '''
    #print '--- vorticity2 uses gradient function'
    def vort(u, v, dx ,dy):
        uy, _ = np.gradient(u, dy, dx)
        _, vx = np.gradient(v, dy, dx)
        xi = vx - uy
        return xi
    Mp, Lp = pm.shape
    if not three_d:
        xi = vort(u, v, 1./pm, 1./pn)
    else:
        # Has depth
        #print "WARNING, changed Jan 2012"
        depth = u.shape[0]
        xi = np.zeros((depth, Mp, Lp))
        for k in np.arange(depth):
            xi[k] = vort(u[k], v[k], 1./pm, 1./pn)
    return xi



def divergence2(u,v,pm,pn):
    '''
    Returns divergence calculated using 'gradient'
    Boundary condition has better extrapolation
    '''
    def div(u,v,dx,dy):
        uy, ux = np.gradient(u,dy,dx)
        vy, vx = np.gradient(v,dy,dx)
        div     = ux + vy
        return div
    Mp, Lp = pm.shape
    if u.ndim < 3:
        # No time dimension
        div = div(u,v,1./pm,1./pn)
    else:
        # Has time dimension
        time = u.shape[0]
        div   = np.zeros((time,Mp,Lp))
        for t in np.arange(time):
            div[t] = div(u[t],v[t],1./pm,1./pn)
    return div



def spa_grad_mag(var,pm,pn,direction=False):
    '''
    Returns magnitude of the spatial gradient of a variable
    If *direction* returns also the direction of the gradient
    '''
    grad = np.gradient(var,1./pm,1./pn)
    grad = np.hypot(grad[0],grad[1])
    if direction:
        direction = np.arctan(grad[0]/grad[1])
        return grad,direction
    return grad


    '''
    def lonlat2dxdy(lon,lat):
    '''
    #Convert 2d fields of lon, lat to
    #dx, dy in meters
    #NOTE: Careful, just an estimate...
    '''
    dlon = np.gradient(lon)[1]
    dlat = np.gradient(lat)[0]
    dlondlat = np.gradient(lon,lat)
    coslat = np.cos(np.deg2rad(lat)) * 111200.
    dx = dlon * coslat
    dy = dlat * coslat
    return dx,dy
    '''

def get_eddy_grad(uPuP,uPvP,vPvP,pm,pn):
    '''
    Returns the eddy gradient
    See Morrow etal 94 pg 2061
    '''
    def eddy_grad(pp1,pp2,dx,dy):
        '''
        with gradient, 1st return is a dy,
        2nd is a dx
        '''
        junk, xx = np.gradient(pp1,dx,dy)
        yy, junk = np.gradient(pp2,dx,dy)
        return xx, yy
    duPuPx, duPvPy = eddy_grad(uPuP,uPvP,1./pm,1./pn)
    duPvPx, dvPvPy = eddy_grad(uPvP,vPvP,1./pm,1./pn)
    duPuPx =np.ma.masked_where(np.isnan(duPuPx),duPuPx)
    duPvPy =np.ma.masked_where(np.isnan(duPvPy),duPvPy)
    duPvPx =np.ma.masked_where(np.isnan(duPvPx),duPvPx)
    dvPvPy =np.ma.masked_where(np.isnan(dvPvPy),dvPvPy)
    return duPuPx,duPvPy,dvPvPy,duPvPx



def get_eddy_force(uPuP,uPvP,vPvP,pm,pn):
    '''
    Returns the eddy force calculated using 'gradient'
      eddy force has zonal and meridional components
    See Morrow etal 94 pg 2061
    '''
    mask = np.ma.getmask(uPuP)
    duPuPx,duPvPy,dvPvPy,duPvPx = get_eddy_grad(uPuP,uPvP,vPvP,pm,pn)
    eddy_force_x = -duPuPx - duPvPy
    eddy_force_y = -dvPvPy - duPvPx
    return np.ma.masked_where(mask,eddy_force_x), \
           np.ma.masked_where(mask,eddy_force_y)



def getPmPnData(grdfile):
    nc   = netcdf.Dataset(grdfile, 'r')
    pm   = nc.variables['pm'][:, :]
    pn   = nc.variables['pn'][:, :]
    mask = nc.variables['mask_rho'][:, :]
    nc.close()
    dx = 1./pm; dy = 1./pn
    return pm, pn, dx, dy, mask



def getAttrs(file):
    # Get certain attributes from an nc file
    nc = netcdf.Dataset(file, 'r')
    attrList = nc.__dict__
    nwrt     = attrList['nwrt']   # No. of timesteps between writes to his
    ntimes   = attrList['ntimes']
    visc2    = attrList['visc2']
    dt       = attrList['dt']     # Timestep size in seconds
    nc.close()
    return nwrt, ntimes, visc2, dt



def get_time_mean(var):
    # Calculate the mean over a 3-D time series
    # Assumes var[time, j, i]
    time, jsize, isize = var.shape
    var_mean           = np.zeros((jsize, isize))
    for i in range(isize):
        for j in range(jsize):
            var_mean[j, i] = np.mean(var[:, j, i])
    return var_mean



def get_time_mean_and_dev(var):
    # Calculate the deviation from the mean over a 3-D time series
    # Assumes var[time, j, i]
    # Returns the deviation AND the mean
    # Requires: get_time_mean
    var_mean           = get_time_mean(var)
    time, jsize, isize = var.shape
    var_dev            = np.zeros((time, jsize, isize))
    for t in range(time):
        var_dev[t]     = np.squeeze(var[t] - var_mean)
    return var_mean, var_dev



def get_EKE(u, v, total_depth):
    # NOT TESTED YET!!!!!!!!!!!
    # Returns a time series of 2-D fields of instantaneous
    # EKE; i.e., 0.5(u_dev^2 + v_dev^2)*depth
    # Requires: get_time_mean_and_dev()
    ntimes, j, i  = u.shape
    # 1st get the time means and deviations for u and v
    u_mean, u_dev = get_time_mean_and_dev(u)
    v_mean, v_dev = get_time_mean_and_dev(v)
    # Calculate the EKE
    EKE = 0.5 * (np.power(u_dev,2) + np.power(v_dev,2)) * total_depth
    return EKE


def get_EKE_instantaneous(u, v, timestep, total_depth):
    # Returns 0.5(u_dev^2 + v_dev^2)*depth at a specified
    # 'timestep'
    EKE_instantaneous = get_EKE(u, v, total_depth)
    EKE_instantaneous = EKE_instantaneous[timestep]
    return EKE_instantaneous



def get_EKE_mean(u, v, dt, total_depth):
    # Returns two 2-D fields of EKE over a 2-D time series:
    #  1. EKE_mean is the total sum divided by T
    #  2. EKE_total is the total sum (NO DIVIDE, the same
    #     as EKE from get_EKE)
    # Requires: get_time_mean_and_dev()
    # By 'total_depth' we mean h + zeta, ie a time series
    # Get a time array of EKE
    ntimes, j, i  = u.shape
    EKE_total     = get_EKE(u, v, total_depth)
    # Do the integration over time
    EKE_total     = EKE_total.sum(axis=0) * dt
    EKE_mean      = (1. / ntimes) * EKE_total
    return EKE_mean, EKE_total



def get_EKE_total(u, v, dx, dy, dt, total_depth):
    # Returns two 1-D time series of the total integrated EKE
    # over a 2-D field time series
    #  1. EKE_total_area_averaged
    #  2. EKE_total
    # Requires: get_EKE
    EKE = get_EKE(u, v, total_depth)
    ntimes, j, i = EKE.shape
    EKE_total    = np.zeros((ntimes))
    for t in range(ntimes):
        EKE_total[t] = (EKE[t] * dx * dy).sum()
    area   = (dx * dy).sum()
    #volume = (dx * dy * total_depth).sum()
    EKE_total_area_averaged = (1. / area) * EKE_total
    return EKE_total_area_averaged, EKE_total




def get_total_depth_plus_ssh(h, zeta):
    # Returns a 2-D array of the total depth plus ssh
    # Hence it is a time-series (zeta is a time-series)
    total_depth_plus_ssh = zeta + h
    return total_depth_plus_ssh



def get_edge(mask, width=4):
    '''
    Return indices of broad mask edge for a 2d ROMS grid
    Input edge_ind = et.get_edge(mask)
    '''
    maskm = mask.copy()
    #maskm = nd.gaussian_filter(maskm,.1,0)
    edge = np.hypot(np.gradient(maskm, width, width)[0],
                    np.gradient(maskm, width, width)[1])
    indices = np.nonzero(edge)
    return indices[1], indices[0]



def grid_it1(z, x, y, xi, yi):
    # Returns interpolated data as ROMS grid
    # using griddata
    # Better to use grid_it2 for now...
    if x.ndim < 2:
        x, y = np.meshgrid(x, y)
    if xi.ndim < 2:
        xi, yi = np.meshgrid(xi, yi)
    x    = x.ravel()
    y    = y.ravel()
    z    = z.ravel()
    zi   = GD.griddata(x, y, z, xi, yi)
    return zi



def grid_it2(z, x, y, xi, yi):
    # Returns interpolated data as ROMS grid
    # using delaunay triangulation
    if x.ndim < 2:
        x, y = np.meshgrid(x, y)
    x      = x.ravel()
    y      = y.ravel()
    z      = z.ravel()
    tri    = tri.Triangulation(x, y)
    Interp = tri.nn_interpolator(z)
    zi     = Interp(xi, yi)
    return zi



def grid_it3(z, x, y, xi, yi):
    # Returns interpolated data as ROMS grid
    # using delaunay triangulation
    zi = interp(z,x,y,xi,yi,order=3)
    return zi


def grid_it_rbf(z, x, y, xi, yi):
    # Radial basis function for interpolation
    rbf = Rbf(x,y,z,function='cubic')
    zi  = rbf(xi,yi)
    return zi


def fillmask2(z,xi,yi,Tri=False,return_Tri=False):
    '''
    Fills mask of masked array with extrapolated values
      z = et.fillmask(z,xi,yi)
    Tri can be supplied from previous call to increase speed.
      z,tri = et.fillmask(z,xi,yi,return_Tri=True)
      z     = et.fillmask(z,xi,yi,Tri=tri)
    But mask must be unchanging!!
    Note: it's worth doing a figure to check the output!!
    '''
    assert np.ma.any(z.mask), 'z must be a masked array'
    xout,yout = xi.copy(),yi.copy()
    if xi.ndim<2: xi,yi = np.meshgrid(xi,yi)

    z_orig = z.copy()
    good = -np.ma.getmask(z) # minus to flip mask
    xi   = xi[good].ravel()
    yi   = yi[good].ravel()
    z    = z[good].ravel()
    if not Tri: Tri  = tri.Triangulation(xi,yi)
    Interp = delaunay.NNInterpolator(Tri,z)
    #Interp = Tri.nn_interpolator(z)
    z = Interp(xout,yout)
    z[good] = z_orig[good]

    print (np.isnan(z).sum())

    if return_Tri: return z,Tri
    return z


def fillmask_rbf(z,lon,lat,rbf=False,return_rbf=False):
    '''
    Fills mask of masked array with extrapolated values using Rbf
      z = et.fillmask(z,lon,lat)
    rbf can be supplied from previous call to increase speed.
      z,rbf = et.fillmask(z,xi,yi,return_rbf=True)
      z     = et.fillmask(z,xi,yi,rbf=rbf)
    But mask must be unchanging!!
    Note: check the output by setting 'debug=True' below
    '''
    assert np.ma.any(z.mask), 'z must be a masked array'
    mask = z.mask*1. # set 1 as land, 0 as sea
    # Gaussian filter degrades the edge (makes it wider)
    # 0.5 seems ok but can be played with..
    # Rbf will be faster with a smaller value
    edge = nd.gaussian_filter(mask,.7,0)
    # edge will be nonzero
    edge[edge==edge.max()] = 0.
    ind = np.nonzero(edge)

    mi    = mask[ind[0],ind[1]].ravel()
    good  = np.where(mi==0.)[0] # indices for sea points (ie, data points)
    bad   = np.where(mi==1.)[0] # indices for land points (ie, ie data)
    xx,yy = np.meshgrid(lon,lat)
    xi    = xx[ind[0],ind[1]].ravel()
    yi    = yy[ind[0],ind[1]].ravel()
    zi    =  z[ind[0],ind[1]].ravel()
    # Radial basis function for extrapolation
    rbf = Rbf(xi[good],yi[good],zi[good],function='cubic')
    zi  = rbf(xi[bad],yi[bad])
    # insert the values
    for m in np.arange(bad.size):
        i,j = nearest(xi[bad][m],yi[bad][m],xx,yy)
        z[j,i] = zi[m]
    if False: # True for debug
        from matplotlib import colors
        ln,lt = pcol_2dxy(xx,yy)
        '''plt.figure(1)
        plt.plot(lon[ind[1]],lat[ind[0]],'.k')
        plt.pcolormesh(ln,lt,edge)
        plt.figure(2)
        norm = colors.Normalize(vmin=zi.min(),vmax=zi.max())
        plt.pcolormesh(ln,lt,z)
        plt.clim(zi.min(),zi.max())
        plt.scatter(xi[bad],yi[bad],s=20,c=zi,norm=norm)
        plt.show()'''
        stop
    return z



def fillmask_kdtree(x, mask, k=4, weights=None, fill_value=9999):
    '''
    Fill missing values in an array with an average of nearest  
    neighbours.
      x    : data 2d-array
      mask : 2d array same size as 'x'; 0s land, 1s+ sea
    From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
    '''
    @njit(cache=True)
    def _get_igood_ibad(x, fill_value):
        '''
        Numba helper for fillmask_kdtree
        '''
        igood = vstack(where(x != fill_value)).T
        ibad = vstack(where(x == fill_value)).T
        return igood, ibad

    assert x.ndim == 2, '`x` must be a 2D array.'
    assert mask.ndim == 2, '`mask` must be a 2D array.'
    mask = mask.astype(bool)
    if (mask == True).all():
        return x
    x[mask == False] = fill_value
    if weights is not None:
        dist, iquery, igood, ibad = weights
    else:
        # Create (i, j) point arrays for good and bad data
        # Bad data is marked by the fill_value, good data elsewhere
        igood, ibad = _get_igood_ibad(x, fill_value)
        # create a tree for the bad points, the points to be filled
        tree = sp.cKDTree(igood)
        # get the four closest points to the bad points
        # here, distance is squared
        dist, iquery = tree.query(ibad, k=k, p=2)

    # Create a normalised weight, the nearest points are weighted as 1
    # Points greater than one are then set to zero
    weight = dist / (dist.min(axis=1)[:, np.newaxis])
    weight[weight > 1.] = 0

    '''Multiply the queried good points by the weight, selecting only  
       the nearest points.  Divide by the number of nearest points
       to get average.'''
    xfill = x[igood[:, 0][iquery], igood[:, 1][iquery]]
    xfill *= weight
    xfill /= weight.sum(axis=1)[:, np.newaxis]

    # Place average of nearest good points, xfill, into bad point locations
    x[ibad[:, 0], ibad[:, 1]] = xfill.sum(axis=1)
    if weights is not None:
        return x
    else:
        return x, (dist, iquery, igood, ibad)



def fillmask_fast_kdtree(x, mask, i, j, inds, k=4, weights=None, fill_value=9999):
    '''
    Fill missing values in an array with an average of nearest  
    neighbours.
    From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
    
    i, j = unravel_index(arange(x.size), x.shape)
    inds = np.array([i.ravel(), j.ravel()]).T
    
    '''
    assert x.ndim == 2, 'x must be a 2D array.'
    if np.alltrue(mask == 1):
        return x
    x[mask == 0] = fill_value
    if weights is not None:
        dist, iquery, igood, ibad = weights
    else:
        # Create (i, j) point arrays for good and bad data
        # Bad data is marked by the fill_value, good data elsewhere
        igood = np.take(
            inds, (x.ravel() != fill_value).nonzero()[0], axis=0)
        ibad = np.take(
            inds, (x.ravel() == fill_value).nonzero()[0], axis=0)

        # create a tree for the bad points, the points to be filled
        tree = sp.cKDTree(igood)
        # get the four closest points to the bad points
        # here, distance is squared
        dist, iquery = tree.query(ibad, k=k, p=2)

    # Create a normalised weight, the nearest points are weighted as 1
    # Points greater than one are then set to zero
    #weight = dist / (dist.min(axis=1)[:, np.newaxis] * np.ones_like(dist))
    weight = dist / (dist.min(axis=1)[:, np.newaxis])
    weight[weight > 1.] = 0

    '''multiply the queried good points by the weight, selecting only  
       the nearest points.  Divide by the number of nearest points
       to get average.'''
    xfill = x[igood[:, 0][iquery], igood[:, 1][iquery]]
    xfill *= weight
    xfill /= weight.sum(axis=1)[:, np.newaxis]

    # Place average of nearest good points, xfill, into bad point locations
    x[ibad[:, 0], ibad[:, 1]] = xfill.sum(axis=1)
    if weights is not None:
        return x
    else:
        return x, (dist, iquery, igood, ibad)



def fillmask_python(X,power=2.,radius=0.):
    '''
    power = 2.  # weighting power
    radius = 0. # distance cut-off radius (0 = all pixels, no cut-off)
    '''
    print ("Use fillmask, it's faster cos uses Fortran")
    assert np.ma.any(X.mask), 'X must be a masked array'
    #cdef int k
    Y     =  X # output array
    XX    =  X.ravel()
    ix    =  X.mask
    rn,cn =  np.where(ix)    # row,col of nans
    ix    = ~X.mask
    r,c   =  np.where(ix)    # row,col of non-nans
    ind   =  np.flatnonzero(ix)  # index of non-nans

    # Break distance-finding loops into with cut-off and without cut-off
    # versions. The cutoff conditional statement adds time
    # if cut-off values near the max pixel distance are used.
    if radius: #distance cut-off loop
        #aaaa ## 'radius doesnt work for some reason'
        radius = radius**2.             # (Urs:allows first step without SQRT())
        for k in np.arange(rn.size):
            D = (rn[k]-r)**2.+(cn[k]-c)**2.
            #print 'D',D
            #print 'D',D[0]
            Dd = np.nonzero(D<radius)
            if np.sum(Dd)>0.:
                #print 'D',D
                #print 'Dd',Dd
                #print 'D[Dd]',D[Dd]
                D = 1./np.sqrt(D[Dd])**power
                #print 'D',D
                Y[rn[k],cn[k]] = np.sum(XX[ind[Dd]]*D) / np.sum(D)
    else: #no distance cut-off loop
        for k in np.arange(rn.size):
            D = 1./(np.sqrt((rn[k]-r)**2.+(cn[k]-c)**2.))**power # (Urs:compute once only)
            Y[rn[k],cn[k]] = np.sum(XX[ind]*D) / np.sum(D)
    return Y.data


def fillmask(x, power=2., radius=0.):
    '''
    This is the f2py version.
    power = 2.  # weighting power
    radius = 0. # distance cut-off radius (0 = all pixels, no cut-off)
    
    Returns
      y : filled array
      x : original array if all values are masked
    '''
    assert np.ma.any(x.mask), 'x must be a masked array'

    import fillnans as fn

    xx = x.ravel()
    ix = x.mask
    #ix = np.asfortranarray(ix)
    rn, cn = np.where(ix)      # row, col of nans
    ix = ~x.mask
    #ix = np.asfortranarray(ix)
    r, c = np.where(ix)      # row, col of non-nans
    ind = np.flatnonzero(ix)  # indices of non-nans
    lx = ind.size
    lrn = rn.size   # length of rn, cn
    lr = r.size    # length of r, c
    m,n = x.shape

    rn = np.asfortranarray([rn])
    cn = np.asfortranarray([cn])
    r = np.asfortranarray([r])
    c = np.asfortranarray([c])
    ind = np.asfortranarray([ind])

    rn += 1
    cn += 1
    r += 1
    c += 1
    ind += 1

    #before = time.time()
    #print '-------', xx.min(), xx.max(), xx.mean()
    try:
        y = fn.fillnans(xx, rn, cn, r, c, ind, m, n, power, radius, lrn, lr, lx)
    except Exception:# all values masked?
        assert x.size == x.mask.sum(), 'Check input array'
        return x
    #print 'fillmask time', time.time() - before
    #print '++++++++', y.min(), y.max(), y.mean()
    return y


def inpaint_nans(z,xi,yi):
    '''
    # Fills nans with extrapolated values
    z = et.inpaint_nans(z,xi,yi)
    # Note: it's worth doing a figure to check the output!!
    '''
    if np.sum(np.isnan(z))<1:
            print ('Warning: No nans detected!')
            return z
    if xi.ndim<2:
        xi,yi = np.meshgrid(xi,yi)
        xout,yout = xi.copy(),yi.copy()
    else:
        xout,yout = xi.copy(),yi.copy()
    z_orig = z.copy()
    good = np.isfinite(z)
    xi   = xi[good].ravel()
    yi   = yi[good].ravel()
    z    = z[good].ravel()
    tri  = tri.Triangulation(xi,yi)
    Interp = tri.nn_interpolator(z)
    z = Interp(xout,yout)
    z[good] = z_orig[good]
    return z


def fill_w_griddata(z,xi,yi):
    '''
    # Fills nans with extrapolated values
    z = et.inpaint_nans(z,xi,yi)
    # Note: it's worth doing a figure to check the output!!
    '''
    if np.sum(np.isnan(z))<1:
            print ('Warning: No nans detected!')
            return z
    if xi.ndim<2:
        xi,yi = np.meshgrid(xi,yi)
        xout,yout = xi.copy(),yi.copy()
    else:
        xout,yout = xi.copy(),yi.copy()
    z_orig = z.copy()
    good = np.isfinite(z)
    xi   = xi[good].ravel()
    yi   = yi[good].ravel()
    z    = z[good].ravel()
    #tri  = delaunay.Triangulation(xi,yi)
    #Interp = tri.nn_interpolator(z)
    #z = Interp(xout,yout)
    goodpoints = np.array([xi,yi]).T
    z = griddata(goodpoints,z,(xout,yout),method='nearest')
    z[good] = z_orig[good]
    return z



def inpaint_nans_BAD(z, xi, yi):
    # Fills nans with extrapolated values
    # Note: it's worth doing a figure to check the output!!
    if np.sum(np.isnan(z))<1:
            print ('Warning: No nans detected!')
            return z
    if xi.ndim<2:
        xi, yi = np.meshgrid(xi, yi)
        xout, yout = xi.copy(), yi.copy()
    else:
        xout, yout = xi.copy(), yi.copy()
    z_orig = z.copy()
    good = np.isfinite(z)
    xi   = xi[good].ravel()
    yi   = yi[good].ravel()
    z    = z[good].ravel()
    tri  = tri.Triangulation(xi, yi)
    Interp = tri.nn_interpolator(z)
    z    = Interp(xout, yout)
    z[good] = z_orig[good]
    return z





def grid_it3(z, x, y, xi, yi, fillval=None):
    # Returns interpolated data as ROMS grid
    # using basemap interp
    # See http://matplotlib.sourceforge.net/matplotlib.toolkits.basemap.basemap.html
    #   for more on how to improve this, re. the masking
    if xi.ndim < 2:
        xi, yi = np.meshgrid(xi, yi)
    if fillval!=None:
        z = MA.masked_equal(z, fillval)
        while np.any(zi1.data==fillval):
            #print 'aaaaaaaaaa'
            zi2 = interp(zi1, x, y,  xi, yi, order=0)
        zi1[zi1.mask] = zi2[zi1.mask]
        #plt.pcolormesh(zi1)
        #plt.colorbar()
        #plt.show()
        return zi1.data
    else:
        zi1 = interp(z, x, y, xi, yi, order=1)
        return zi1


def sakov_nn(x, y, z):
    if x.ndim < 2:
        x, y = np.meshgrid(x, y)
    x      = x.ravel()
    y      = y.ravel()
    z      = z.ravel()
    np.tofile('aaaa', x)
    F.close()
    zi     = os.system('nnbathy -i [x, y, z] -P alg=nn')
    return zi


def grid_it4(z, x, y, xi, yi):
    # This one copied from Rob, very similar (maybe
    #    even the same as grid_it2 above
    if x.ndim < 2:
        x, y = np.meshgrid(x, y)
    x  = x.ravel()
    y  = y.ravel()
    z  = z.ravel()
    zi = tri.Triangulation(x, y).nn_extrapolator(z)(xi, yi)
    return zi



def getSurfGeostrVel(f, zeta, pm, pn, umask, vmask):
    '''
    Returns u and v geostrophic velocity at
    surface from ROMS variables f, zeta, pm, pn...
    Note: output in rho points
    Adapted from IRD surf_geostr_vel.m function
    by Evan Mason
    Changed to gv2, 14 May 07...
    '''
    def gv2(f, zeta, pm, pn, umask, vmask): # Pierrick's version
        gof = 9.81 / f
        ugv = -gof * v2rho_2d(vmask * (zeta[1:] - zeta[:-1]) \
                   * 0.5 * (pn[1:] + pn[:-1]))
        vgv =  gof * u2rho_2d(umask * (zeta[:, 1:] - zeta[:, :-1]) \
                   * 0.5 * (pm[:, 1:] + pm[:, :-1]))
        return ugv, vgv
    if zeta.ndim < 3:
        # No time dimension
        ugv, vgv = gv2(f, zeta, pm, pn, umask, vmask)
    else:
        # Has time dimension, so loop through...
        time, Lshp, Mshp = zeta.shape
        ugv = np.empty((0, Lshp, Mshp))
        vgv = np.empty((0, Lshp, Mshp))
        for t in np.arange(time):
            ugv_tmp, vgv_tmp = gv2(f, zeta[t, :], pm, pn, umask, vmask)
            ugv = np.ma.concatenate((ugv, ugv_tmp[np.newaxis, :]), axis=0)
            vgv = np.ma.concatenate((vgv, vgv_tmp[np.newaxis, :]), axis=0)
    return ugv, vgv


def tridim(twod_field, num_levels):
    ''''
    Make a 3d variable out of a 2d.  Useful for converting mask_rho
    into 3d, for example.
    Input the field and the number of vertical levels
    '''
    threed_field = twod_field.copy()
    twod_field   = twod_field[np.newaxis]
    threed_field = threed_field[np.newaxis]
    for lev in np.arange(num_levels-1):
        threed_field = np.concatenate((threed_field, twod_field), axis=0)
    return threed_field



def interp_nans(var):
    ###MAKE SURE IT HAS SOME NANS###
    #var[var>1000.]    = np.nan # Put in some nans...
    # Taken from Rob Hetland
    # First, extrapolate adjacent (boundary) cells
    mask              = np.where(~np.isnan(var), 1., 0.)
    varb              = np.nan*np.zeros_like(var)
    varb[1:-1, 1:-1]  = np.nansum([var[:-2,:-2], var[2:,:-2], \
                             var[:-2,2:], var[2:,2:]], axis=0)
    maskb             = np.zeros_like(var)
    maskb[1:-1, 1:-1] = np.nansum([mask[:-2,:-2], mask[2:,:-2], \
                                mask[:-2,2:], mask[2:,2:]], axis=0)
    varb              = varb/maskb
    var               = np.where((maskb>0) & (np.isnan(var)), varb, var)
    # Next, extrapolate over the entire domain
    igood = np.where(~np.isnan(var))
    umin  = var[igood].min()
    umax  = var[igood].max()
    i, j  = np.mgrid[:var.shape[0], :var.shape[1]]
    tri   = tri.Triangulation(i[igood], j[igood])
    out = tri.nn_extrapolator(var[igood])(i, j).clip(umin, umax)
    print ('fff')
    return out


def extrapolate_maskxxxxxxxxxx(a, mask=None):
    # Taken from Rob Hetland
    if mask is None and not isinstance(a, MA.MaskedArray): 
        return a
    if mask is None:
        mask = a.mask
    else:
        if isinstance(a, MA.MaskedArray):
            mask = mask | a.mask
    a = a[:]    # make a copy of array a
    jj, ii = np.indices(a.shape)
    igood = ii[mask==False] #ii[~mask]
    jgood = jj[mask==False] #jj[~mask]
    ibad = ii[mask==True]
    jbad = jj[mask==True]
    tri = tri.Triangulation(igood, jgood)
    # interpolate from the good points (mask == 1)
    interp = tri.nn_extrapolator(a[mask==False])
    # to the bad points (mask == 0)
    a[mask] = interp(ibad, jbad)
    return a


def multi_interp(z, x, y, lon_grd, lat_grd):
    ############ DOES NOT WORK AT PRESENT ##########
    ### Iterates over 2d grid doing interp
    ### Grid gets progressively finer, smaller
    # max and min prevent grid getting too small
    lonmax = lon_grd.max() + 0.5
    lonmin = lon_grd.min() - 0.5
    latmax = lat_grd.max() + 0.5
    latmin = lat_grd.min() - 0.5
    # Conditions for 'while' statement below...
    diff_grd  = max(np.mean(np.diff(lon_grd)), np.mean(np.diff(lat_grd)))
    diff_data = max(np.mean(np.diff(x)), np.mean(np.diff(y)))
    # Prepare vars
    xi = lon_grd
    yi = lat_grd
    # Loop...
    while diff_data >= diff_grd:
        xi = np.linspace(x[0] + np.sqrt(0.25**2/2), x[-1] - np.sqrt(0.25**2/2), len(x)+1)
        yi = np.linspace(y[0] + np.sqrt(0.25**2/2), y[-1] - np.sqrt(0.25**2/2), len(y)+1)
        if xi[0] > lonmin or xi[-1] < lonmax or \
           yi[0] > latmin or yi[-1] < latmax:
            diff_data = diff_grd - 1000000 # Break the 'while'
        else:
            diff_data = np.mean(np.diff(xi))
            z = grid_it1(z, x, y, xi, yi)
            x = xi.copy()
            y = yi.copy()
    return z, x, y



def inpaint_nans_old(zin, xin, yin):
    # Fill NaN values with extrapolated data
    # Uses natural neighbour...
    # VERY SLOW!!!!
    z_orig  = zin.copy()
    if xin.ndim < 2:
        xout, yout = np.meshgrid(xin, yin)
    else:
        xout, yout = xin, yin
    D       = np.isfinite(zin)
    zin     = zin[D].ravel()
    xin     = xout[D].ravel()
    yin     = yout[D].ravel()
    zout    = GD.griddata(xin, yin, zin, xout, yout)
    zout[D] = z_orig[D]
    return zout




def get_LonLat_subset(data,lon,lat,lonmin,lonmax,latmin,latmax):
    '''
    Returns data_out, lon_out and lat_out which
    are a subset of the 2d field, data.
    lon and lat should be monotonic increasing vectors
    i.e., a regular grid
    '''
    if lon.ndim!=1: lon = lon[0]
    if lat.ndim!=1: lat = lat[:,0]
    lon_ind = np.logical_and(lon>lonmin,lon<lonmax)
    lat_ind = np.logical_and(lat>latmin,lat<latmax)
    lon_out = lon[lon_ind]
    lat_out = lat[lat_ind]
    data_out = np.squeeze(data[lat_ind][:, lon_ind])
    return data_out,lon_out,lat_out



def rotateVectors(u_in, v_in, angle):
    ''''
    Rotates velocity vectors a la
    Roms_tools...
    'angle' is from nc file and should be radians
    '''
    cosa = np.cos(angle)
    sina = np.sin(angle)
    u_out = (u_in * cosa) + (v_in * sina)
    v_out = (v_in * cosa) - (u_in * sina)
    return u_out, v_out
    
    
    
    
    
def scoord2z(h, theta_s, theta_b, hc, Nlev, point_type, scoord, zeta=0., alpha=0., beta=1., message=True):
    """
    z = scoord2z(h, theta_s, theta_b, hc, N, point_type, scoord, zeta)
    scoord2z finds z at either rho or w points (positive up, zero at rest surface) 
      h          = array of depths (e.g., from grd file)
      theta_b    = surface/bottom focusing parameter
      theta_s    = strength of focusing parameter
      hc         = critical depth
      N          = number of vertical rho-points
      point_type = 'r' or 'w'
      scoord     = 2:new scoord 2008, 1:new scoord 2006, or 0 for old Song scoord
      zeta       = sea surface height
      message    = set to False if don't want message
    """
    def CSF(sc,theta_s,theta_b):
        '''
        Allows use of theta_b > 0 (July 2009)
        '''
        if theta_s > 0.:
            csrf = (1. - np.cosh(theta_s * sc)) / (np.cosh(theta_s) - 1.)
        else:
            csrf = -sc ** 2
        sc1 = csrf + 1.
        if theta_b > 0.:
            Cs = (np.exp(theta_b * sc1) - 1.) / (np.exp(theta_b) - 1.) - 1.
        else:
            Cs = csrf
        return Cs

    new_scoord = 'Using new s-coord (2006)'
    cff1 = 1. / np.sinh(theta_s)
    cff2 = 0.5 / np.tanh(0.5 * theta_s)
    sc_w = np.arange(-1, 1. / Nlev, 1. / Nlev, dtype='d')
    sc_r = 0.5 * (sc_w[1:] + sc_w[:-1])
    
    if 'w' in point_type:
        sc = sc_w
        Nlev += 1 # add a level
    else:
        sc = sc_r
    Cs = ((1. - theta_b) * cff1 * np.sinh(theta_s * sc) +
           theta_b * (cff2 * np.tanh(theta_s * (sc + 0.5)) - 0.5))
    z = np.empty((Nlev,) + h.shape, dtype='d')
    if 'new2008' in scoord:   # new s-coord is Sasha's 2008
        new_scoord = 'Using new s-coord (2008)'
        Cs = CSF(sc,theta_s,theta_b)

    if scoord in ('new2006', 'new2008'):   # new s-coord is Sasha's...
        if message:
            print (new_scoord)
        ds   = 1. / Nlev
        hinv = 1. / (h + hc)
        cff_r   = hc * sc # Bug fix 5Sep08 CORRECT
        for k in np.arange(Nlev) + 1:
            cff1_r  = Cs[k - 1]
            z[k - 1] = zeta + (zeta + h) * (cff_r[k - 1] + cff1_r * h) * hinv
    
    elif scoord in 'old1994':
        if message: print ('Using old s-coord (1994)')
        hinv = 1. / h
        cff  = hc * (sc - Cs)
        cff1 = Cs
        cff2 = sc + 1
        for k in np.arange(Nlev) + 1:
            z0      = cff[k-1] + cff1[k-1] * h
            z[k-1, :] = z0 + zeta * (1. + z0 * hinv)
    
    else:
        # You really don't want to end up here!!!
        print ('Warning!! You have not set scoord in romstools_param, or elsewhere...')
        exit()
    
    return z.squeeze()



def calcEKE(u_prime, v_prime, mask=None):
    # Next calculate the seasonal EKE
    # ...average the square of the primes
    u_prime = np.power(u_prime, 2)
    v_prime = np.power(v_prime, 2)
    EKE = 0.5 * (u_prime + v_prime)
    EKE = np.ma.masked_where(mask == False, EKE)
    return EKE




def mon2seas(monthly):
    '''
    Take any monthly array [12,:,:]
    and make it seasonal [5,:,:] (last one is annual)
    '''
    seasonal = np.zeros((5, monthly.shape[1], monthly.shape[2]))
    season_ind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    cff = np.array([0, 0, 0, 0, 0])
    for ind in np.arange(0, 12):
        seasonal[season_ind[ind]] += monthly[ind]
        seasonal[4] += monthly[ind]
        cff[season_ind[ind]] += 1.
        cff[4] += 1.
    for ind in np.arange(0, 5):
        seasonal[ind] /= cff[ind]
    return seasonal


def vec_cor(u1, v1, u2, v2):
    '''
    This program does a vector correlation analysis as per the paper
    by Breaker et al JGR May 2003...
    0 means no correlation
    2 means perfect correlation
    '''
    su1u1 = np.cov(u1, u1)
    su1u2 = np.cov(u1, u2)
    su1v1 = np.cov(u1, v1)
    su1v2 = np.cov(u1, v2)
    su2u2 = np.cov(u2, u2)
    su2v2 = np.cov(u2, v2)
    sv1v1 = np.cov(v1, v1)
    sv1v2 = np.cov(v1, v2)
    sv1u2 = np.cov(v1, u2)
    sv2v2 = np.cov(v2, v2)
    su1u1 = su1u1[1, 0]
    su1u2 = su1u2[1, 0]
    su1v1 = su1v1[1, 0]
    su1v2 = su1v2[1, 0]
    su2u2 = su2u2[1, 0]
    su2v2 = su2v2[1, 0]
    sv1v1 = sv1v1[1, 0]
    sv1v2 = sv1v2[1, 0]
    sv1u2 = sv1u2[1, 0]
    sv2v2 = sv2v2[1, 0]
    f1 = (su2u2 * sv1v2**2) + (sv2v2 * sv1u2**2)
    f2 = (su2u2 * su1v2**2) + (sv2v2 * su1u2**2)
    f3 = 2. * (su1v1 * su1v2 * sv1u2 * su2v2)
    f4 = 2. * (su1v1 * su1u2 * sv1v2 * su2v2)
    f5 = 2. * (su1u1 * sv1u2 * sv1v2 * su2v2)
    f6 = 2. * (sv1v1 * su1u2 * su1v2 * su2v2)
    f7 = 2. * (su2u2 * su1v1 * su1v2 * sv1v2)
    f8 = 2. * (sv2v2 * su1v1 * su1u2 * sv1u2)
    f = (su1u1 * f1) + (sv1v1 * f2) + f3 + f4 - f5 - f6 - f7 - f8
    g1          = (su1u1 * sv1v1) - su1v1**2
    g2          = (su2u2 * sv2v2) - su2v2**2
    g           = g1 * g2
    rho_squared = f / g
    #print 'rho squared = ', rho_squared
    return rho_squared


def vec_subsamp(vec, no_of_points):
    """
    Simple subsample from vec; starts at 0
    continues to end
    """
    subsamp = vec[np.r_[0:len(vec):no_of_points]]
    return subsamp


def nearest1d(pt,vec):
    '''
    Return index of value in a vector (vec)
    nearest to value (pt)
    index = nearest1d(point,vector)
    '''
    ind = np.argmin(np.abs(vec-pt))
    return ind



def nearest(lon_pt, lat_pt, lon2d, lat2d, four=False, test=False):
    """
    Return the nearest i, j point to a given lon, lat point
    in a ROMS grid
        i,j = nearest(lon_pt,lat_pt,lon2d,lat2d,four=False,test=False)
    If four is not None then the four nearest (i, j)
    points surrounding the point of interest are returned
    To test with a figure put test=True
    """
    def test_it(i, j, lon_pt, lat_pt, lon2d, lat2d):
        print ('---Testing the locations...')
        i1, i2, i3, i4 = i
        j1, j2, j3, j4 = j
        '''plt.figure(1)
        plt.plot(lon2d, lat2d, 'ro')
        plt.text(lon_pt, lat_pt, '0')
        plt.axis('image')
        plt.text(lon2d[j1, i1], lat2d[j1, i1], 'A')
        plt.text(lon2d[j2, i2], lat2d[j2, i2], 'B')
        plt.text(lon2d[j3, i3], lat2d[j3, i3], 'C')
        plt.text(lon2d[j4, i4], lat2d[j4, i4], 'D')
        plt.title('Is the 0 in the middle?')
        plt.show()'''
        return
    def four_points(i1, j1, lon_pt, lat_pt, lon2d, lat2d):
        iii   = np.array([1, -1, -1,  1])
        jjj   = np.array([1,  1, -1, -1])
        big_dist = 1e99 # arbitrary large distance
        for ii, jj in zip(iii, jjj):
            dist2 = np.hypot(lon2d[j1 + jj, i1 + ii] - lon_pt,
                             lat2d[j1 + jj, i1 + ii] - lat_pt)
            if dist2<big_dist:
                big_dist = dist2
                i2, j2 = i1+ii, j1+jj
        if i2>i1: i3 = i2; i4 = i1
        else:     i3 = i2; i4 = i1
        if j2>j1: j3 = j1; j4 = j2
        else:     j3 = j1; j4 = j2
        return np.array([i1, i2, i3, i4]), \
               np.array([j1, j2, j3, j4])
    # Main...
    d    = np.hypot(lon2d-lon_pt, lat2d-lat_pt)
    j, i = np.unravel_index(d.argmin(), d.shape)
    dist = d[j,i] # j,i not i,j
    
    if four is True:
        #print '---Getting 4 surrounding points...'
        i, j = four_points(i, j, lon_pt, lat_pt, lon2d, lat2d)
        if test is True: test_it(i, j, lon_pt, lat_pt, lon2d, lat2d)
        return i, j
    else:
        return i, j




def bilinear_interp(data, xin, yin):
    '''
    Simple bilinear interpolation
    See http://en.wikipedia.org/wiki/Bilinear_interpolation
    ONLY useful for data/xin/yin.shape = (xxx, 2, 2)
    xxx permits a time dimension
    APPEARS TO BE BOLLOCKS
    '''
    def binterp(data, x, y):
        b1  = data[1, 0]
        b2  = data[1, 1] - b1
        b3  = data[0, 0] - b1
        b4  = b1 - data[1, 1] - data[0, 0]- data[0, 1]
        out = b1 + (b2*x) + (b3*y) + (b4*x*y)
        return out
    x = np.average(np.diff(xin, axis=1))
    y = np.average(np.diff(yin, axis=0))
    if data.ndim==3:
        # Has time dimension
        time = data.shape[0]
        out  = np.zeros((time))
        for t in range(time):
            out[t] = binterp(data[t], x, y)
    else:
        # No time dimension
        out = binterp(data, x, y)
    return out



def get_island_perim(mask_rho, i, j, fig=False):
    '''
    Returns i, j of sea points surrounding an island within a ROMS
    mask domain
    Input: mask_rho: ROMS mask rho
           i, j: Should be somewhere within the island
    '''
    i = np.array(i)
    j = np.array(j)
    # Iterate north until sea point
    while mask_rho[j,i]==0:
        j += 1
    # Set initial perimeter location
    j0 = j.copy()
    i0 = i.copy()
    # Initial array
    jj = np.array([0])
    ii = np.array([0])
    # Clockwise additions
    sumj  = [1, 1, 0, -1, -1, -1,  0,  1, 1]
    sumi  = [0, 1, 1,  1,  0, -1, -1, -1, 0]
    cnt   = 0
    loop  = True
    pass1 = False
    start = True
    # Iterate clockwise
    while loop is True:
        if start==True:
            start = False
            jj[0] = j0
            ii[0] = i0
        if cnt==8: cnt = 0
        if mask_rho[j+sumj[cnt], i+sumi[cnt]]==1:
            if mask_rho[j+sumj[cnt+1], i+sumi[cnt+1]]==0:
                j += sumj[cnt]
                i += sumi[cnt]
                jj = np.append(jj, j)
                ii = np.append(ii, i)
                if np.all([j0==jj[-1], i0==ii[-1]]):
                    if pass1 is True:
                        loop = False
                pass1 = True
        cnt += 1
    
    if fig: # DEBUGGING
        '''plt.figure(1)
        plt.subplot(121)
        plt.pcolor(mask_rho)
        plt.axis('image')
        plt.grid()
        mask_rho[jj[:-1], ii[:-1]] = 0.5
        plt.subplot(122)
        plt.pcolor(mask_rho)
        plt.axis('image')
        plt.grid()
        plt.show()'''
    
    return ii[:-1], jj[:-1]



def temppot0(S,T,P):
    """Compute potential temperature relative to surface

    Usage: temppot0(S, T, P)

    Input:
        S = Salinity,                [PSS-78]
        T = Temperature,             [°C]
        P = Pressure,                [dbar]
 
    Output:
        Potential temperature,       [°C]

    Algorithm: Bryden 1973

    Note: Due to different algorithms,
        temppot0(S, T, P) != tempot(S, T, P, Pref=0)
  
    """

    P *= 0.1  # Conversion from dbar

    a0 =  3.6504e-4
    a1 =  8.3198e-5
    a2 = -5.4065e-7
    a3 =  4.0274e-9

    b0 =  1.7439e-5
    b1 = -2.9778e-7

    c0 =  8.9309e-7
    c1 = -3.1628e-8
    c2 =  2.1987e-10

    d0 =  4.1057e-9

    e0 = -1.6056e-10
    e1 =  5.0484e-12

    S0 = S - 35.0

    return  T - (a0 + (a1 + (a2 + a3*T)*T)*T)*P  \
              - (b0 + b1*T)*P*S0                 \
              - (c0 + (c1 + c2*T)*T)*P*P         \
              + d0*S0*P*P                        \
              - (e0 + e1*T)*P*P*P



def rho_eos(Tt, Ts, z_r=None, z_w=None, g=None, rho0=None):
    '''
    Computes density via Equation Of State (EOS) for seawater.
    If so prescribed, non-linear EOS of Jackett and McDougall (1995)
    is used.

    Tt potential temperature [deg Celsius].
    Ts salinity [PSU].
    Tz pressure/depth, [depth in meters and negative].

    K0, K1 and K2 are the pressure polynomial coefficients for secant
    bulk modulus, so that

               bulk = K0 - K1 * z + K2 * z**2 ;

    while rho1 is sea-water density [kg/m^3] at standard pressure
    of 1 Atm, so that the density anomaly at in-situ pressure is

               rho = rho1 / (1 + z / bulk) - 1000

    If so prescribed, it also computes the Brunt-Vaisala frequency
    [1/s^2] at horizontal RHO-points and vertical W-points,

                   bvf = - g/rho0 d(rho)/d(z).

    In computation of bvf the density anomaly difference is computed
    by adiabatically lowering/rising the water parcel from RHO point
    above/below to the W-point depth at "z_w".

    Reference:

    Jackett, D. R. and T. J. McDougall, 1995, Minimal Adjustment of
    Hydrostatic Profiles to Achieve Static Stability, Journ of Atmos.
    and Oceanic Techn., vol. 12, pp. 381-389.

    Check Values: T=3 C S=35.5 PSU Z=-5000 m rho=1050.3639165364

    Taken from ROMS_tools
    '''

    #if z_r>0.:
    #    print 'Warning!! z_r should be negative... and it has been set so!!!'
    #    z_r = -z_r

    if z_r is None: z_r = 0.

    A00 = +19092.56;    A01 = +209.8925
    A02 = -3.041638;    A03 = -1.852732e-3; A04 = -1.361629e-5; A10 = +104.4077
    A11 = -6.500517;    A12 = +0.1553190;   A13 = +2.326469e-4; AS0 = -5.587545
    AS1 = +0.7390729;   AS2 = -1.909078e-2; B00 = +4.721788e-1; B01 = +1.028859e-2
    B02 = -2.512549e-4; B03 = -5.939910e-7; B10 = -1.571896e-2; B11 = -2.598241e-4
    B12 = +7.267926e-6; BS1 = +2.042967e-3; E00 = +1.045941e-5; E01 = -5.782165e-10
    E02 = +1.296821e-7; E10 = -2.595994e-7; E11 = -1.248266e-9; E12 = -3.508914e-9

    QR  = +999.842594;  Q01 = +6.793952e-2; Q02 = -9.095290e-3
    Q03 = +1.001685e-4; Q04 = -1.120083e-6; Q05 = +6.536332e-9; Q10 = +0.824493
    Q11 = -4.08990e-3;  Q12 = +7.64380e-5;  Q13 = -8.24670e-7;  Q14 = +5.38750e-9
    QS0 = -5.72466e-3;  QS1 = +1.02270e-4;  QS2 = -1.65460e-6;  Q20 = +4.8314e-4

    sqrtTs = np.ma.sqrt(Ts)

    K0 = A00 + Tt * (A01 + Tt * (A02 + Tt * (A03 + Tt * A04)))               \
             + Ts * (A10 + Tt * (A11 + Tt * (A12 + Tt * A13))                \
             + sqrtTs * (AS0 + Tt * (AS1 + Tt * AS2)))
    K1 = B00 + Tt * (B01 + Tt * (B02 + Tt * B03))                              \
             + Ts * (B10 + Tt * (B11 + Tt * B12) + sqrtTs * BS1)
    K2 = E00 + Tt * (E01 + Tt * E02)                                         \
             + Ts * (E10 + Tt * (E11 + Tt * E12))
    rho1 = QR + Tt * (Q01 + Tt * (Q02 + Tt * (Q03 + Tt * (Q04 + Tt * Q05)))) \
              + Ts * (Q10 + Tt * (Q11 + Tt * (Q12 + Tt * (Q13 + Tt * Q14)))  \
              + sqrtTs * (QS0 + Tt * (QS1 + Tt * QS2)) + Ts * Q20)
    rho = rho1 / (1. + 0.1 * z_r / (K0 - z_r * (K1 - z_r * K2)))

    if z_w is not None:
        print ('THIS NEEDS TO BE CHECKED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        N, M, L = Tt.shape
        bvf = 0. * z_w
        cff = g / rho0
        bvf[1:N] = -cff * (rho1[1:N] /                                        \
              (1. + 0.1 * z_w[1:N] /                                          \
              (K0[1:N] - z_w[1:N] * (K1[1:N] - z_w[1:N] * K2[1:N])))          \
              -rho1[0:N-1] / ( 1. + 0.1 * z_w[1:N] /                          \
              (K0[0:N-1] - z_w[1:N] * (K1[0:N-1] - z_w[1:N] * K2[0:N-1])))) / \
              (z_r[1:N] - z_r[0:N-1])
    else:
        bvf = 0

    return rho, bvf



def rho_eos_duko(temp, salt, z_r, rho0=1027.40898956694):
    '''
    Get Dukowicz density
    Input: rho_eos_duko(temp, salt, z_r, rho0=1027.40898956694)
    Output: rho
    '''
    r00=999.842594;   r01=6.793952E-2;  r02=-9.095290E-3
    r03=1.001685E-4;  r04=-1.120083E-6; r05=6.536332E-9

    r10=0.824493;     r11=-4.08990E-3;  r12=7.64380E-5
    r13=-8.24670E-7;  r14=5.38750E-9

    rS0=-5.72466E-3;  rS1=1.02270E-4;   rS2=-1.65460E-6
    r20=4.8314E-4

    K00=19092.56;     K01=209.8925;     K02=-3.041638
    K03=-1.852732e-3; K04=-1.361629e-5

    K10=104.4077;     K11=-6.500517;    K12=0.1553190
    K13=2.326469e-4

    KS0=-5.587545;    KS1=+0.7390729;   KS2=-1.909078e-2

    B00=0.4721788;    B01=0.01028859;   B02=-2.512549e-4
    B03=-5.939910e-7
    B10=-0.01571896;  B11=-2.598241e-4; B12=7.267926e-6
    BS1=2.042967e-3

    E00=+1.045941e-5; E01=-5.782165e-10;E02=+1.296821e-7
    E10=-2.595994e-7; E11=-1.248266e-9; E12=-3.508914e-9

    temp0 = 3.8e0
    salt0 = 34.5e0
    
    sqrt_salt0 = np.sqrt(salt0)
    dr00 = r00 - 1000.e0
    rho1_0 = dr00+temp0*(r01+temp0*(r02+temp0*(r03+temp0*(r04+temp0*r05)))) \
                 +salt0*(r10+temp0*(r11+temp0*(r12+temp0*(    \
                  r13+temp0*r14)))          \
                 +sqrt_salt0*(rS0+temp0*(rS1+temp0*rS2))+salt0*r20)

    K0_Duk = temp0*( K01+temp0*( K02+temp0*( K03+temp0*K04 )))  \
                  +salt0*( K10+temp0*( K11+temp0*( K12+temp0*K13 )) \
                  +sqrt_salt0*( KS0+temp0*( KS1+temp0*KS2 )))

    dr00 = r00 - rho0
    sqrt_salt = np.sqrt(salt)
    
    # Rewritten with numexpr
    rho1 = ne.evaluate('dr00+temp*( r01+temp*( r02+temp*( r03+temp*( \
                                           r04+temp*r05 ))))       \
                          +salt*( r10+temp*( r11+temp*( r12+temp*( \
                                            r13+temp*r14 )))       \
                             +sqrt_salt*(rS0+temp*(                \
                                   rS1+temp*rS2 ))+salt*r20 )')


    K0 = ne.evaluate('temp*( K01+temp*( K02+temp*( K03+temp*K04 )))  \
                      + salt*( K10+temp*( K11+temp*( K12+temp*K13 ))  \
                      + sqrt_salt*( KS0+temp*( KS1+temp*KS2 )))')

    qp1 = ne.evaluate('0.1*( rho0+rho1 )*( K0_Duk-K0 )/(( K00+K0 )*( K00+K0_Duk ))')
    qp2 = 0.0000172
    rho = ne.evaluate('rho1 + qp1 * abs(z_r) * ( 1-qp2 * abs(z_r) )')
    return ne.evaluate('rho + rho0')




def get_slope(xr, yr, pm, pn, h):
    # Calculate bathymetric slope given lon, lat, pm, pn, h
    Lp, Mp = h.shape
    L    = Lp - 1
    M    = Mp - 1
    x    = np.zeros((L, M))
    y    = np.zeros((L, M))
    hx   = np.zeros((L, Mp))
    hy   = np.zeros((Lp, M))
    dhdx = np.zeros((L, M))
    dhde = np.zeros((L, M))
    x[0:L, 0:M] = 0.25 * (xr[0:L,   0:M] + xr[1:Lp,  0:M]  \
                        + xr[0:L,  1:Mp] + xr[1:Lp, 1:Mp])
    y[0:L, 0:M] = 0.25 * (yr[0:L,   0:M] + yr[1:Lp,  0:M]  \
                        + yr[0:L,  1:Mp] + yr[1:Lp, 1:Mp])
    # Compute bathymetry gradients at psi-points
    hx[0:L, 0:Mp] = 0.5 * (pm[0:L, 0:Mp] + pm[1:Lp, 0:Mp]) \
                        * (h[1:Lp, 0:Mp] - h[0:L, 0:Mp])
    hy[0:Lp,0:M]  = 0.5 * (pm[0:Lp, 0:M] + pm[0:Lp, 1:Mp]) \
                        * (h[0:Lp, 1:Mp] - h[0:Lp,0:M])
    dhdx[0:L,0:M] = 0.5 * (hx[0:L,0:M] + hx[0:L,1:Mp])
    dhde[0:L,0:M] = 0.5 * (hy[0:L,0:M] + hy[1:Lp,0:M])
    # Compute slope psi-points
    slope = np.sqrt(dhdx * dhdx + dhde * dhde)
    return slope


def pcol_2dxy(x, y):
    '''
    Function to shift x, y for subsequent use with pcolor
    by Jeroen Molemaker UCLA 2008
    '''
    Mp, Lp = x.shape
    M = Mp - 1
    L = Lp - 1

    x_pcol = np.zeros((Mp,Lp))
    y_pcol = np.zeros((Mp,Lp))
    x_tmp             = 0.5 * (x[:,0:L]     + x[:,1:Lp]    )
    x_pcol[1:Mp,1:Lp] = 0.5 * (x_tmp[0:M,:] + x_tmp[1:Mp,:])
    x_pcol[0,:]       = 2.  * x_pcol[1,:]     - x_pcol[2,:]
    x_pcol[:,0]       = 2.  * x_pcol[:,1]     - x_pcol[:,2]
    y_tmp             = 0.5 * (y[:,0:L]     + y[:,1:Lp]    )
    y_pcol[1:Mp,1:Lp] = 0.5 * (y_tmp[0:M,:] + y_tmp[1:Mp,:])
    y_pcol[0,:]       = 2.  * y_pcol[1,:]     - y_pcol[2,:]
    y_pcol[:,0]       = 2.  * y_pcol[:,1]     - y_pcol[:,2]
    return x_pcol, y_pcol


def arg_nearest(val, vec):
    '''
    ii = arg_nearest(val, vec)
    Return nearest index for 'val' in vector 'vec'
    '''
    nearest = np.argmin(np.abs(vec-val))
    return nearest


def getmycm(which_one,reverse=False):
    '''
    getmycm(which_one,reverse=False)
        which_one is a string: name of colormap, e.g., 'rainbow.pal'
        reverse=True is like _r
    '''
    cm = np.loadtxt(which_one)
    if reverse: cm = cm[::-1]
    cm = plt.matplotlib.colors.ListedColormap(cm) 
    return cm


def getncvcm(which_one, reverse=False, half=0):
    '''
    Returns ncview colormaps
    getncvcm(which_one, reverse=False)
        which_one is a string: name of colormap, e.g., 'rainbow'
        reverse=True is like _r
    Initial path (cmf) to ncview scr dir must be set
    Options are:
      3gauss.h  blu_red.h  default.h  helix2.h  jaisnb.h  rainbow.h
      3saw.h    bright.h   detail.h   helix.h   jaisnc.h  ssec.h
      banded.h  bw.h       extrema.h  hotres.h  jaisnd.h
    '''
    #cmf = '/opt/ncview/ncview-2.0beta4/src/colormaps_'
    #cmf = '/opt/ncview/ncview-2.1.1/src/colormaps_'
    #cmf = '/opt/ncview/ncview-2.1.2/src/colormaps_'
    #cmf = '/opt/netcdf_hdf5/ncview-2.1.7/src/colormaps_'
    #cmf = '/opt/netcdf4_hdf5/ncview-2.1.7/src/colormaps_'
    #cmf = '/home/emason/Dropbox/colormaps_'
    cmf = '/home/ulg/mast/emason/colormaps_'
    cmf = cmf + which_one + '.h'
    f = open(cmf, 'r')
    cm = f.read()
    cm = cm.partition('{')[-1].partition('}')[0].replace('\n', '')
    cm = np.fromstring(cm, dtype=float, sep=',' )
    '''try:
        while aa:
            aa = f.next()
            if not '*' in aa:
                #print 'aa',aa
                for a in aa:
                    if a.isdigit() or a == ',':
                        cm = cm + a
    except: pass'''
    f.close()
    #print('cm:  ', cm)
    #cm = cm.split(',')
    #print('cm',cm)
    #cm = np.array(cm, dtype=float)
    #print 'cm.shape',cm.shape
    cm = cm.reshape(256,3)
    cm /= 256.0
    if reverse:
        cm = cm[::-1]
    #print 'cm.shape before',cm.shape
    if half:
        if half == 1:
            cm = cm[:128]
            cm = nd.zoom(cm, (2,1), order=0)
            #print 'cm.shape after',cm.shape
        elif half == 2:
            cm = cm[128:]
            cm = nd.zoom(cm, (2,1), order=0)
            #print 'cm.shape after',cm.shape
        else:
            raise ValueError('half must be integer 0, 1 or 2')
    return plt.matplotlib.colors.ListedColormap(cm)


def get_limits(lon, lat, pad):
    '''
    Get lat/lon limits, add a 'pad' in degrees
        lonmin,lonmax,latmin,latmax = get_limits(lon,lat,pad)
    '''
    lonmax = lon.max() + pad
    lonmin = lon.min() - pad
    latmax = lat.max() + pad
    latmin = lat.min() - pad
    return lonmin, lonmax, latmin, latmax


def get_cax(subplot, dx=0, width=.015, position='r'):
    '''
    Return axes for colorbar same size as subplot
      subplot - subplot object
      dx - distance from subplot to colorbar
      width - width of colorbar
      position - 'r' - right (default)
               - 'b' - bottom
    '''
    cax = subplot.get_position().get_points()
    if position=='r':
        cax = plt.axes([cax[1][0]+dx,cax[0][1],width,cax[1][1]-cax[0][1]])
    if position=='b':
        cax = plt.axes([cax[0][0],cax[0][1]-2*dx,cax[1][0]-cax[0][0],width])
    return cax



def get_cax2(subplot1, subplot2, dx=0, width=.015, position='r', clip=0.):
    '''
    Return axes for colorbar same size as 2 subplots,
    either side by side or one over the other
      subplot1 - subplot object
      subplot2 - subplot object
      dx - distance from subplot to colorbar
      width - width of colorbar
      position - 'r' - right (default)
               - 'b' - bottom
      clip - clip length of colorbar by 'clip'
    '''
    cax1 = subplot1.get_position()
    cax2 = subplot2.get_position()
    if 'r' in position and position in 'r':
        # THIS WILL CRASH NEXT TIME CALLED, FIX AS FOR 'B' BELOW
        #cax = plt.axes([cax2[1][0]+dx,cax2[0][1],width,cax1[1][1]-cax2[0][1]])
        cax = plt.axes([cax2.x1+dx, cax2.y0+clip, width, cax1.y1-cax2.y0-clip*2])
    elif 'b' in position and position in 'b':
        # this needs checking
        #cax = plt.axes([cax2[0][0],cax2[0][1]-2*dx,cax1[1][0]-cax1[0][0],width])
        cax = plt.axes([cax1.x0+clip, cax1.y0+dx, cax2.x1-cax1.x0-clip*2, width])
    else: Exception
    return cax


def get_bry_subgrid(lonABCD,latABCD,lon,lat):
    '''
    Return 4 enclosing i,j index points for lon/lat.
    lonABCD,latABCD are lon/lat arrays with 4 points: starting
    bottom left and going anticlockwise
    imin,imax,jmin,jmax = get_bry_subgrid(lonABCD,latABCD,lon,lat)

    POSITIONS OF OUTPUTS CHANGED, 5 AUG 2010, WAS imin,jmin,imax,jmax
    MAY AFFECT FILES:
      fig_NEA_seas_paper_CVFZ_spice_velocity.py:    
      fig_NEA_seas_paper_CVFZ_ts_prof.py:    
      fig_NEA_seas_paper_ts_profiles_2.py:    
      fig_NEA_seas_paper_velocity_fields.py:   
      fig_DEA_pres_bry_comparison.py
      fig_bry_paper_bry_comparison_volcon.py
      fig_bry_paper_bry_comparison.py

    '''
    Ai, Aj = nearest(lonABCD[0],latABCD[0],lon,lat)
    Bi, Bj = nearest(lonABCD[1],latABCD[1],lon,lat)
    Ci, Cj = nearest(lonABCD[2],latABCD[2],lon,lat)
    Di, Dj = nearest(lonABCD[3],latABCD[3],lon,lat)
    imin = np.array([Ai, Bi, Ci, Di]).min()
    jmin = np.array([Aj, Bj, Cj, Dj]).min()
    imax = np.array([Ai, Bi, Ci, Di]).max() + 1
    jmax = np.array([Aj, Bj, Cj, Dj]).max() + 1
    return imin,imax,jmin,jmax



def isoslice2(var, prop, isoval=0, axis=0, masking=True):
    """
    result = isoslice(variable,property[,isoval=0])
    
    result is a a projection of variable at property == isoval in the first
    nonsingleton dimension.  In the case when there is more than one zero
    crossing, the results are averaged.
    
    EXAMPLE:
    Assume two three dimensional variable, s (salt), z (depth), and
    u (velocity), all on the same 3D grid.  x and y are the horizontal 
    positions, with the same horizontal dimensions as z (the 3D depth 
    field).  Here, assume the vertical dimension, that will be projected,
    is the first.  
    
    s_at_m5  = isoslice(s,z,-5);        # s at z == -5
    h_at_s30 = isoslice(z,s,30);       # z at s == 30
    u_at_s30 = isoslice(u,s,30);       # u at s == 30
    """
    if (len(var.squeeze().shape) <= 2):
        raise (ValueError, 'variable must have at least two dimensions')
    if not prop.shape == var.shape:
        raise (ValueError, 'dimension of var and prop must be identical')
    var = var.swapaxes(0, axis)
    prop = prop.swapaxes(0, axis)
    prop = prop - isoval
    sz = var.shape
    var = var.reshape(sz[0], -1)
    prop = prop.reshape(sz[0], -1)
    #find zero-crossings (zc == 1)
    zc = np.where((prop[:-1] * prop[1:]) < 0 ,1., 0.)
    varl = var[:-1] * zc
    varh = var[1:] * zc
    propl = prop[:-1] * zc
    proph = prop[1:] * zc
    result = varl - propl * (varh - varl) / (proph - propl)
    result = np.where(zc == 1., result, 0.)
    szc = zc.sum(axis=0)
    szc = np.where(szc == 0., 1, szc)
    result = result.sum(axis=0) / szc
    if masking:
        result = np.ma.masked_where(zc.sum(axis=0) == 0, result)
        if np.all(result.mask):
            raise (Warning, 'property==%f out of range (%f, %f)' % \
                           (isoval, (prop+isoval).min(), (prop+isoval).max()))
    result = result.reshape(sz[1:])
    return result




def isoslice(q, z, zo=0, mode='spline'):
    """Return a slice a 3D field along an isosurface.
    
    result is a a projection of variable at property == isoval in the first
    nonsingleton dimension.  In the case when there is more than one zero
    crossing, the results are averaged.
    
    EXAMPLE:
    Assume two three dimensional variable, s (salt), z (depth), and
    u (velicity), all on the same 3D grid.  x and y are the horizontal 
    positions, with the same horizontal dimensions as z (the 3D depth 
    field).  Here, assume the vertical dimension, that will be projected,
    is the first.  
    
    s_at_m5  = isoslice(s,z,-5);        # s at z == -5
    h_at_s30 = isoslice(z,s,30);       # z at s == 30
    u_at_s30 = isoslice(u,s,30);       # u at s == 30
    """
    import iso as _iso
    
    if 'linear' in mode:
        imode = 0
    elif 'spline' in mode:
        imode = 1
    else:
        imode = 1
        raise (Warning, '%s not supported, defaulting to splines' % mode)
    
    q = np.atleast_3d(q)
    z = np.atleast_3d(z)
    assert q.shape == z.shape, ('z and q must have the same shape')
    
    zo *= np.ones(z.shape[1:])
    
    q2d = _iso.zslice(z, q, zo, imode)
    if np.any(q2d == 1e20):
        q2d = np.ma.masked_where(q2d == 1e20, q2d)
    
    return q2d








def get_floats(directory, datafile, metafile):
    '''
    Read in SVP drifter data (obtained from
        http://www.aoml.noaa.gov/envids/gld/)
    Data come in slow-to-load ascii, therefore saved
    into mat file for future use
    '''
    def _get_floats_(datafile, metafile, fillval):
        print ('---reading float datafile:', datafile)
        # datafile header = id date time lat lon t ve vn speed varlat varlon vart
        usecols = [0,1,3,4]
        flt     = np.load(datafile, skiprows=1, usecols=usecols,
                         converters={1:plt.datestr2num})
        print ('---reading float metafile:', metafile)
        # metafile header = id wmo expno typebuoy ddate dtime dlat dlon edate etime
        #                   elat elon ldate ltime typedeath
        usecols = [0,4,6,7,8,10,11]
        meta = np.load(metafile, skiprows=1, usecols=usecols,
                         converters={4:plt.datestr2num, 8:plt.datestr2num})
        i_d_flt, lon_flt, lat_flt, date_flt = flt[:,0], flt[:,3],   \
                                              flt[:,2], flt[:,1]
        i_d_meta, lon_meta, lat_meta, rdate = meta[:,0], meta[:,3], \
                                              meta[:,2], meta[:,1]
        # All floats are big long list, want to separate them out
        # we'll call them just 'lon' and 'lat' etc...
        diff = np.diff(i_d_flt).nonzero() # find the breaks
        lon  = np.insert(lon_flt, np.squeeze(diff)+1,  fillval)
        lat  = np.insert(lat_flt, np.squeeze(diff)+1,  fillval)
        date = np.insert(date_flt, np.squeeze(diff)+1, fillval)
        fid  = np.insert(i_d_flt, np.squeeze(diff)+1, fillval)
        #lon  = MA.masked_where(lon==fillval,  lon)
        #lat  = MA.masked_where(lat==fillval,  lat)
        #date = MA.masked_where(date==fillval, date)
        # Here want to separate into a big matrix, each column
        # has individual track
        longest_traj = np.diff(diff).max() # longest trajectory
        num_traj = len(np.unique(i_d_flt)) # How many floats
        lon_mat  = np.ones((longest_traj, num_traj)) * fillval
        lat_mat  = lon_mat.copy()
        rdt_mat  = lon_mat.copy()
        fid_mat  = lon_mat.copy()
        i = 0; j = 0
        for k in np.arange(len(lon)):
            if lon[k]==fillval:
                j += 1; i  = 0
            else:
                lon_mat[i,j], lat_mat[i,j], rdt_mat[i,j], fid_mat[i,j] = \
                lon[k], lat[k], date[k], fid[k]
                i += 1
        lon_mat = np.vstack((lon_meta, lon_mat))
        lat_mat = np.vstack((lat_meta, lat_mat))
        rdt_mat = np.vstack((rdate,    rdt_mat))
        fid_mat = np.vstack((i_d_meta, fid_mat))
        lon_mat = MA.masked_where(lon_mat==fillval, lon_mat)
        lat_mat = MA.masked_where(lat_mat==fillval, lat_mat)
        rdt_mat = MA.masked_where(rdt_mat==fillval, rdt_mat)
        fid_mat = MA.masked_where(fid_mat==fillval, fid_mat)
        # put all into a Bunch
        floats = Bunch(lon=lon_mat, lat=lat_mat,
                       rdate=rdt_mat, flt_id=fid_mat)
        # Save to mat file
        mat = {  'lon':lon_mat,    'lat':lat_mat,
               'rdate':rdt_mat, 'flt_id':fid_mat}
        io.savemat(datafile, mat, appendmat=True)
        return floats
    fillval = -9999.
    # Check if mat file already available, if not make one...
    mat_exists = datafile + '.mat'
    if os.path.exists(directory + mat_exists):
        print ('---reading float matfile:', directory + mat_exists)
        floats = io.loadmat(directory + mat_exists)
        lon_mat = floats['lon']
        lat_mat = floats['lat']
        rdt_mat = floats['rdate']
        fid_mat = floats['flt_id']
        lon_mat = MA.masked_where(lon_mat==fillval, lon_mat)
        lat_mat = MA.masked_where(lat_mat==fillval, lat_mat)
        rdt_mat = MA.masked_where(rdt_mat==fillval, rdt_mat)
        fid_mat = MA.masked_where(fid_mat==fillval, fid_mat)
        # put all into a Bunch
        floats = Bunch(lon=lon_mat, lat=lat_mat,
                       rdate=rdt_mat, flt_id=fid_mat)
    else:
        datafile = directory + datafile
        metafile = directory + metafile
        floats = _get_floats_(datafile, metafile, fillval)
    return floats
    
    


def okubo_weiss(u, v, pm, pn, divergence=False):
    '''
    Calculate the Okubo-Weiss parameter
    See e.g. http://ifisc.uib.es/oceantech/showOutput.php?idFile=61
    Returns: lambda2 - Okubo-Weiss parameter [s^-2]
             xi      - rel. vorticity [s^-1]
    Copied from Roms_tools
    '''
    Mp, Lp = pm.shape
    L  = Lp-1
    M  = Mp-1
    Lm = L-1
    Mm = M-1
    xi   = np.zeros((M,L))
    mn_p = np.zeros((M,L))
    uom  = np.zeros((M,Lp))
    von  = np.zeros((Mp,L))
    uom  = 2. * u / (pm[:,:L] + pm[:,1:Lp])
    uon  = 2. * u / (pn[:,:L] + pn[:,1:Lp])
    von  = 2. * v / (pn[:M,:] + pn[1:Mp,:])
    vom  = 2. * v / (pm[:M,:] + pm[1:Mp,:])
    mn   = pm * pn
    mn_p = (mn[:M, :L]  + mn[:M,  1:Lp] +    \
            mn[1:Mp,1:Lp] + mn[1:Mp, :L]) * 0.25
    # Relative vorticity
    xi = mn * psi2rho(von[:,1:Lp] - von[:,:L] - uom[1:Mp,:] + uom[:M,:])
    # Sigma_T
    ST = mn * psi2rho(von[:,1:Lp] - von[:,:L] + uom[1:Mp,:] - uom[:M,:])
    # Sigma_N
    SN = np.zeros((Mp,Lp))
    SN[1:-1,1:-1] = mn[1:-1,1:-1] * (uon[1:-1,1:]    \
                                   - uon[1:-1,:-1]   \
                                   - vom[1:,1:-1]    \
                                   + vom[:-1,1:-1])
    lambda2 = np.power(SN,2) + np.power(ST,2) - np.power(xi,2)
    if divergence: return SN
    else:          return lambda2, xi
    
    
    
def spiciness(t,s,p=0):
    '''
    Compute spiciness = spice(t,s,p)
     adapted from algorithm developed by plt. Flament.
    SCKennan(Dec92)
    From http://www.satlab.hawaii.edu/spice/
    
    Usage: spiciness = spiciness(t,s,p=0)
    '''
    B = np.zeros((6,5))
    B[0,0] = 0
    B[0,1] = 7.7442e-001
    B[0,2] = -5.85e-003
    B[0,3] = -9.84e-004
    B[0,4] = -2.06e-004

    B[1,0] = 5.1655e-002
    B[1,1] = 2.034e-003
    B[1,2] = -2.742e-004
    B[1,3] = -8.5e-006
    B[1,4] = 1.36e-005

    B[2,0] = 6.64783e-003
    B[2,1] = -2.4681e-004
    B[2,2] = -1.428e-005
    B[2,3] = 3.337e-005
    B[2,4] = 7.894e-006

    B[3,0] = -5.4023e-005
    B[3,1] = 7.326e-006
    B[3,2] = 7.0036e-006
    B[3,3] = -3.0412e-006
    B[3,4] = -1.0853e-006

    B[4,0] = 3.949e-007
    B[4,1] = -3.029e-008
    B[4,2] = -3.8209e-007
    B[4,3] = 1.0012e-007
    B[4,4] = 4.7133e-008

    B[5,0] = -6.36e-010
    B[5,1] = -1.309e-009
    B[5,2] = 6.048e-009
    B[5,3] = -1.1409e-009
    B[5,4] = -6.676e-010

    def calc(t,s,B):
        r,c = np.atleast_2d(t).shape
        sp  = np.zeros((r,c))
        s   = s - 35. * np.ones((r,c))
        T   = 1. * np.ones((r,c))
        for i in np.arange(6):
            S = np.ones((r,c))
            for j in np.arange(5):
                sp = sp + B[i,j] * T * S
                S = S * s
            T = T * t
        return sp
    if np.ndim(t)==3: # Loop for a 3d
        z,y,x = t.shape
        sp = np.zeros((z,y,x))
        for k in np.arange(z):
            sp[k] = calc(t[k],s[k],B)
    else:
        sp = calc(t,s,B)
    return sp
    
    
    
def get_pressure(rho, zeta, zr, g=9.81):
    '''
    Get ROMS pressure
    '''
    nz,ny,nx = rho.shape
    press    = np.ma.zeros((nz,ny,nx))
    # Surface pressure
    press[nz-1] = g * rho[nz-1] * zeta
    # Integrate down to get 3d hydrostatic pressure
    dz = zr[1:] - zr[0:-1]
    for k in np.arange(nz-2,-1,-1):
        press[k] = press[k+1] + dz[k]*g*0.5*np.ma.add(rho[k],rho[k+1])
    return press
    
    
    
def oceantime2ymd(ocean_time, integer=False):
    '''
    Return strings y, m, d given ocean_time (seconds)
    If integer=True return integer rather than string
    '''
    if np.isscalar(ocean_time):
        ocean_time = np.array([ocean_time])
    ocean_time /= 86400.
    year = np.floor(ocean_time / 360.)
    month = np.floor((ocean_time - year * 360.) / 30.)
    day = np.floor(ocean_time - year * 360. - month * 30.)
    year = (year.astype(np.int16) + 1)[0]
    month = (month.astype(np.int16) + 1)[0]
    day = (day.astype(np.int16) + 1)[0]
    if not integer:
        year = str(year)
        month = str(month)
        day = str(day)
        if len(year) < 2: year = '0' + year
        if len(month) < 2: month = '0' + month
        if len(day) < 2: day = '0' + day
    return year, month, day
    
    
def ymd2oceantime(y, m, d, integer=True):
    '''
    Return integer oceantime given y, m, d.
    If integer=False return string rather than integer
    '''
    return (((y - 1) * 360 * 86400) +
            ((m - 1) * 30 * 86400) +
             (d * 86400)) - 86400
    
    
    
    
def get_obcvolcons_2d(u, v, zw, pm, pn, mask, dbd=True, dzu=None, dzv=None, obc=None):
    '''
    Get volume-conserving flux correction around the domain
    Adapted from roms2roms get_obcvolcons_2d.m by
    Jeroen Molemaker.
    In contrast to Roms_tools get_obcvolcons.m, this correction
    is not constant in barotropic flux but barotropic velocity,
    in shallow areas in particular this is a significant difference.

    dbd is 'divide by depth'; set to False if wanting a transport

    Evan Mason
    '''
    assert u.ndim == 3, 'u must be 3d'
    assert v.ndim == 3, 'v must be 3d'

    if obc is None:
        obc = (1., 1., 1., 1.)

    umask, vmask, psimask = uvp_mask(mask)

    # Estimate the barotropic velocity
    if dzu is None:
        print ('using zw')
        dz  = np.diff(zw, axis=0)
        dzu = 0.5 * (dz[..., :-1] + dz[..., 1:]) # convert to u
        dzv = 0.5 * (dz[:, :-1] + dz[:, 1:]) # and v points
    else:
        assert u.shape == dzu.shape, 'u and dzu must have same shape'
        assert v.shape == dzv.shape, 'v and dzv must have same shape'
    hu = (dzu * u).sum(axis=0)
    hv = (dzv * v).sum(axis=0)
    du = dzu.sum(axis=0)
    dv = dzv.sum(axis=0)
    if dbd:
        ubar = umask * (hu / du)
        vbar = vmask * (hv / dv)
    else:
        print ('No division by depth, so returning a transport')
        ubar = umask * hu
        vbar = vmask * hv

    ubar[np.isnan(ubar)] = 0.
    vbar[np.isnan(vbar)] = 0.

    for k in np.arange(u.shape[0]):
        u[k] -= ubar
        v[k] -= vbar

    dyu = du * 2. * umask / (pn[:, 1:] + pn[:, :-1])
    dxv = dv * 2. * vmask / (pm[1:] + pm[:-1])
    udy = ubar * dyu
    vdx = vbar * dxv
    
    Flux  = obc[0] * (vdx[0, 1:-1]).sum() - \
            obc[1] * (udy[1:-1,-1]).sum() - \
            obc[2] * (vdx[-1,1:-1]).sum() + \
            obc[3] * (udy[1:-1, 0]).sum()
    Cross = obc[0] * (dxv[0, 1:-1]).sum() + \
            obc[1] * (dyu[1:-1,-1]).sum() + \
            obc[2] * (dxv[-1,1:-1]).sum() + \
            obc[3] * (dyu[1:-1, 0]).sum()

    vcorr = Flux / Cross
    print ('--- barotropic velocity correction:', str(vcorr), 'm/s')

    vbar[0] = obc[0] * (vbar[0]  - vcorr)
    ubar[:, -1] = obc[1] * (ubar[:, -1] + vcorr)
    vbar[-1] = obc[2] * (vbar[-1] + vcorr)
    ubar[:, 0] = obc[3] * (ubar[:, 0]  - vcorr)
    
    ubar *= umask
    vbar *= vmask
    
    for k in np.arange(u.shape[0]):
        u[k] += ubar
        v[k] += vbar

    return u, v, ubar, vbar



def get_psi(u, v, pm, pn, mask):
    '''
    Compute a stream function from a ROMS vector
    field (velocity or transport)
       1 - get boundary conditions for psi
       2 - diffuse these boundary conditions on the land points
       3 - get Xi the vorticity
       4 - inverse laplacian(psi)=Xi
    Adapted from Roms_tools
    '''

    #-----------------------------------------------------------------
    def step_sor(rhs,psi,xi,a2,a3,a4,a5,b1,b2,ijeven,parity,W,J2,
                 bcpsi):
        '''
        1 Step of the elliptic solver (SOR) used in get_psi.m
        Copyright (c) 2001-2006 by Pierrick Penven
        e-mail:Pierrick.Penven@ird.fr  
        Adpated from a matlab script of S. Herbette (UBO)
        '''
        # Get the right hand side terms of the equations
        rhs[1:-1, 1:-1] = a2[1:-1, 1:-1] * psi[1:-1, 2:]    + \
                          a3[1:-1, 1:-1] * psi[1:-1, 0:-2]  + \
                          a4[1:-1, 1:-1] * psi[2:, 1:-1]    + \
                          a5[1:-1, 1:-1] * psi[:-2, 1:-1]   + \
                          b1[1:-1, 1:-1] * psi[1:-1, 1:-1] + \
                          b2[1:-1, 1:-1] * xi[1:-1, 1:-1]
        #Step PSI
        psi[ijeven == parity] = psi[ijeven == parity]     - \
                              W * rhs[ijeven == parity] / \
                              b1[ijeven == parity]
        # Advance the surrelaxation parameter
        W = 1. / (1. - J2 * 0.25 * W)
        #Apply the mask (but not to the islands bcpsi != 0)
        psi[np.logical_and(pmask == 0, bcpsi != 0)] = \
                  bcpsi[np.logical_and(pmask == 0, bcpsi != 0)]
        return psi, rhs, W
    #-----------------------------------------------------------------

    M, L  = pm.shape
    pmask = mask[1:, 1:] * mask[1:, :-1] * \
            mask[0:-1, 1:] * mask[:-1, :-1]

    # Get lateral boundary conditions for psi
    psi = np.zeros((M-1, L-1))
    uon = 2. * u / (pn[:, :-1] + pn[:, 1:])
    vom = 2. * v / (pm[:-1]   + pm[1:])
    '''
    Quick fix for a bug: stepsor cherche bcpsi~=0 pour ne
    pas appliquer de condition aux limites sur les iles.
    or si psi(1,1)=0 et si c'est masque, alors c'est considere
    comme une ile.
    le fixer a 1e7 devrait prevenir le probleme, mais ce n'est
    pas fiable a 100%.
    '''
    psi[0, 0] = 1e7

    for j in np.arange(1, M-1):
        psi[j, 0] = psi[j-1, 0] - uon[j, 0]
    for i in np.arange(1, L-1):
        psi[-1, i] = psi[-1, i-1] + vom[-1, i]
    psiend = psi[-1, -1]
    for i in np.arange(1, L-1):
        psi[0, i] = psi[0, i-1] + vom[0, i]
    for j in np.arange(1, M-1):
        psi[j, -1] = psi[j-1, -1] - uon[j, -1]
    if psiend != 0:
        deltapsi = 100. * np.abs((psi[-1,-1] - psiend) / psiend)
    else:
        deltapsi = 100. * np.abs(psi[-1,-1])
    if deltapsi > 1e-10:
        print ('Warning: no mass conservation (deltapsi=', str(deltapsi),')')
    '''
    Diffuse the psi boundaries condition on land to get
    a constant psi on the mask.
    WARNING !!! This does not work for islands
    '''
    land = 1. - pmask
    bcpsi = land * psi
    mean1 = bcpsi.mean()
    epsil = 1.e-6
    delta_mean = epsil + 1.
    n = 1.
    nmax = 500.
    while np.logical_and(delta_mean > epsil, n < nmax):
        lmask = 0 * bcpsi
        lmask[bcpsi!=0.] = 1.
        denom = (land[0:-2, 1:-1] * lmask[0:-2, 1:-1] +
                 land[2:,   1:-1] * lmask[2:,   1:-1] +
                 land[1:-1, 0:-2] * lmask[1:-1, 0:-2] +
                 land[1:-1, 2:]   * lmask[1:-1, 2:])
        denom[denom == 0.] = 1.
        rhs = (land[0:-2, 1:-1] * bcpsi[0:-2, 1:-1] +
               land[2:,   1:-1] * bcpsi[2:,   1:-1] +
               land[1:-1, 0:-2] * bcpsi[1:-1, 0:-2] +
               land[1:-1, 2:]   * bcpsi[1:-1, 2:]) / denom
        bcpsi[1:-1, 1:-1] = land[1:-1,1:-1] * rhs
        mean2 = bcpsi.mean()
        delta_mean = np.abs((mean1 - mean2) / mean2)
        mean1 = mean2.copy()
        n += 1.
        if n >= nmax:
            print ('Mask: no convergence')
    print ('Mask:', str(n), 'iterations')
    '''
    Prepare for psi integration div(psi)=xi
    '''
    mn  = pm * pn
    mon = pm / pn
    nom = pn / pm
    a1 = 0.25 * (mn[0:-1, 0:-1] + mn[0:-1,  1:]
               + mn[1:,   0:-1] + mn[1:,    1:])
    a2 = 0.5 * (mon[0:-1, 1:]   + mon[1:,   1:])
    a3 = 0.5 * (mon[0:-1, 0:-1] + mon[1:,   0:-1])
    a4 = 0.5 * (nom[1:,   0:-1] + nom[1:,   1:])
    a5 = 0.5 * (nom[0:-1, 0:-1] + nom[0:-1, 1:])
    b1 = -(a2 + a3 + a4 + a5)
    b2 = -1. / a1
    J2 = np.power( ( (np.cos(np.pi/L) + np.power(nom,2) * np.cos(np.pi/M)) /
                     (1. + np.power(nom,2) ) ).max(), 2)
    '''
    Get the vorticity
    '''
    uom = 2 * u / (pm[:, 0:-1] + pm[:, 1:])
    von = 2 * v / (pn[:-1] + pn[1:])
    xi  = pmask * a1 * (von[:, 1:] - von[:, :-1] - uom[1:] + uom[:-1])
    '''
    Inversion of the elliptic equation (SOR)
    Initial value of surrelaxation
    '''
    W = 1.
    '''
    Get a matrix of parity (i.e. i+j = odd or even)
    '''
    myi = np.arange(L-1)
    myj = np.arange(M-1)
    MYI, MYJ = np.meshgrid(myi, myj)
    ijeven = np.mod(MYJ + MYI, 2)
    '''
    First step (odd and even)
    '''
    rhs = 0 * psi
    parity = 1. # (i+j) even
    psi, rhs, W = step_sor(rhs, psi, xi, a2, a3, a4, a5, b1, b2,
                           ijeven, parity, W, J2, bcpsi)
    parity = 0. # (i+j) odd
    psi, rhs, W = step_sor(rhs, psi, xi, a2, a3, a4, a5, b1, b2,
                           ijeven, parity, W, J2, bcpsi)
    '''
    Integration until convergence
    '''
    meanrhs = np.abs(rhs[pmask==1]).mean()
    Norm0 = meanrhs
    epsil = 1.e-4
    n = 1.
    nmax = M * L
    while meanrhs > (epsil * Norm0):
        n += 1.
        parity = 1. # (i+j) even
        psi, rhs, W = step_sor(rhs, psi, xi, a2, a3, a4, a5,
                               b1, b2, ijeven, parity, W, J2, bcpsi)
        parity = 0. #(i+j) odd
        psi, rhs, W = step_sor(rhs, psi, xi, a2, a3, a4, a5,
                               b1, b2, ijeven, parity, W, J2, bcpsi)
        meanrhs = np.abs(rhs[pmask==1]).mean()
        if n > nmax:
            meanrhs = 0.
            print ('PSI: No convergence')
    print ('PSI: ', str(n), ' iterations ')

    psi -= psi.mean()

    return psi



def strabsint(x):
    '''
    Return a string, abs, int
    '''
    return str(np.abs(np.int(x)))



def monthnum(month, short=False):
    '''
    Return month string (eg, January) given
      a months index (eg, 1 for January)
    If short=True, return Jan
    '''
    if month == 0:
        raise (Warning, 'month must be between 1 and 12')
    months = ['January','February','March',
              'April','May','June','July',
              'August','September','October',
              'November','December']
    mon_short = ['Jan','Feb','Mar',
                 'Apr','May','Jun','Jul',
                 'Aug','Sep','Oct',
                 'Nov','Dec']
    mon_shortest = ['J','F','M','A','M','J',
                    'J','A','S','O','N','D']
    if short is False:
        return months[month - 1]
    else: return months_short[month - 1]



def get_vars_regular_grid(limits, lon, lat, mask, fillval,
        vars2d=False, vars3d=False):
    '''
    Return 2d and 3d ROMS fields on regular (unrotated) grid

    lon,lat,mask,vars2d,vars3d = get_vars_regular_grid(limits,
                            lon,lat,mask,vars2d,vars3d,fillval)

    limits = [lonmin,lonmax,latmin,latmax] We want to interp
                                           all the variables
                                           into this new grid
    lon,lat from grd file
    vars2d - dictionary of 2d variables
    vars3d - dictionary of 3d variables

    Note!! Linear interpolation, so mask will be corrupted
           Solution is maski[maski>=1.] = 1.
                       maski[maski<1.] = 0
    '''
    #from scipy.interpolate import griddata

    vars2di = {}
    vars3di = {}
    imin,imax,jmin,jmax = enclose_box(limits[0],limits[1],
                                      limits[2],limits[3],
                                      lon,lat)
    points = np.array([lon[jmin:jmax,imin:imax].ravel(),
                       lat[jmin:jmax,imin:imax].ravel()]).T
    lati,loni = np.mgrid[limits[2]:limits[3]:(jmax-jmin)*1.05j,
                         limits[0]:limits[1]:(imax-imin)*1.05j]

    maski = griddata(points, mask[jmin:jmax,imin:imax].ravel(),
                     (loni, lati), method='nearest')
    #maski[maski>=1.] = 1.
    #maski[maski<1.]  = 0
    #plt.pcolormesh(maski);aa
    # Loop over the 2d fields
    if vars2d:
        for key,var in vars2d.items():
            vari = griddata(points,var[jmin:jmax,imin:imax].ravel(),
                            (loni,lati),method='linear',fill_value=fillval)
            vari[maski==0] = fillval
            vars2di[key]   = vari
    # Loop over the 3d fields
    if vars3d:
        for key,var in vars3d.items():
            vari = tridim(np.ma.zeros((loni.shape)),var.shape[0])
            for k in np.arange(var.shape[0]):
                vari[k] = griddata(points,var[k,jmin:jmax,imin:imax].ravel(),
                                   (loni,lati),method='linear')
                vari[k][maski==0] = fillval
            vars3di[key] = vari
    # Do the returns
    if vars2d:
        if vars3d: return loni,lati,maski,vars2di,vars3di
        else:      return loni,lati,maski,vars2di
    else:
        return loni,lati,maski,vars3di
    return loni,lati,maski
    


def lonlat2dist(lon_vec,lat_vec):
    '''
    Return vector of distances along the two vectors
    lon_vec and lat_vec
    dist = lonlat2dist(lon_vec,lat_vec)
    '''
    dist = lon_vec * 0
    ind = 0
    for dx,dy in zip(lon_vec,lat_vec):
        dist[ind]= distLonLat(lon_vec[0],lat_vec[0],dx,dy)[0]
        ind += 1
    return dist



def wind_spd_to_stress(u, v):
    '''
    Calculate wind stress following Yelland etal 1998
    (a correction of Yelland and Taylor 1996)
    u_stress, v_stress = et.wind_spd_to_stress(u, v)
    '''
    rho_air  = 1.3
    U10 = np.hypot(u, v)
    # prevent divide by zeros
    U10[U10 == 0.] = 1e-10
    CD = np.zeros_like(U10)
    
    wind_min = 0.
    wind_max = 6.
    
    Cd_ind = np.logical_and(U10 >= wind_min, U10 <= wind_max)
    
    CD[Cd_ind] = ((0.29 + 3.1 / U10[Cd_ind] 
                       + 7.7 / np.power(U10[Cd_ind], 2))
                             / 1000.)
    wind_min = 6.
    wind_max = 100. # 26 m/s is upper limit in paper...
                    # but see Summary for open ocean conditions
                    
    Cd_ind = np.logical_and(U10 > wind_min, U10 <= wind_max)
    
    ''' # Below from Yelland & taylor 96
    CD[(U10>=wind_min)&(U10<=wind_max)] =                   \
       (0.60 + 0.07*U10[(U10>=wind_min)&(U10<=wind_max)])   \
             / 1000.'''
    
    # Below from Yelland etal 98 (correction to 96)
    CD[Cd_ind] = (0.50 + 0.071 * U10[Cd_ind]) / 1000.
    
    if U10.max()>wind_max:
        print ('!!!  wind_max greater than',  wind_max)
        print ('     U10 max is', U10.max())
    
    u_stress = rho_air * CD * u * U10
    v_stress = rho_air * CD * v * U10
    return u_stress, v_stress



def wind_stress_to_spd(u_stress, v_stress):
    # Calculate wind speed inverting Yelland and Taylor 1996

    ##### NEEDS WORK #################	

    rho_air  = 1.3
    Ustrs10  = np.hypot(u_stress, v_stress)
    CD       = np.zeros_like(Ustrs10)
    wind_strs_min = 0.
    wind_strs_max = 0.047736 # stress for 6 m/s
    CD[(Ustrs10>=wind_strs_min)&(Ustrs10<=wind_strs_max)] =                   \
       (-0.29 - Ustrs10[(Ustrs10>=wind_strs_min)&(Ustrs10<=wind_strs_max)]/3.1
              - Ustrs10[(Ustrs10>=wind_strs_min)&(Ustrs10<=wind_strs_max)]**2)/7.7 \
             * 1000.
    wind_strs_min = 0.047736 # stress for 6 m/s
    wind_strs_max = 100
    CD[(Ustrs10>=wind_strs_min)&(Ustrs10<=wind_strs_max)] =                   \
       (0.50 + 0.071*Ustrs10[(Ustrs10>=wind_strs_min)&(Ustrs10<=wind_strs_max)])   \
             / 1000.
    if Ustrs10.max()>wind_strs_max:
        print ('!!!  wind_strs_max greater than',  wind_strs_max)
        print ('     Ustrs10 max is', Ustrs10.max())
    
    uu10 = ustrs / (rho_air * CD)
    vv10 = vstrs / (rho_air * CD)
    u = np.sqrt(uu10) * int(u_stress / abs(u_stress))
    v = np.sqrt(vv10) * int(v_stress / abs(v_stress))

    return u,v



def boundary(lon,lat):
    # Get the perimeter around a grid
    lon = np.hstack((lon[0:,0],      lon[-1,1:-1],
                     lon[-1::-1,-1], lon[0, -2::-1]))
    lat = np.hstack((lat[0:,0],      lat[-1,1:-1],
                     lat[-1::-1,-1], lat[0, -2::-1]))
    return lon, lat




#def omega2w_OLD(u, v, omega, zr, pm, pn):
    #'''
    #Compute vertical velocity ROMS output u,v,omega
    
        #w = omega2w(u, v, omega, zr, pm, pn)
    
    #'''
    #u     = np.ma.swapaxes(u,2,0)
    #v     = np.ma.swapaxes(v,2,0)
    #omega = np.ma.swapaxes(omega,2,0)
    #zr    = np.ma.swapaxes(zr,2,0)
    #pm    = np.ma.swapaxes(pm,1,0)
    #pn    = np.ma.swapaxes(pn,1,0)

    #Lr, Mr, Np = omega.shape
    #N = Np - 1
    #w   = np.ma.zeros((Lr,   Mr,  N))
    #Wxi = np.ma.zeros((Lr-1, Mr,  N))
    #Wet = np.ma.zeros((Lr,   Mr-1,N))

    ##  dsigma/dt component
    #w[:,:,N-1]   =  0.375  *  omega[:,:,N] + 0.75 *                           \
                              #omega[:,:,N-1] - 0.125 * omega[:,:,N-2]
    #w[:,:,1:N-1] =  0.5625 * (omega[:,:,2:N]   + omega[:,:,1:N-1]) - 0.0625 * \
                             #(omega[:,:,3:N+1] + omega[:,:,0:N-2])
    #w[:,:,0]     = -0.125  *  omega[:,:,2]     + 0.75 * omega[:,:,1] +        \
                                                #0.375 * omega[:,:,0]

    ## domega/dx, domega/dy component
    #for kk in np.nditer(np.arange(N)):

        #Wxi[0:Lr-1,:,kk] = np.squeeze(u[:,:,kk]  *                              \
                                    #(zr[1:Lr,:,kk] - zr[0:Lr-1,:,kk])) * 0.5 *  \
                                    #(pm[1:Lr] + pm[0:Lr-1])
        #Wet[:,0:Mr-1,kk] = np. squeeze(v[:,:,kk] *                              \
                                     #(zr[:,1:Mr,kk] - zr[:,0:Mr-1,kk])) *0.5 *  \
                                     #(pn[:,1:Mr] + pn[:,0:Mr-1])

    #w[1:Lr-1,1:Mr-1] =  w[1:Lr-1,1:Mr-1]   + 0.5 *                \
                       #(Wxi[0:Lr-2,1:Mr-1] + Wxi[1:Lr-1,1:Mr-1] + \
                        #Wet[1:Lr-1,0:Mr-2] + Wet[1:Lr-1,1:Mr-1])

    #w[:,0]    = w[:,1]
    #w[:,Mr-1] = w[:,Mr-2]
    #w[0]      = w[1]
    #w[Lr-1]   = w[Lr-2]

    #return np.ma.swapaxes(w, 2, 0)



def omega2w(u_in, v_in, omega_in, zr_in, pm_in, pn_in):
    '''
    Compute vertical velocity ROMS output u,v,omega
    
        w = omega2w(u, v, omega, zr, pm, pn)
    
    '''
    u = u_in.view().swapaxes(2, 0)
    v = v_in.view().swapaxes(2, 0)
    omega = omega_in.view().swapaxes(2, 0)
    zr = zr_in.view().swapaxes(2, 0)
    pm = pm_in.view().swapaxes(1, 0)
    pn = pn_in.view().swapaxes(1, 0)
    
    Lr, Mr, Np = omega.shape
    N = Np - 1
    w   = np.zeros((Lr, Mr, N))
    Wxi = np.zeros((Lr-1, Mr, N))
    Wet = np.zeros((Lr, Mr-1, N))

    #  dsigma/dt component
    o_N = omega[..., N].view()
    o_Nm1 = omega[...,N - 1].view()
    o_Nm2 = omega[...,N - 2].view()
    w[..., N - 1] = ne.evaluate('0.375 * o_N + 0.75 * o_Nm1 - 0.125 * o_Nm2')
    o_2N = omega[..., 2:N].view()
    o_1Nm1 = omega[..., 1:N - 1].view()
    o_3Np1 = omega[..., 3:N + 1].view()
    o_Nm2 = omega[..., :N - 2].view()
    w[..., 1:N - 1] = ne.evaluate('0.5625 * (o_2N + o_1Nm1) - 0.0625 * (o_3Np1 + o_Nm2)')
    o_2 = omega[..., 2].view()
    o_1 = omega[..., 1].view()
    o_0 = omega[..., 0].view()
    w[..., 0] = ne.evaluate('-0.125 * o_2 + 0.75 * o_1 + 0.375 * o_0')

    # domega/dx, domega/dy component
    pm_1L = pm[1:Lr].view()
    pm_Lm1 = pm[:Lr - 1].view()
    pn_1M = pn[:, 1:Mr].view()
    pn_Mm1 = pn[:, :Mr - 1].view()
    for kk in np.nditer(np.arange(N)):
        
        u_kk = u[..., kk].view()
        zr_1L_kk = zr[1:Lr, :, kk].view()
        zr_Lm1_kk = zr[:Lr - 1, :, kk].view()
        Wxi[:Lr - 1, :, kk] = ne.evaluate('(u_kk * (zr_1L_kk - zr_Lm1_kk)) * 0.5 * (pm_1L + pm_Lm1)')
        
        v_kk = v[..., kk].view()
        zr_1M_kk = zr[:, 1:Mr, kk].view()
        zr_Mm1_kk = zr[:, :Mr - 1, kk].view()
        Wet[:, :Mr - 1, kk] = ne.evaluate('(v_kk * (zr_1M_kk - zr_Mm1_kk)) * 0.5 * (pn_1M + pn_Mm1)')
    
    w_LM = w[1:Lr - 1, 1:Mr - 1].view()
    Wxi_1 = Wxi[:Lr - 2, 1:Mr - 1].view()
    Wxi_2 = Wxi[1:Lr - 1, 1:Mr - 1].view()
    Wet_1 = Wet[1:Lr - 1, :Mr - 2].view()
    Wet_2 = Wet[1:Lr - 1, 1:Mr - 1].view()
    w[1:Lr - 1, 1:Mr - 1] =  ne.evaluate('w_LM + 0.5 * (Wxi_1 + Wxi_2 + Wet_1 + Wet_2)')

    w[:, 0] = w[:,1]
    w[:, Mr - 1] = w[:, Mr - 2]
    w[0] = w[1]
    w[Lr -1] = w[Lr - 2]

    return w.view().swapaxes(2, 0)


def remap_array(x, msize, nsize, order=1, mode='nearest'):
    '''
    Increase the resolution of a 2d field using
    ndimage.map_coordinates...
    Input:
      x    - 2d array
      msize - desired size of y-dimension
      nsize - desired size of x-dimension
    Output:
      Returns a masked array...
      newx - remapped x, 2d array, shape (msize,nsize)
    Usage: newx = resize_array(x,msize,nsize)
    
    '''
    m,n = x.shape
    newm,newn = mlab.frange(m-1,npts=msize), \
                mlab.frange(n-1,npts=nsize)
    newn,newm = np.meshgrid(newn,newm)
    newx = nd.map_coordinates(x,[newm,newn],order=order,mode=mode)
    if np.ma.isMaskedArray(x):
        mask = np.ma.getmask(x)
        mask = nd.map_coordinates(mask,[newm,newn],order=0)
        #plt.figure(20);plt.pcolormesh(mask);plt.colorbar()
        newx = np.ma.masked_where(mask==True,newx)
    newx = np.ma.masked_outside(newx,x.min(),x.max())
    return newx


def set_cb_fontsize(cb,size):
    '''
    Set the fontsize for a colorbar
    Example usage:
        cb = plt.colorbar()
        et.set_cb_fontsize(cb,10)
    '''
    if cb.orientation=='vertical':
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(size)
    elif cb.orientation=='horizontal':
        for t in cb.ax.get_xticklabels():
            t.set_fontsize(size)
    return



def shift_to_180_180(data,lon,lat=None,latmin=None,latmax=None,product=None):
    '''
    Shift a grid and data from 0:360 to -180:180

        lon,data = et.shift_to_180_180(data,lon)

    This routine often used in loops for opening many files, so
    the lat,latmin,latmax keywords can be useful to return one time only
    the j-range indices for lat, which can be used in subsequent nc reads
    to reduce memory usage.
    *args:
      lat     : 1- or 2-d field
      latmin  : latmin - 2-degree 'pad'
      latmax  : latmax + 2-degree 'pad'
      product : special treatment for particular products,
                currently AVISO
    '''
    ##assert np.any(lon[lon>=180]), 'z and q must be the same shape'
    if product=='AVISO_Med':
        lon -= 360.
        product = 'AVISO'
    else:
        lon[lon>=180.] -= 360.
    east,west = np.hsplit(lon, 2)
    lon = np.concatenate((west,east))
    east,west = np.vsplit(data,2)
    data = np.concatenate((west,east))
    if product=='AVISO': data = data.T
    ####
    if np.sometrue([lat,latmin,latmax]):
        if not np.alltrue([lat,latmin,latmax]):
            assert False, 'lat,latmin,latmax must either all be defined or none at all'
        else:
            latmin -= 2
            latmax += 2.
            if lat.ndim==1: jmin,jmax = nearest1d(latmin,lat),nearest1d(latmax,lat)
            else: jmin,jmax = nearest1d(latmin,lat[:,0]),nearest1d(latmax,lat[:,0])
        return lon,data,jmin,jmax
    ####
    return lon,data


def padmask(mask, n):
    '''
    Put an n-cell pad of zeros along
    a coastline
    '''
    mask = np.asarray(mask).copy()
    mask[1:-1,1:-1] = mask[ :-2,   :-2] * \
                      mask[ :-2,  1:-1] * \
                      mask[ :-2,  2:  ] * \
                      mask[1:-1,  :-2 ] * \
                      mask[1:-1, 1:-1 ] * \
                      mask[1:-1, 2:   ] * \
                      mask[2:,    :-2 ] * \
                      mask[2:,   1:-1 ] * \
                      mask[2:,   2:   ]
    mask[0] = mask[1]
    mask[-1] = mask[-2]
    mask[:,0] = mask[:,1]
    mask[:,-1] = mask[:,-2]
    return mask


def get_basemap_lm(M,x,y,pad=False):
    '''
    Get basemap landmask
      Inputs:
          M   : basemap instance
          x,y : 2d fields
          pad : pad the mask one point further into the domain
    '''
    mask = np.zeros_like(x)
    ind = 0
    for xi,yi in zip(x.ravel(),y.ravel()):
        mask.ravel()[ind] = M.is_land(xi,yi) 
        ind += 1
    if pad:
        mask = 1+(mask*-1)
        mask = padmask(mask)
        mask = 1+(mask*-1)
    return mask



def get_skip(datafile):
    '''
    Returns length of a header in an ascii file,
    needed for *skiprows in loadtxt and genfromtxt
    '''
    skip = 0 # no. of rows in header
    header = False
    while header==False:
        try:
            np.loadtxt(datafile,skiprows=skip)
            header = True
        except:
            skip += 1
    return skip


def cart2pol(x,y,deg=False):
    '''
    Taken from http://www.physics.rutgers.edu/~masud/computing/WPark_recipes_in_python.html
    -- Radian if deg==False; degree if deg==True
    -- Returns speed, direction
    -- Note: 0  is east;   west is +180
             90 is north; south is -90 
    -- use rot_minus_90(ang) to change to north centred
    '''
    if deg:
        return np.hypot(x,y), np.rad2deg(np.arctan2(y,x))
    else:
        return np.hypot(x,y), np.arctan2(y,x)
    
    
def rot_minus_90(ang):
    '''
    Shift from east-centred (-180-180) to north-centred (0-360)
    '''
    ang    = np.atleast_1d(ang).astype('float64')
    angout = ang.copy()
    if np.any(ang<0): angout[ang<0] = np.abs(ang[ang<0]-90.)
    if np.any(ang>=0):
        angout[ang>=0] = np.abs(ang[ang>=0]-90.)
        if np.any(ang>90.): angout[ang>90.] = np.abs(360-ang[ang>90.])+90
    return angout
    
    
##run centroid.py


#import pylab           as plt
#import numpy           as np


## Algorithm:
##  X0 = Int{x*ds}/Int{ds}, where ds - area element
##  so that Int{ds} is total area of a polygon.
##    Using Green's theorem the area integral can be 
##  reduced to a contour integral:
##  Int{x*ds} = -Int{x^2*dy}, Int{ds} = Int{x*dy} along
##  the perimeter of a polygon.
##    For a polygon as a sequence of line segments
##  this can be reduced exactly to a sum:
##  Int{x^2*dy} = Sum{ (x_{i}^2+x_{i+1}^2+x_{i}*x_{i+1})*
##  (y_{i+1}-y_{i})}/3;
##  Int{x*dy} = Sum{(x_{i}+x_{i+1})(y_{i+1}-y_{i})}/2.
##    Similarly
##  Y0 = Int{y*ds}/Int{ds}, where
##  Int{y*ds} = Int{y^2*dx} = 
##  = Sum{ (y_{i}^2+y_{i+1}^2+y_{i}*y_{i+1})*
##  (x_{i+1}-x_{i})}/3.


def centroid(x,y):
    '''
    % CENTROID Center of mass of a polygon.
    [X0,Y0] = CENTROID(X,Y) Calculates centroid 
    (center of mass) of planar polygon with vertices 
    coordinates X, Y.
    
    Algorithm:
    X0 = Int{x*ds}/Int{ds}, where ds - area element
    so that Int{ds} is total area of a polygon.
    Using Green's theorem the area integral can be 
    reduced to a contour integral:
    Int{x*ds} = -Int{x^2*dy}, Int{ds} = Int{x*dy} along
    the perimeter of a polygon.
      For a polygon as a sequence of line segments
    this can be reduced exactly to a sum:
    Int{x^2*dy} = Sum{ (x_{i}^2+x_{i+1}^2+x_{i}*x_{i+1})*
    (y_{i+1}-y_{i})}/3;
    Int{x*dy} = Sum{(x_{i}+x_{i+1})(y_{i+1}-y_{i})}/2.
      Similarly
    Y0 = Int{y*ds}/Int{ds}, where
    Int{y*ds} = Int{y^2*dx} = 
    = Sum{ (y_{i}^2+y_{i+1}^2+y_{i}*y_{i+1})*
    (x_{i+1}-x_{i})}/3.
    
    Taken from:
    http://puddle.mit.edu/~glenn/kirill/SAGA/centroid.m
    
    
    Note: TO DO FOR SPHERICAL CENTROID:
        http://www.jennessent.com/arcgis/shapes_poster.htm
    
    '''
    # Close the polygon
    x = np.append(x,x[0])
    y = np.append(y,y[0])

    # Check length
    l = x.size
    assert l==y.size,' Vectors x and y must have the same length'

    # X-mean: Int{x^2*dy}
    delta = np.diff(y) #y[1:] - y[:-1]
    v     = np.power(x[:-1],2) + np.power(x[1:],2) + x[:-1] * x[1:]
    x0    = np.dot(v,delta)
    
    # Y-mean: Int{y^2*dx}
    delta = np.diff(x)   #x[1:] - x[:-1]
    v     = np.power(y[:-1],2) + np.power(y[1:],2) + y[:-1] * y[1:]
    y0    = np.dot(v,delta)

    # Calculate area: Int{y*dx}
    a   = np.dot(y[:-1] + y[1:],delta)
    tol = 2 * np.finfo(float).eps
    if np.abs(np.any(a))<tol:
        print (' Warning: area of polygon is close to 0')
        a += np.sign(a) * tol + (np.equal(a,0))*tol

    # Multiplier
    a = 1./3./a

    # Divide by area
    x0 = -x0 * a
    y0 =  y0 * a
    return x0,y0

    
    
    
class horizInterp(interpnd.CloughTocher2DInterpolator):
    '''
    
    '''
    def __init__(self, tri, values, fill_value=np.nan,
                 tol=1e-6, maxiter=400):
        interpnd.NDInterpolatorBase.__init__(self, tri.points, values, ndim=2,
                                             fill_value=fill_value)
        self.tri = tri
        self.grad = interpnd.estimate_gradients_2d_global(self.tri, self.values,
                                                          tol=tol, maxiter=maxiter)
    
    
    
    
def transect(lontrans, lattrans, lon2d, lat2d, var, tri=None):
    '''
    
    '''
    points = np.array([lon2d.ravel(),
                       lat2d.ravel()]).T
    values = var.ravel()
    xi = np.array([lontrans, lattrans]).T
    if tri is None:
        tri = sp.Delaunay(points)
        trans = horizInterp(tri, values)
        return trans(xi), tri
    else:
        trans = horizInterp(tri, values)
        return trans(xi)
    

def get_rfactor(h):
    '''
    Return hi - hi-1 / hi + hi-1
    '''
    rx = np.diff(h, axis=1)
    ry = np.diff(h, axis=0)
    #rx = u2rho_2d(rx)
    #ry = v2rho_2d(ry)
    rpx = np.add(h[:,:-1], h[:,1:])
    rpy = np.add(h[:-1], h[1:])
    r = np.hypot(rx[:-1], ry[:,:-1])
    r /= np.hypot(rpx[:-1], rpy[:,:-1])
    return r

def rx1(zw):
    '''
    Calculate grid stiffness ratio, rx1 (Haney number)
    '''
    rx1 = 0
    rx1max = np.zeros_like(zw[0])
    for k in np.arange(1, zw.shape[0]):

        rx1x = np.abs(zw[k,:,1:] - zw[k,:,:-1] + zw[k-1,:,1:] - zw[k-1,:,:-1])
        rx1x /=      (zw[k,:,1:] + zw[k,:,:-1] - zw[k-1,:,1:] - zw[k-1,:,:-1])
        rx1x = u2rho_2d(rx1x)
        rx1y = np.abs(zw[k,1:] - zw[k,:-1] + zw[k-1,1:] - zw[k-1,:-1])
        rx1y /=      (zw[k,1:] + zw[k,:-1] - zw[k-1,1:] - zw[k-1,:-1])
        rx1y = v2rho_2d(rx1y)
        rx1_max = np.maximum(rx1x.max(), rx1y.max())
        if rx1_max > rx1:
            rx1 = rx1_max
        rx1max = np.maximum(rx1x, rx1max)
        rx1max = np.maximum(rx1y, rx1max)
    return rx1, rx1max


def flatten(l):
    '''
    http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python
    NOTE: returns a generator
    '''
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el


def loess_smooth_handmade(data, fc, step=1, t=None, t_final=None):
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Loess filtering of a time serie
    % 
    % data_smooth = loess_smooth_handmade(data, fc)
    %
    % IN:
    %       - data      : time series to filter
    %       - fc        : cut frequency
    % OPTIONS:
    %       - step      : step between two samples  (if regular)
    %       - t         : coordinates of input data (if not regular)
    %       - t_final   : coordinates of output data (default is t)
    %
    % OUT:
    %       - data_smooth : filtered data
    %
    % Written by R. Escudier (2018)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
  
    if t is None:
        t = np.arange(0, data.size * step, step) 
    
    
    if t_final is None:
        t_final = t
  
    # Remove NaNs
    id_nonan = np.where(~np.isnan(data))
    t = t[id_nonan]
    data = data[id_nonan]
  
    # Period of filter
    tau = 1 / fc
    
    # Initialize output vector
    data_smooth = np.ones(t_final.shape) * np.nan
 
    # Only compute for the points where t_final is in the range of t
    sx = np.where(np.logical_and(t_final >= t.min(), t_final <= t.max()))

    # Loop on final coordinates
    for i in sx[0]:
        # Compute distance between current point and the rest
        dn_tot = np.abs(t - t_final[i]) / tau
        # Select neighbouring points
        idx_weights = np.where(dn_tot < 1)
        n_pts = idx_weights[0].size
    
        # Only try to adjust the polynomial if there are at least 4 neighbors
        if n_pts > 3:
            dn = dn_tot[idx_weights]
            weights = 1 - dn * dn * dn
            weights **= 3
            # adjust a polynomial to these data points
            X = np.stack((np.ones((n_pts,)), t[idx_weights], t[idx_weights]**2)).T
            W = np.diag(weights)
            B = np.linalg.lstsq(np.dot(W, X), np.dot(W, data[idx_weights]))
            coeff = B[0]
            # Smoothed value is the polynomial value at this location
            data_smooth[i] = coeff[0] + coeff[1] * t_final[i] + coeff[2] * t_final[i]**2
        
    return data_smooth



def f_cor(lat):  # ;% Calculates the coriolis parameter (f)
    '''% Inputs
    % lat = [m,n]
    %
    % Outputs
     % f = coriolis parameter
    % b = Beta == df/dy

    function [f,b] = f_cor(lat);
    '''
    omega = 2 * np.pi / 86400.
    re = 6371000.

    f = 2 * omega * np.sin(np.deg2rad(lat))


    b = (2 * omega * np.cos(np.deg2rad(lat))) / re
    return f, b


def datetime_mtopy(datenum):
    '''
    Input
        The fractional day count according to datenum datatype in matlab
    Output
        The date and time as a instance of type datetime in python
    Notes on day counting
        matlab: day one is 1 Jan 0000
        python: day one is 1 Jan 0001
        hence a reduction of 366 days, for year 0 AD was a leap year
    https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
    https://stackoverflow.com/questions/36680402/typeerror-only-length-1-arrays-can-be-converted-to-python-scalars-while-plot-sh
    '''
    def f(x):
        return datetime.datetime.fromordinal(int(x) - 366)
    def g(x):
        return datetime.timedelta(days=x%1)
    f2 = np.vectorize(f)
    g2 = np.vectorize(g)
    ii = f2(datenum)
    ff = g2(datenum) #datetime.timedelta(days=datenum%1)
    return ii + ff



