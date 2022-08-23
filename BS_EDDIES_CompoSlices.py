import os
import numpy as np
import netCDF4

import xarray as xr
from glob import glob

runyear = "????"

directory_Z   = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"
directory_RHO = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"
files_Z       = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_"+runyear+"????_T.nc"))
files_RHO     = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_"+runyear+"????_T.nc"))

outdir        = directory_Z
outfname      = outdir+'CompoRotated_%s_Region_%s_%s.nc' ## % Z/RHO regionint aggtype

for region in range(5):
    print('Processing Region %s'%region)

    print(' .. Z files')
    # Z FILES
    for i, f in enumerate(files_Z):
        xa=xr.load_dataset(f)
        if int(xa.region.values[0]) != region:
            continue
        if i==0:
            xas = [xa.drop(['xcoord','ycoord','index'])]
        else:
            xas.append(xa.drop(['xcoord','ycoord','index']))

    xm=xr.concat(xas,'index')

    xmean  = xm.mean ('index')
    xstd   = xm.std  ('index')
    xcount = xm.count('index')

    xmean.to_netcdf( outfname %('Z', str(region),'mean'))
    xstd.to_netcdf(  outfname %('Z', str(region),'std'))
    xcount.to_netcdf(outfname %('Z', str(region),'count'))

    print(' .. RHO files')

# RHO FILES
    for i, f in enumerate(files_RHO):
        xa=xr.load_dataset(f)
        if int(xa.region.values[0]) != region:
            continue
        if i==0:
            xas = [xa.drop(['xcoord','ycoord','index'])]
        else:
            xas.append(xa.drop(['xcoord','ycoord','index']))

    xm=xr.concat(xas,'index')

    xmean  = xm.mean ('index')
    xstd   = xm.std  ('index')
    xcount = xm.count('index')

    xmean.to_netcdf(outfname %('RHO', str(region),'mean'))
    xstd.to_netcdf(outfname %('RHO', str(region),'std'))
    xcount.to_netcdf(outfname %('RHO', str(region),'count'))
