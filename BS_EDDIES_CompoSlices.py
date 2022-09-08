import os
import numpy as np
import netCDF4

import xarray as xr
from glob import glob

import matplotlib.dates as mdates
old_epoch = "0000-12-31T00:00:00"
mdates.set_epoch(old_epoch)

runyear = "2014"

directory_Z   = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/Z_slices2/"
directory_RHO = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/RHO_slices2/"
files_Z       = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_"+runyear+"????_T.nc"))
files_RHO     = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_"+runyear+"????_T.nc"))

outdir        = directory_Z+'/../CompoSlices/'
outfname      = outdir+'CompoRotated_%s_Region_%s_%s_%s.nc' ## % Z/RHO regionint aggtype

seasonfilter  =  'ALLYEAR' # 'SUMMER' #


for region in range(5):
    print('Processing Region %s'%region)

    print(' .. Z files')
    # Z FILES
    if True:
        xas=[]
        for i, f in enumerate(files_Z):
            
            xa=xr.load_dataset(f)
            if int(xa.region.values[0]) != region:
                continue
            # Get date
            ddate= mdates.num2date(xa.time.data)[0]
            if i%100==0:
                print(i,f,xa.region.values[0])
            if (seasonfilter == 'SUMMER') & (ddate.month not in [4,5,6,7,8,9,10]):
#                print( 'month %s discarded'%ddate.month)
                continue
#            else:
#                print( 'month %s is kept '%ddate.month)
            xas.append(xa.drop(['xcoord','ycoord','index']))

        xm=xr.concat(xas,'index')

        xmean  = xm.mean ('index')
        xstd   = xm.std  ('index')
        xcount = xm.count('index')

        xmean.to_netcdf( outfname %('Z', str(region),'mean' ,seasonfilter))
        xstd.to_netcdf(  outfname %('Z', str(region),'std'  ,seasonfilter))
        xcount.to_netcdf(outfname %('Z', str(region),'count',seasonfilter))

        xmedian  = xm.median ('index')
        xmedian.to_netcdf(outfname %('Z', str(region),'median',seasonfilter))
        
        xp25    = xm.quantile(.25,'index')
        xp25.to_netcdf(outfname %('Z', str(region),'p25',seasonfilter))
        
        xp75  = xm.quantile (.75,'index')
        xp75.to_netcdf(outfname %('Z', str(region),'p75',seasonfilter))


    print(' .. RHO files')

    # RHO FILES
    if False: 
        xas=[]
        for i, f in enumerate(files_RHO):
            xa=xr.load_dataset(f)
            if i%100==0: print(i,f,xa.region.values[0])
            if int(xa.region.values[0]) != region:
                continue
            # Get date
            ddate= mdates.num2date(xa.time.data)[0]
            if i%100==0:
                print(i,f,xa.region.values[0])
                print(ddate.month)
            if (seasonfilter == 'SUMMER') & (ddate.month not in [4,5,6,7,8,9,10]):
                continue
            xas.append(xa.where((xa.rho>=12)&(xa.rho<=18), drop=True).drop(['xcoord','ycoord','index']))

        xm=xr.concat(xas,'index')

        xmean  = xm.mean ('index')
        xmean.to_netcdf(outfname %('RHO', str(region),'mean',seasonfilter))

        xstd   = xm.std  ('index')
        xstd.to_netcdf(outfname %('RHO', str(region),'std',seasonfilter))

        xmedian  = xm.median ('index')
        xmedian.to_netcdf(outfname %('RHO', str(region),'median',seasonfilter))
        xp25    = xm.quantile(.25,'index')
        xp25.to_netcdf(outfname %('RHO', str(region),'p25',seasonfilter))
        xp75  = xm.quantile (.75,'index')
        xp75.to_netcdf(outfname %('RHO', str(region),'p75',seasonfilter))

        xcount = xm.count('index')
        xcount.to_netcdf(outfname %('RHO', str(region),'count',seasonfilter))
