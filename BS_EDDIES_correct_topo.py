# %run make_BS_correct_1d_topo.py

from numpy import array, deg2rad
import xarray as xr
from matplotlib import pyplot as plt
from glob import glob
from netCDF4 import Dataset
import argparse
import pyproj
import math

geodesic = pyproj.Geod(ellps='WGS84')

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year" , type=str, help="start year")

args = parser.parse_args()
runyear  = args.year

def completemyd(dd,dtype):
    dd['BGC']      = dd['NPPO']      - dd['bac_oxygenconsumption']  - dd['OXIDATIONBYDOX']   - dd['ZooResp']
    dd['BGC_P']    = dd['NPPO_P']    - dd['bac_oxygenconsumption_P']- dd['OXIDATIONBYDOX_P'] - dd['ZooResp_P']

    dd['Resp']     = dd['ZooResp']   + dd['bac_oxygenconsumption']
    dd['Resp_P']   = dd['ZooResp_P'] + dd['bac_oxygenconsumption_P']

    dd['DOXODU']   = dd['DOX']       - dd['ODU']
    dd['DOXODU_P'] = dd['DOX_P']     - dd['ODU_P']
    
    if (dtype=='z'):
        dd['BGCI']      = dd['NPPOI']      - dd['bac_oxygenconsumptionI']  - dd['OXIDATIONBYDOXI']   - dd['ZooRespI']
        dd['BGCI_P']    = dd['NPPOI_P']    - dd['bac_oxygenconsumptionI_P']- dd['OXIDATIONBYDOXI_P'] - dd['ZooRespI_P'] 
        dd['NetI']      = dd['BGCI']       + dd['AirSeaOxygenFlux'] 
        dd['NetI_P']    = dd['BGCI_P']     + dd['AirSeaOxygenFlux_P']
    return(dd)


var3Dlonely = ['vort','w','u','v']
var2Dlonely = ['mld','lon','lat','topo','ssh',]

varphy3Dlist=['rho','temp','salt']
varbio3Dlist=['NPPO','ZooResp','DOC','bac_oxygenconsumption','OXIDATIONBYDOX','CHL', 'DOX', 'NOS', 'POC', 'PAR', 'NHS', 'ODU']
varbio2Dlist=['bac_oxygenconsumptionI', 'ZooRespI', 'NPPOI', 'OXIDATIONBYDOXI', 'AirSeaOxygenFlux']

def flipmeifneeded(xl, flip,ztype):

    if flip:
        if ztype=='z':
            for v in var2Dlonely:
                xl[v][:] = xl[v].values[:, ::-1]
            for v in varbio2Dlist:
                xl[v][:] = xl[v].values[:,  ::-1]
                xl[v+"_P"][:] = xl[v+'_P'].values[:, ::-1]
        for v in var3Dlonely:
            xl[v][:] = xl[v].values[:, :, ::-1]
        for v in varphy3Dlist:
            if (v=='rho') & (ztype=='rho'):
                continue
            xl[v][:] = xl[v].values[:, :, ::-1]
            xl[v+"P"][:] = xl[v+'P'].values[:, :, ::-1]
        for v in varbio3Dlist:
            xl[v][:] = xl[v].values[:, :, ::-1]
            xl[v+"_P"][:] = xl[v+'_P'].values[:, :, ::-1]
           
    return(xl)

if __name__ == "__main__":

    plt.close("all")

    #savedir = "/Users/emason/Dropbox/BlackSea/figures_BS/"
    
    directory_Z = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/Z_slices2/"
    directory_RHO = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/RHO_slices2/"
    filedir='../files/'    
    files_Z = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_"+runyear+"????.nc"))
    files_RHO = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_"+runyear+"????.nc"))

           
    ######  For region assignement  ################################
    xreg  = xr.load_dataset(filedir+'../files/region_REVISITED.nc')
    xgrid = xr.load_dataset(filedir+'../files/mesh_mask_31levels.nc')
    xreg  = xreg.assign_coords({'x':('x',xgrid.nav_lon[0,:].values)})
    xreg  = xreg.assign_coords({'y':('y',xgrid.nav_lat[:,0].values)})
    #################################################################

    for i, file_Z in enumerate(files_Z):

        ds_Z = xr.open_dataset(file_Z) 

        ######  For region assignement  ################################
        newreg=xreg.region.interp({'x':ds_Z.centlon, 'y':ds_Z.centlat}, method='nearest').values
        #################################################################

        topo = ds_Z.topo[0].values
        topo_start, topo_end = (topo[0], topo[-1])
           
        if topo_start <= topo_end:
            print("No flip needed %s"%file_Z)
            flip = False
        else:
            print("Need to flip %s"%file_Z)
            flip = True

        ds_Z= flipmeifneeded(ds_Z,flip,'z')

        ## 
        lon_away = ds_Z.lon.interp({'norm_eddy_radius':2.5}).values
        lat_away = ds_Z.lat.interp({'norm_eddy_radius':2.5}).values
        lon_cen  = ds_Z.centlon.values
        lat_cen  = ds_Z.centlat.values
   
        fwd_azimuth,back_azimuth,distance = geodesic.inv(lon_cen,lat_cen,lon_away,lat_away)
        fwd_azimuth=fwd_azimuth[0]

        ds_Z['v_across']= ds_Z['u']*math.cos(deg2rad(fwd_azimuth)) - ds_Z['v']*math.sin(deg2rad(fwd_azimuth))
        ds_Z['v_along'] = ds_Z['u']*math.sin(deg2rad(fwd_azimuth)) + ds_Z['v']*math.cos(deg2rad(fwd_azimuth))
        ##

        ds_Z['region']=('one',newreg)
        ds_Z = completemyd (ds_Z,'z')
        ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
        ds_Z.close()

        if True:    
            ds_RHO = xr.open_dataset(files_RHO[i])
                
            ds_RHO= flipmeifneeded(ds_RHO,flip,'rho')

            ds_RHO['v_across']= ds_RHO['u']*math.cos(deg2rad(fwd_azimuth)) - ds_RHO['v']*math.sin(deg2rad(fwd_azimuth))
            ds_RHO['v_along'] = ds_RHO['u']*math.sin(deg2rad(fwd_azimuth)) + ds_RHO['v']*math.cos(deg2rad(fwd_azimuth))

            ds_RHO['region']=('one',newreg)
            ds_RHO = completemyd (ds_RHO,'rho')
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        
    print("Done")
    
