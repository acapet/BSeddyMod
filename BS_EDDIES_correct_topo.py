# %run make_BS_correct_1d_topo.py

from numpy import array
import xarray as xr
from matplotlib import pyplot as plt
from glob import glob
from netCDF4 import Dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-y", "--year" , type=str, help="start year")

args = parser.parse_args()
runyear  = args.year

if __name__ == "__main__":

    plt.close("all")

    #savedir = "/Users/emason/Dropbox/BlackSea/figures_BS/"
    
    directory_Z = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/Z_slices/"
    directory_RHO = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/RHO_slices/"
    filedir='../files/'    
    files_Z = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_"+runyear+"????.nc"))
    files_RHO = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_"+runyear+"????.nc"))

    var3Dlonely = ['vort','w']
    var2Dlonely = ['mld','lon','lat','topo','ssh',]

    varphy3Dlist=['rho','temp','salt']
    varbio3Dlist=['NPPO','ZooResp','DOC','bac_oxygenconsumption','OXIDATIONBYDOX','CHL', 'DOX', 'NOS', 'POC', 'PAR', 'NHS', 'ODU']
    varbio2Dlist=['bac_oxygenconsumptionI', 'ZooRespI', 'NPPOI', 'OXIDATIONBYDOXI', 'AirSeaOxygenFlux']
           

    #For region assignement
    xreg  = xr.load_dataset(filedir+'../files/region_REVISITED.nc')
    xgrid = xr.load_dataset(filedir+'../files/mesh_mask_31levels.nc')
    xreg  = xreg.assign_coords({'x':('x',xgrid.nav_lon[0,:].values)})
    xreg  = xreg.assign_coords({'y':('y',xgrid.nav_lat[:,0].values)})

    for i, file_Z in enumerate(files_Z):

        #For region assignement    
        ds_Z = xr.open_dataset(file_Z) 

        newreg=xreg.region.interp({'x':ds_Z.centlon, 'y':ds_Z.centlat}).values
    
        topo = ds_Z.topo[0].values
        topo_start, topo_end = (topo[0], topo[-1])
           
        if topo_start <= topo_end:
            print("No flip needed %s"%file_Z)
            flip = False
        else:
            print("Need to flip %s"%file_Z)
            flip = True

        if flip:
            with Dataset(file_Z) as nc:
                for v in var2Dlonely:
                    ds_Z[v][:] = nc.variables[v][:, ::-1]
                for v in var3Dlonely:
                    ds_Z[v][:] = nc.variables[v][:, :, ::-1]
                for v in varphy3Dlist:
                    ds_Z[v][:] = nc.variables[v][:, :, ::-1]
                    ds_Z[v+"P"][:] = nc.variables[v+'P'][:, :, ::-1]
                for v in varbio3Dlist:
                    ds_Z[v][:] = nc.variables[v][:, :, ::-1]
                    ds_Z[v+"_P"][:] = nc.variables[v+'_P'][:, :, ::-1]
                for v in varbio2Dlist:
                    ds_Z[v][:] = nc.variables[v][:,  ::-1]
                    ds_Z[v+"_P"][:] = nc.variables[v+'_P'][:, ::-1]               

            ds_Z['region']=('one',newreg)
            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            ds_RHO = xr.open_dataset(files_RHO[i])
            
            with Dataset(files_RHO[i]) as nc:
                for v in var3Dlonely:
                    ds_RHO[v][:] = nc.variables[v][:, :, ::-1]
                for v in varphy3Dlist:
                    if v == 'rho':
                        continue
                    ds_RHO[v][:] = nc.variables[v][:, :, ::-1]
                    ds_RHO[v+"P"][:] = nc.variables[v+'P'][:, :, ::-1]
                for v in varbio3Dlist:
                    ds_RHO[v][:] = nc.variables[v][:, :, ::-1]
                    ds_RHO[v+"_P"][:] = nc.variables[v+'_P'][:, :, ::-1]
#                for v in varbio2Dlist:
#                    ds_RHO[v][:] = nc.variables[v][:,  ::-1]
#                    ds_RHO[v+"_P"][:] = nc.variables[v+'_P'][:, ::-1]             
            
            ds_RHO['region']=('one',newreg)
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        #sasa
        
        else:
            #file_Z = file_Z.replace(".nc", "_T.nc")
            ds_Z['region']=('one',newreg)
            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            #file_RHO = files_RHO[i].replace(".nc", "_T.nc")
            ds_RHO['region']=('one',newreg)
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        
    print("Done")
    
