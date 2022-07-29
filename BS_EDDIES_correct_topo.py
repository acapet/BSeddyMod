# %run make_BS_correct_1d_topo.py

from numpy import array
import xarray as xr
from matplotlib import pyplot as plt
from glob import glob
from netCDF4 import Dataset

if __name__ == "__main__":

    plt.close("all")

    #savedir = "/Users/emason/Dropbox/BlackSea/figures_BS/"
    
    directory_Z = "~/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"
    directory_RHO = "~/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"

from numpy import array
import xarray as xr
from matplotlib import pyplot as plt
from glob import glob
from netCDF4 import Dataset

if __name__ == "__main__":

    plt.close("all")

    #savedir = "/Users/emason/Dropbox/BlackSea/figures_BS/"
    
    directory_Z = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"
    directory_RHO = "/home/arthur/Desktop/PAPERS_UNDER_WORK/Evan1/diagfiles/"
    
    files_Z = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_????????.nc"))
    files_RHO = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_????????.nc"))

    var3Dlonely = ['vort','w']
    var2Dlonely = ['mld','lon','lat','topo','ssh',]

    varphy3Dlist=['rho','temp','salt']
    varbio3Dlist=['NPPO','ZooResp','DOC','bac_oxygenconsumption','OXIDATIONBYDOX','CHL', 'DOX', 'NOS', 'POC', 'PAR', 'NHS', 'ODU']
    varbio2Dlist=['bac_oxygenconsumptionI', 'ZooRespI', 'NPPOI', 'OXIDATIONBYDOXI', 'AirSeaOxygenFlux']
           
    for i, file_Z in enumerate(files_Z):
        
        ds_Z = xr.open_dataset(file_Z) 
    
        topo = ds_Z.topo[0].values
        topo_start, topo_end = (topo[0], topo[-1])
           
        if topo_start <= topo_end:
            print("No flip needed")
            flip = False
        else:
            print("Need to flip")
            flip = True

        if flip:
            with Dataset(file_Z) as nc:
                print(file_Z)
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

            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            ds_RHO = xr.open_dataset(files_RHO[i])
            
            with Dataset(files_RHO[i]) as nc:
                print(files_RHO[i])

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
            
            #file_RHO = files_RHO[i].replace(".nc", "_T.nc")
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        #sasa
        
        else:
            #file_Z = file_Z.replace(".nc", "_T.nc")
            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            #file_RHO = files_RHO[i].replace(".nc", "_T.nc")
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        
    print("Done")
    

    
    
    
    
    
    
    
    
    #ds["vort"].plot(yincrease=False)
    #ds["topo"].plot(yincrease=False)
    
        
        
    plt.show()
        
    
    
    files_Z = sorted(glob(directory_Z + "BS_Z_slices_ACYC_track-????_????????.nc"))
    files_RHO = sorted(glob(directory_RHO + "BS_RHO_slices_ACYC_track-????_????????.nc"))
    
    #ds = xr.open_dataset(directory + "BS_Z_slices_ACYC_track-1641_20160918.nc") # Shallow to deep
    #ds = xr.open_dataset(directory_Z + "BS_Z_slices_ACYC_track-1642_20160919.nc") # Deep to shallow
    #sasa
    
    for i, file_Z in enumerate(files_Z):
        
        ds_Z = xr.open_dataset(file_Z) 
    
        topo = ds_Z.topo[0].values
        topo_start, topo_end = (topo[0], topo[-1])
    
        #print(topo_start, topo_end)
        
        if topo_start <= topo_end:
            print("No flip needed")
            flip = False
        else:
            print("Need to flip")
            flip = True

        if flip:
            with Dataset(file_Z) as nc:
                print(file_Z)
                ds_Z["topo"][:] = nc.variables['topo'][:, ::-1]
                ds_Z["mld"][:] = nc.variables['mld'][:, ::-1]
                ds_Z["ssh"][:] = nc.variables['ssh'][:, ::-1]
                ds_Z["vort"][:] = nc.variables['vort'][:, :, ::-1]
                ds_Z["rho"][:] = nc.variables['rho'][:, :, ::-1]
                ds_Z["rhoP"][:] = nc.variables['rhoP'][:, :, ::-1]
                ds_Z["temp"][:] = nc.variables['temp'][:, :, ::-1]
                ds_Z["tempP"][:] = nc.variables['tempP'][:, :, ::-1]
                ds_Z["salt"][:] = nc.variables['salt'][:, :, ::-1]
                ds_Z["saltP"][:] = nc.variables['saltP'][:, :, ::-1]
                ds_Z["w"][:] = nc.variables['w'][:, :, ::-1]
                ds_Z["dox"][:] = nc.variables['dox'][:, :, ::-1]
                ds_Z["doxP"][:] = nc.variables['doxP'][:, :, ::-1]
                ds_Z["npp"][:] = nc.variables['npp'][:, :, ::-1]
                ds_Z["nppP"][:] = nc.variables['nppP'][:, :, ::-1]
                ds_Z["nos"][:] = nc.variables['nos'][:, :, ::-1]
                ds_Z["nosP"][:] = nc.variables['nosP'][:, :, ::-1]
                ds_Z["pho"][:] = nc.variables['pho'][:, :, ::-1]
                ds_Z["phoP"][:] = nc.variables['phoP'][:, :, ::-1]
                ds_Z["par"][:] = nc.variables['par'][:, :, ::-1]
                ds_Z["parP"][:] = nc.variables['parP'][:, :, ::-1]
                ds_Z["poc"][:] = nc.variables['poc'][:, :, ::-1]
                ds_Z["pocP"][:] = nc.variables['pocP'][:, :, ::-1]
                ds_Z["bac"][:] = nc.variables['bac'][:, :, ::-1]
                ds_Z["bacP"][:] = nc.variables['bacP'][:, :, ::-1]
                ds_Z["chl"][:] = nc.variables['chl'][:, :, ::-1]
                ds_Z["chlP"][:] = nc.variables['chlP'][:, :, ::-1]

            #file_Z = files_Z[i].replace(".nc", "_T.nc")
            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            
            ds_RHO = xr.open_dataset(files_RHO[i])
            
            with Dataset(files_RHO[i]) as nc:
                print(files_RHO[i])
                ds_RHO["vort"][:] = nc.variables['vort'][:, :, ::-1]
                ds_RHO["rhoP"][:] = nc.variables['rhoP'][:, :, ::-1]
                ds_RHO["temp"][:] = nc.variables['temp'][:, :, ::-1]
                ds_RHO["tempP"][:] = nc.variables['tempP'][:, :, ::-1]
                ds_RHO["salt"][:] = nc.variables['salt'][:, :, ::-1]
                ds_RHO["saltP"][:] = nc.variables['saltP'][:, :, ::-1]
                ds_RHO["w"][:] = nc.variables['w'][:, :, ::-1]
                ds_RHO["dox"][:] = nc.variables['dox'][:, :, ::-1]
                ds_RHO["doxP"][:] = nc.variables['doxP'][:, :, ::-1]
                ds_RHO["npp"][:] = nc.variables['npp'][:, :, ::-1]
                ds_RHO["nppP"][:] = nc.variables['nppP'][:, :, ::-1]
                ds_RHO["nos"][:] = nc.variables['nos'][:, :, ::-1]
                ds_RHO["nosP"][:] = nc.variables['nosP'][:, :, ::-1]
                ds_RHO["pho"][:] = nc.variables['pho'][:, :, ::-1]
                ds_RHO["phoP"][:] = nc.variables['phoP'][:, :, ::-1]
                ds_RHO["par"][:] = nc.variables['par'][:, :, ::-1]
                ds_RHO["parP"][:] = nc.variables['parP'][:, :, ::-1]
                ds_RHO["poc"][:] = nc.variables['poc'][:, :, ::-1]
                ds_RHO["pocP"][:] = nc.variables['pocP'][:, :, ::-1]
                ds_RHO["bac"][:] = nc.variables['bac'][:, :, ::-1]
                ds_RHO["bacP"][:] = nc.variables['bacP'][:, :, ::-1]
                ds_RHO["chl"][:] = nc.variables['chl'][:, :, ::-1]
                ds_RHO["chlP"][:] = nc.variables['chlP'][:, :, ::-1]
            
            
            #file_RHO = files_RHO[i].replace(".nc", "_T.nc")
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        #sasa
        
        else:
            #file_Z = file_Z.replace(".nc", "_T.nc")
            ds_Z.to_netcdf(files_Z[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_Z.close()
            
            #file_RHO = files_RHO[i].replace(".nc", "_T.nc")
            ds_RHO.to_netcdf(files_RHO[i].replace(".nc", "_T.nc"), unlimited_dims='index', format="NETCDF4")
            ds_RHO.close()
        
    print("Done")
    

    
    
    
    
    
    
    
    
    #ds["vort"].plot(yincrease=False)
    #ds["topo"].plot(yincrease=False)
    
        
        
    plt.show()
        
    