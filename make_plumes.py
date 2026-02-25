"""
Auto-generate plots for 'How long could volcanic plumes persist in the Venus atmosphere?'
(Cohen et al. 2026).

Usage:
    python make_plumes.py

No inputs required as filepaths are specified in the script.
"""

# %% Filepaths
plumes = '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.5nocl.nc'
sens_tests = ['/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.4.nc',
              '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.3.nc',
              '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.2.nc',
              '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.1.nc',
            '/exomars/data/internal/working/mc5526/VPCM_plumes/Xins1.03nocl.nc']
surf = '/exomars/data/internal/working/mc5526/VPCM_age_of_air/surf_96x96x50/Xins_211to220.nc'
chem_dir = '/exomars/data/internal/simulations/venus/VPCM_chemistry_withSO2sink_2025/data/'
chem = [chem_dir + 'Xins4.nc',  chem_dir + 'Xins5.nc', chem_dir + 'Xins6.nc', chem_dir + 'Xins7.nc']
savedir = '/exomars/projects/mc5526/VPCM_volcanic_plumes/figures/'
filetype = 'pdf'

# %%
# Import packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
# Import local
from classes import PlumeSim
from plotting_functions import zmage, dispersal_time, dispersal_map, animate_chem_plume, summ_stats, sensitivity_test, plume_cross_section
from tools import venusdict, plume_dict, name_dict

# %% Main code block
if __name__ == "__main__":

    # Create PlumeSim object and load data for surface run
    surf_sim = PlumeSim(venusdict, None, 'aoa_surf')
    surf_sim.load_file(surf)
    surf_sim.set_resolution()

    # Figure 1: Zonal mean age of air
    zmage(surf_sim, hmin=0, hmax=None, time_slice=-1,
          convert2yr=True, plume_markers=plume_dict, 
          levels=np.arange(0,50,1), savepath=savedir,
          save=True, sformat=filetype, savename='fig1' + '.' + filetype)
    surf_sim.close()
    del surf_sim

    # Create PlumeSim object and load data for chemistry run
    chem_sim = PlumeSim(venusdict, None, 'chem_4days')
    chem_sim.load_file(chem)
    chem_sim.set_resolution()
    
    # Figure 5: Background variability of H2O at 8 km
    summ_stats(chem_sim, keys='h2o', lev=10, t0=0, tf=None,
               savename='fig5' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    # Figure 8: Background variability of H2O and HCl at 20 km
    summ_stats(chem_sim, keys=['h2o', 'hcl'], lev=14, t0=0, tf=None,
               savename='fig8' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)
    # Figure 11: Background variability of four gases at 35 km
    summ_stats(chem_sim, keys=['h2o', 'hcl', 'co', 'ocs'], lev=18, t0=0, tf=None,
               savename='fig11' + '.' + filetype,
               savepath=savedir,
               save=True, sformat=filetype)

    chem_sim.close()
    del chem_sim
    
    # Create PlumeSim object and load data for plume run
    plume_sim = PlumeSim(venusdict, plume_dict, 'scale_1.5')
    plume_sim.load_file(plumes)
    plume_sim.set_resolution()

    # Figure 2: Plume cross section of H2O at 35 km
    plume_cross_section(plume_sim, key='h2o', lev=18, times=[54,104,174],
                        save=True, savename='fig2' + '.' + filetype,
                        savepath=savedir, sformat=filetype)

    # Figure 4: Dispersal map of H2O plume at 8 km
    dispersal_map(plume_sim, lev=10, keys=['h2o'], labels=name_dict, save=True, savename='fig4' + '.' + filetype, sformat=filetype, savepath=savedir)
    # Figure 7: Dispersal map H2O and HCl plumes at 20 km
    dispersal_map(plume_sim, lev=14, keys=['h2o', 'hcl'], labels=name_dict, save=True, savename='fig7' + '.' + filetype, sformat=filetype, savepath=savedir)
    # Figure 10: Dispersal map of four gas plumes at 35 km
    dispersal_map(plume_sim, lev=18, keys=['h2o', 'hcl', 'co', 'ocs'], labels=name_dict, save=True, savename='fig10' + '.' + filetype, sformat=filetype, savepath=savedir)
    # Figure 12: Animation + cover image of H2O, HCl, CO, OCS plumes at 35 km
    animate_chem_plume(plume_sim, lev=18, keys=['h2o', 'hcl', 'co', 'ocs'], labels=name_dict, t0=0, tf=500, savepath=savedir, savename='fig12', snapshot=100)
    # Figure 13: Animation + cover image of SO2 plume at 70 km
    #animate_chem_plume(plume_sim, lev=35, keys=['so2'], t0=0, tf=250, qscale=5, savepath=savedir, savename='fig13', snapshot=100)
    
    # Figure 3: Dispersal time of H2O plumes at 8 km
    dispersal_time(plume_sim, lev=10, keys=['h2o'], lats=[49,82], lons=[92,47], 
                   labels=name_dict, axis_len=500,
                   save=True, savename='fig3' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    # Figure 6: Dispersal time of H2O and HCl plumes at 20 km
    dispersal_time(plume_sim, lev=14, keys=['h2o', 'hcl'], lats=[49,82], lons=[92,47], 
                   labels=name_dict, axis_len=500,
                   save=True, savename='fig6' + '.' + filetype, sformat=filetype,
                   savepath=savedir)
    # Figure 9: Dispersal time of four gas plumes at 35 km
    dispersal_time(plume_sim, lev=18, keys=['h2o', 'hcl', 'co', 'ocs'], lats=[49,82], lons=[92,47], 
                   labels=name_dict, axis_len=500,
                   save=True, savename='fig9' + '.' + filetype, sformat=filetype,
                   savepath=savedir)

    plume4 = PlumeSim(venusdict, plume_dict, 'scale_1.4')
    plume4.load_file(sens_tests[0])
    plume4.set_resolution()

    plume3 = PlumeSim(venusdict, plume_dict, 'scale_1.3')
    plume3.load_file(sens_tests[1])
    plume3.set_resolution() 

    plume2 = PlumeSim(venusdict, plume_dict, 'scale_1.2')
    plume2.load_file(sens_tests[2])
    plume2.set_resolution()

    plume1 = PlumeSim(venusdict, plume_dict, 'scale_1.1')
    plume1.load_file(sens_tests[3])
    plume1.set_resolution()

    plume0 = PlumeSim(venusdict, plume_dict, 'scale_1.03')
    plume0.load_file(sens_tests[4])
    plume0.set_resolution()

    # Figure 13: Sensitivity test for H2O plume dispersal time at three altitudes
    sens_vals = sensitivity_test(plume5=plume_sim, plume4=plume4, plume3=plume3,
                     plume2=plume2, plume1=plume1, plume0=plume0,
                     labels=name_dict, levs=[10,14,18], key='h2o',
                     save=True, savename='fig13' + '.' + filetype,
                     savepath=savedir, sformat=filetype)
    print(sens_vals)

    for plume in [plume_sim, plume4, plume3, plume2, plume1]:
        plume.close()
        del plume