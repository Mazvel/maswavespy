# -*- coding: utf-8 -*-
"""
Test MASWavesPy inversion module

This example covers the use of InvertDC objects to:
- Initialize an inversion object. 
- Initialize the inversion routine. The inversion is conducted using 
  Monte Carlo sampling as described in Olafsdottir et al. (2020) with 
  the Fast delta matrix algorithm used for forward modelling (i.e., 
  computation of theoretical dispersion curves).
- Pickle the inversion object (save the inversion results to disk).
- Post-processing of the inversion results:
  - View all shear wave velocity profiles whose associated dispersion curves
    fall within the boundaries of the experimental data at all frequencies.
  - Compute the median (or mean) shear wave velocity profile and the
    associated theoretical dispersion curve.
  - Identify the 10 lowest-misfit shear wave velocity profiles whose associated
    dispersion curves fall within the boundaries of the experimental data.
  - Compute the time-averaged shear wave velocity down to a depth z. 

Input files (prepared):
- Data/Oysand_dc.txt (Experimental dispersion curve with upper/lower boundary curves)
- Data/Oysand_initial.csv (Initial values for soil model parameters for use in inversion)

Outputs:
- inv_TestSite: Initialized inversion object (type inversion.InvertDC).
- Plot showing the initial shear wave velocity (Vs) profile and comparing the 
  corresponding theoretical dispersion curve (DC) to the experimental data.
- Set of sampled Vs profiles and corresponding theoretical DCs, saved to the InvertDC object inv_TestSite.
- Plot showing sampled Vs profiles and corresponding theoretical DCs. 
- InvertDC object inv_TestSite saved to disk using Python's Pickle module.
- Plot showing accepted Vs profiles and corresponding theoretical DCs. 
- TestSite_median_profile: Median and percentile values of the accepted Vs profiles (type dict).
- Plot showing the median and percentile Vs values and comparing the theoretical DC of 
  the median profile to the experimental data.
- lowest_misfit_profiles: Set of 'no_profiles' lowest-misfit Vs profiles retrieved 
  in the inversion (type dict).
- Plot showing the set of lowest-misfit Vs profiles and comparing the corresponding 
  theoretical DCs to the experimental data.
- Time-averaged Vs (Vsz) values computed for z=5 m, z=10 m, z=20 m and z=30 m.
  - Vsz_median: Vsz computed from the median of the accepted Vs profile (type tuple).
  - Vsz_lowest_misfit: Vsz computed from the lowest-misfit Vs profile (type tuple).
  
References
----------
Fast delta matrix algorithm
 - Buchen, P.W. & Ben-Hador, R. (1996). Free-mode surface-wave computations. 
   Geophysical Journal International, 124(3), 869–887. 
   https://doi.org/10.1111/j.1365-246X.1996.tb05642.x 
Inversion scheme 
 - Olafsdottir, E.A., Erlingsson, S. & Bessason, B. (2020). Open-Source 
   MASW Inversion Tool Aimed at Shear Wave Velocity Profiling for Soil Site 
   Explorations. Geosciences, 10(8), 322. https://doi.org/10.3390/geosciences10080322
    
"""
from maswavespy import inversion

import pandas as pd
import numpy as np

# Path to sample data
# Experimental dispersion curve (stored as a .txt file in the folder Data)
filename_dc = 'Data/Oysand_dc.txt' 
# Initial values for soil model parameters (stored as a .csv file in the folder Data)
filename_initial = 'Data/Oysand_initial.csv' 

# Import experimental dispersion curves
wavelengths = []
c_mean = []; c_low = []; c_up = []
with open(filename_dc, 'r') as file_dc:
    next(file_dc) # Skip the header
    for value in file_dc.readlines():
        wavelengths.append(float(value.split()[0]))
        c_mean.append(float(value.split()[1]))
        c_low.append(float(value.split()[2]))
        c_up.append(float(value.split()[3]))
wavelengths = np.array(wavelengths, dtype='float64')
c_mean = np.array(c_mean, dtype='float64'); 
c_low = np.array(c_low, dtype='float64')
c_up = np.array(c_up, dtype='float64')
    
# Import initial soil model parameters
initial_parameters = pd.read_csv(filename_initial)
h = np.array(initial_parameters['h [m]'].values[0:-1], dtype='float64')
n = int(len(h))
Vs = np.array(initial_parameters['Vs [m/s]'].values, dtype='float64')
rho = np.array(initial_parameters['rho [kg/m3]'].values, dtype='float64')
Vp = []
n_unsat = 0; nu = None
for item in range(len(initial_parameters['saturated/unsaturated'].values)):
    if initial_parameters['saturated/unsaturated'].values[item] == 'unsat':
        nu = initial_parameters['nu [-]'].values[item]
        Vp.append(np.sqrt((2*(1-nu))/(1-2*nu))*Vs[item])
        n_unsat = n_unsat + 1
    else:
        Vp.append(initial_parameters['Vp [m/s]'].values[item])
Vp = np.array(Vp, dtype='float64')

# Print message to user
print('The sample dispersion curve has been imported.')
print('The initial soil model parameters have been imported.')

#%%
# Initialize an inversion object.    
site = 'Oysand'
profile = 'P1'
inv_TestSite = inversion.InvertDC(site, profile, c_mean, c_low, c_up, wavelengths)

# Print message to user
print('An inversion (InvertDC) object has been initialized.') 

#%%
# Initialize the inversion routine. The inversion is conducted using 
# Monte Carlo sampling as described in Olafsdottir et al. (2020).

# Range for testing phase velocity
c_min = 50; c_max = 300; c_step = 0.1; delta_c = 3
c_test = {'min' : c_min, 
          'max' : c_max,
          'step' : c_step,
          'delta_c' : delta_c}

# Initial model parameters
initial = {'n' : n,
           'n_unsat' : n_unsat,
           'alpha' : Vp,
           'nu_unsat' : 0.35,
           'alpha_sat' : 1500,
           'beta' : Vs,
           'rho' : rho,
           'h' : h,
           'reversals' : 0}

# Inversion algorithm settings. See further in Olafsdottir et al. (2020).
settings = {'run' : 20,
            'bs' : 5,
            'bh' : 10,
            'N_max' : 1000}        

# View the initial shear wave velocity profile.
# Compute the associated dispersion curve and show relative to the experimental
# data. The misfit value is printed to the screen.
max_depth = 16.5
inv_TestSite.view_initial(initial, max_depth, c_test, col='crimson', DC_yaxis='linear', 
                 fig=None, ax=None, figwidth=16, figheight=12, return_ct=False)

# Print message to user
print('The initial estimate of the Vs profile and the corresponding theoretical DC have been plotted.')

#%%
# Start the inversion analysis (optimization) process.
print('Inversion initiated.')
inv_TestSite.mc_inversion(c_test, initial, settings)

# Plot sampled Vs profiles and associated dispersion curves
inv_TestSite.plot_sampled(max_depth, runs='all', figwidth=16, figheight=12, col_map='viridis', 
                  colorbar=True, DC_yaxis='linear', return_axes=False, show_exp_dc=True)

# Print message to user
print('All runs completed.')
print('The sampled Vs profiles and the corresponding theoretical DCs have been plotted.')

#%%
# Pickle the inversion object
file = 'Oysand_inversion'
inv_TestSite.save_to_pickle(file)

# Print message to user
print('The InvertDC object has been saved to disk as ' + file + '.p using pickle.')

#%%
# Post-processing 
#
# Plot sampled Vs profiles whose associated dispersion curves fall within the
# boundaries defined by c_low and c_up at all wavelengths
inv_TestSite.plot_within_boundaries(max_depth, show_all=True, runs='all', figwidth=16, figheight=12, 
                           col_map='viridis', colorbar=True, DC_yaxis='linear', return_axes=False)

# Print message to user
print('The set of accepted Vs profiles and the corresponding theoretical DCs have been plotted.')

#%%
# Post-processing 
#
# Compute and plot the median shear wave velocity profile (defined in terms of
# shear wave velocity and depth of layer interfaces) and the 90-th percentiles 
# of each parameter. The associated theoretical dispersion curve is also 
# computed and shown relative to the experimetnal data. 
# The mean shear wave velocity profile can be obtained in a comparable way 
# using the inv_TestSite.mean_profile method. (See further in the documentation 
# of inversion.py.)

percentiles = [10,90]
TestSite_median_profile = inv_TestSite.median_profile(q=percentiles, dataset='selected')
fig, ax = inv_TestSite.plot_profile(TestSite_median_profile, max_depth, c_test, initial, 
                                    col='red', up_low=True, fig=None, ax=None, 
                                    return_axes=True, return_ct=False)

# Print message to user
print('The median of accepted Vs profiles has been computed.')
print('The median profile and the corresponding theoretical DC have been plotted.')

#%%
# Post-processing 
#
# Get and plot the 'no_profiles' shear wave velocity profiles (here 10) that show the lowest value 
# of the dispersion misfit function and whose associated dispersion curves 
# fall within the boundaries specified for the experimental disperion curve. 

lowest_misfit_profiles = {}
no_profiles = 10
# Ensure that at least no_profiles fall within the experimental DC boundaries
no_profiles_checked = min(no_profiles, len(inv_TestSite.selected['beta']))
for no in range(-1*no_profiles_checked,0):
    profile_dict = {'beta': inv_TestSite.selected['beta'][no], 'z': inv_TestSite.selected['z'][no]}
    if no == -1*no_profiles_checked:
        fig, ax = inv_TestSite.plot_profile(profile_dict, max_depth, c_test, initial, col='gray', 
                              up_low=False, DC_yaxis='linear', fig=None, ax=None, return_axes=True, show_legend=True)
    else:
        inv_TestSite.plot_profile(profile_dict, max_depth, c_test, initial, col='gray', 
                              up_low=False, DC_yaxis='linear', fig=fig, ax=ax, show_legend=False)
    lowest_misfit_profiles[no] = profile_dict

# Print message to user
print('The ' + str(no_profiles) + ' lowest-misfit Vs profiles have been identified.')
print('The set of lowest-misfit profiles and the corresponding theoretical DCs have been plotted')

#%%
# Post-processing 
#
# Compute the average shear wave velocity (Vsz) for the top most z=5 m, z=10 m, z=20 m and z=30 m
# using (i) the median Vs profile and (ii) the lowest-misfit Vs profile.

depth = [5, 10, 20, 30]
layer_parameter = 'z'

# Median Vs profile
Vsz_median = inv_TestSite.compute_vsz(depth, TestSite_median_profile['beta'], 
                                      TestSite_median_profile['z'], layer_parameter)
print('Median Vs profile, z and Vsz values')
print(Vsz_median[0]) # Depths (z)
print([round(val, 2) for val in Vsz_median[1]]) # Computed Vsz values

# Lowest-misfit Vs profile
Vsz_lowest_misfit = inv_TestSite.compute_vsz(depth, inv_TestSite.selected['beta'][-1], 
                                             inv_TestSite.selected['z'][-1], layer_parameter)
print('Lowest-misfit Vs profile, z and Vsz values')
print(Vsz_lowest_misfit[0]) # Depths (z)
print([round(val, 2) for val in Vsz_lowest_misfit[1]]) # Computed Vsz values
