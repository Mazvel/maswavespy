# -*- coding: utf-8 -*-
#
#    MASWavesPy, a Python package for processing and inverting MASW data
#    Copyright (C) 2023  Elin Asta Olafsdottir (elinasta(at)hi.is)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MASWavesPy inversion

Invert experimental dispersion curves using a guided Monte Carlo search procedure 
(Olafsdottir et al., 2020). Computations of theoretical dispersion curves are based 
on the fast delta matrix algorithm (Buchen and Ben-Hador, 1996).

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import copy
import pickle

from maswavespy import supplemental as s

# Computation of theoretical dispersion curves (cython optimization)
from maswavespy import cy_theoretical_dc as t_dc


class InvertDC():
    
    """
    Class for inverting experimental dispersion curves. 
    
    Instance attributes
    -------------------
    General site attributes:
    site : str
        Name of test site.
    profile : str 
        Identification code for measurement profile.
    metadata : str or dict
        Meta-information for object.    
    
    Experimental dispersion curve:
    c_obs : numpy.ndarray
        Rayleigh wave phase velocity array [m/s].
    c_obs_low : numpy.ndarray
        Rayleigh wave phase velocity array, lower boundary values [m/s].     
    c_obs_up : numpy.ndarray
        Rayleigh wave phase velocity array, upper boundary values [m/s].
    wavelength : numpy.ndarray
        Wavelength array [m].
    
    Shear wave velocity profiles/theoretical dispersion curves:
    profiles : dict
        Dictionary containing sampled shear wave velocity profiles and 
        corresponding theoretical dispersion curves. The keys of profiles are 
        the following: 'n_layers', 'beta', 'alpha', 'h', 'c_t', 'misfit'.    
    selected : dict
        Dictionary containing sampled shear wave velocity profiles whose 
        theoretical dispersion curves fall within the upper/lower boundaries 
        of the experimental data. The keys of selected are the following: 
        'n_layers', 'beta', 'alpha', 'h', 'z', 'c_t', 'misfit'.   
        
        
    Instance methods
    ----------------
    compute_vsz(self, depth, beta_vec, layers, layer_parameter)
        Compute the average shear wave velocity (Vsz) of the top-most 'depth' meters.
    
    mc_initiation(self, c_vec, c_step, delta_c, n, n_unsat, alpha_initial, nu_unsat, 
                 alpha_sat, beta_initial, rho, h_initial, reversals, bs, bh, N_max)
        Monte Carlo based inversion for shear wave velocity and layer thicknesses, 
        single initiation of the sampling scheme.
        
    mc_inversion(self, c_test, initial, settings)
        Monte Carlo based inversion for shear wave velocity and layer thicknesses. 
        The sampling scheme is initiated ´run´ number of times starting from the 
        initially specified layered soil model.
    
    mean_profile(self, stdev=True, no_stdev=1, dataset='selected')    
        Return the mean shear wave velocity profile and (optional) upper and 
        lower boundaries that correspond to plus/minus no_stdev standard 
        deviations from the mean value.
    
    median_profile(self, perc=True, q=[10, 90], dataset='selected')
        Return the median shear wave velocity profile and (optional) the q-th
        percentiles of each parameter (shear wave velocity, depth of layer 
        interfaces).
    
    misfit(self, c_t)
        Evaluate dispersion misfit.
        
    plot_profile(self, profile, max_depth, c_test, initial, col='crimson', up_low=False, 
                     DC_yaxis='linear', fig=None, ax=None, return_axes=False, 
                     return_ct=False, figwidth=16, figheight=12)
        Plot the median or mean shear wave velocity profile and the associated 
        theoretical dispersion curve.
    
    plot_sampled(self, max_depth, runs='all', figwidth=16, figheight=12, 
                 col_map='viridis', colorbar=True, DC_yaxis='linear', 
                 return_axes=False, show_exp_dc=True)
        Plot sampled shear wave velocity profiles and the corresponding  
        theoretical dispersion curves. 
    
    plot_within_boundaries(self, max_depth, show_all=True, runs='all', figwidth = 16, 
                           figheight = 12, col_map='viridis', colorbar=True, 
                           DC_yaxis='linear', return_axes=False, **kwargs)
        Plot sampled profiles whose theoretical dispersion curves fall 
        within the upper and lower boundaries of the experimental data. 
    
    save_to_pickle(self, file)
        Pickle an inversion analysis object.
    
    view_initial(self, initial, max_depth, c_test, col='crimson', DC_yaxis='linear', 
                     fig=None, ax=None, figwidth=16, figheight=12, return_ct=False):
        Plot the initially specified shear wave velocity profile (initial). 
        The associated theoretical dispersion curve is computed and compared 
        to the experimental data.
    
    within_boundaries(self, runs='all')
        Identify sampled profiles whose theoretical dispersion curves fall 
        within the upper and lower boundaries of the experimental data.
    
    _check_runs(self, runs)
        Check parameters of plotting methods.
    
    _group_by_layers(self, parameter, dataset='selected')
        Group model parameters by layers.
    
    _h_to_z_profile(self, h, max_depth)
        Format layer thickness arrays for plotting interval velocity profiles.
    
    
    Class methods
    -------------
    from_dict(site, profile, d, metadata=None)
        Initialize an inversion analysis object from a dictionary containing  
        a composite experimental dispersion curve.
        
    
    Static methods
    --------------      
    _beta_profile(beta)
        Format shear wave velocity arrays for plotting interval velocity profiles.
    
    _check_length(l, arr, var_name)
        Ensure that the array arr is of length l.
    
    _combine_runs(list_of_sublists)
        Create a one-dimensional list from a list of lists.
    
    _h_to_z(h, n)
        Get locations of layer interfaces (z) from layer thicknesses (h). 
    
    _sort_by(a, b, reverse=False)
        Sort the elements of array a according to the elements of array b.
    
    _to_list(obj)
        Return integer/float as a list.     
    
    _z_profile(z, max_depth)
        Format depth arrays (locations of layer interfaces) for plotting
        interval velocity profiles. 
     
    _z_to_h(z, n)
        Get layer thicknesses from locations of layer interfaces. 

    """    
    
    def __init__(self, site, profile, c_obs, c_obs_low, c_obs_up, wavelength, metadata=None):    
        
        """
        Initialize an inversion analysis object.

        Parameters
        ----------
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
        c_obs : numpy.ndarray
            Rayleigh wave phase velocity array [m/s].
        c_obs_low : numpy.ndarray
            Rayleigh wave phase velocity array, lower boundary values [m/s].     
        c_obs_up : numpy.ndarray
            Rayleigh wave phase velocity array, upper boundary values [m/s].
        wavelength : numpy.ndarray
            Wavelength array [m].
        metadata : str or dict, optional
            Meta-information for object. Default is metadata=None.
 
        Returns
        -------
        InvertDC
            Initialized inversion analysis object.

        """
        # General information on test site and experimental data
        self.site = site
        self.profile = profile
        self.metadata = metadata
        
        # Experimental (observed) dispersion curve.
        self.c_obs = c_obs
        self.c_obs_low = c_obs_low
        self.c_obs_up = c_obs_up
        self.wavelength = wavelength
        
        # All sampled profiles.
        self.profiles = {
                'n_layers' : None,
                'beta' : None,
                'alpha' : None, 
                'h' : None, 
                'c_t' : None,
                'misfit' : None
                }
    
        # Sampled profiles whose theoretical dispersion curves fall within
        # the upper/lower boundaries of the experimental data.
        self.selected = {
                'n_layers' : None,
                'beta' : None,
                'alpha' : None, 
                'h' : None, 
                'z': None,
                'c_t' : None,
                'misfit' : None
                }
    
    @classmethod
    def from_dict(cls, site, profile, d, metadata=None):
        
        """
        Initialize an inversion analysis object from a dictionary containing 
        a composite experimental dispersion curve.
        
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
        d : dict
            Dictionary containing an experimental dispersion curve with keys
            'c_mean', 'c_low', 'c_up' and 'wavelength'.
        metadata : str or dict, optional
            Meta-information for object. Default is metadata=None.
            
        Returns
        -------
        InvertDC
            Initialized inversion analysis object.
        
        """       
        return InvertDC(site, profile, d['c_mean'], d['c_low'], d['c_up'], d['wavelength'], metadata=metadata)

    
    def misfit(self, c_t):
    
        """
        Evaluate dispersion misfit.
        
        Parameters
        ----------       
        c_t : numpy.ndarray
            Theoretical dispersion curve, Rayleigh wave phase velocity values [m/s].
            
        Returns
        -------
        float 
            Misfit between experimental and theoretical dispersion curves [%].
        
        """
        return (1 / len(c_t)) * sum(np.sqrt((self.c_obs - c_t) ** 2) / self.c_obs) * 100
    
    
    @staticmethod
    def _check_length(l, arr, var_name):
        
        """
        Ensure that the array arr is of length l.
        
        Parameters
        ----------
        l : int
            Expected length of arr.
        arr : list or numpy.ndarray 
            Array.
        var_name : str
            Name of the variable that is being checked.
        
        Returns
        -------       
        arr : list or numpy.ndarray 
            Checked array (array of size (l,)).
        
        Raises
        ------
        ValueError
            If the length of arr is not equal to l.    
        
        """
        if not len(arr) == l:
            message = f'´{var_name}´ must be of length {l}'
            raise ValueError(message)
        else:
            return arr
          
    
    def view_initial(self, initial, max_depth, c_test, col='crimson', DC_yaxis='linear', 
                     fig=None, ax=None, figwidth=16, figheight=12, return_ct=False):
        
        """
        Plot the initially specified shear wave velocity profile (initial). 
        The associated theoretical dispersion curve is computed and compared 
        to the experimental data.
        
        Parameters
        ----------        
        initial : dict
            Dictionary containing information on model parameterization and 
            parametric values for the shear wave velocity profile. 
            For further information on initial, please refer to the 
            documentation of mc_inversion. The following key: value pairs are required.
        
            initial['n'] : int
                Number of finite thickness layers.
            initial['alpha'] : numpy.ndarray 
                Initial values for the compressional wave velocity of each layer [m/s] 
                (array of size (n+1,)).
            initial['beta'] : numpy.ndarray 
                Initial values for the shear wave velocity of each layer [m/s] 
                (array of size (n+1,)).
            initial['rho'] : numpy.ndarray 
                Mass density of each layer [kg/m^3] (array of size (n+1,)).
            initial['h'] : numpy.ndarray 
                Initial thickness of each layer [m] (array of size (n,)).
            
        max_depth : int or float
            Maximum depth of shear wave velocity profile (for plotting) [m].
        c_test : dict
            Dictionary containing information on testing Rayleigh wave phase
            velocity values. For further information on c_test, please refer 
            to the documentation of mc_inversion.  
        col : a Matplotlib color or sequence of color, optional
            Linecolor.
            Default is col='crimson'.    
        DC_yaxis : {'linear', 'log'}, optional
            Scale of dispersion curve wavelength axis. 
            - DC_yaxis='linear': Linear scale for wavelengths.
            - DC_yaxis='log': Logarithmic scale for wavelengths.
            Default is DC_yaxis='linear'.
        fig : figure, optional
            Figure object.
            Default is fig=None, a new figure object is created.
        ax : axes object, optional 
            The axes of the subplots. 
            Default is ax=None, a new set of axes is created.
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=16. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.  
        return_ct : boolean, optional
            If return_ct is True, the theoretical dispersion curve associated
            with the given shear wave velocity profile is returned. 
            Default is return_ct=False.
        
        Returns
        -------
        c_t : numpy.ndarray, optional
            Theoretical dispersion curve, Rayleigh wave phase velocity array [m/s].
            The associated wavelengths are stored in self.wavelength.
        
        """
        # Testing range for Rayleigh wave phase velocity [m/s]
        c_vec = np.arange(c_test['min'], c_test['max'], c_test['step'], dtype = np.float64)
        
        # Check input parameters
        initial['alpha'] = self._check_length(initial['n']+1, initial['alpha'], 'alpha_initial')
        initial['beta'] = self._check_length(initial['n']+1, initial['beta'], 'beta_initial')
        initial['rho'] = self._check_length(initial['n']+1, initial['rho'], 'rho')
        initial['h'] = self._check_length(initial['n'], initial['h'], 'h_initial')
        
        # Compute the theoretical dispersion curve and evaluate the dispersion  
        # misfit value for the initial set of model parameters
        c_t = t_dc.compute_fdma(c_vec, c_test['step'], self.wavelength, initial['n'], initial['alpha'], initial['beta'], initial['rho'], initial['h'], c_test['delta_c'])
        e = self.misfit(c_t)
        

        if ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)), constrained_layout=True)
            
            # Plot experimental dispersion curve
            ax[0].plot(self.c_obs, self.wavelength,'-', color='k', label='Mean')
            ax[0].plot(self.c_obs_low, self.wavelength, '--', color='k', label='Upper/lower')
            ax[0].plot(self.c_obs_up, self.wavelength, '--', color='k')

            ax[0].grid(color='gainsboro', linestyle=':')
            ax[1].grid(color='gainsboro', linestyle=':')
            
            # Axes labels and limits 
            ax[0].set_xlabel('Rayleigh wave velocity [m/s]', fontweight='bold')
            ax[0].set_ylabel('Wavelength [m]', fontweight='bold')
            ax[0].set_xlim(s.round_down_to_nearest(min(min(c_t), min(self.c_obs_up))-10, 20), s.round_up_to_nearest(max(max(c_t), max(self.c_obs_up))+10, 20))
            ax[0].set_ylim(0, s.round_up_to_nearest(max(self.wavelength), 5))            
            ax[0].invert_yaxis() 

            ax[1].set_xlabel('Shear wave velocity [m/s]', fontweight='bold')
            ax[1].set_ylabel('Depth [m]', fontweight='bold')
            ax[1].set_xlim(s.round_down_to_nearest(min(initial['beta'])-10, 20), s.round_up_to_nearest(max(initial['beta'])+10, 20))
            ax[1].set_ylim(0, max_depth)
            ax[1].invert_yaxis() 
            
        # Plot theoretical dispersion curve
        ax[0].plot(c_t, self.wavelength, c=col, label='Theoretical')
        
        # Plot shear wave velocity profile
        beta_plot = self._beta_profile(initial['beta'])
        z_plot = self._h_to_z_profile(initial['h'], max_depth)
        ax[1].plot(beta_plot, z_plot, c=col, label='Initial Vs profile')

        # Create legends
        ax[0].legend(frameon=False)
        ax[1].legend(frameon=False)
                      
        # Print misfit value
        print(f'Misfit: {round(e,3)} %')
        
        if return_ct:
            return c_t
    

    def mc_initiation(self, c_vec, c_step, delta_c, n, n_unsat, alpha_initial, nu_unsat, 
                alpha_sat, beta_initial, rho, h_initial, reversals, bs, bh, N_max):

        """
        Monte Carlo based inversion for shear wave velocity and layer thicknesses, 
        single initiation of the sampling scheme.

        Parameters
        ----------
        c_vec : numpy.ndarray
            Rayleigh wave phase velocity, testing values [m/s].
        c_step : float
            Testing Rayleigh wave phase velocity increment [m/s].
        delta_c : float
            Zero search initiation parameter [m/s].
                At wave number k_{i} the zero search is initiated at a phase velocity 
                value of c_{i-1} - delta_c, where c_{i-1} is the theoretical  
                Rayleigh wave phase velocity value at wave number k_{i-1}.
        n : int
            Number of finite thickness layers.
        n_unsat : int
            Number of unsaturated (soft) soil layers. 
            n_unsat = 0 for a fully saturated profile (gwt at surface).
            n_unsat = 1 for an unsaturated surficial layer.
            ...
            n_unsat = n+1 for a fully unsaturated profile.
        alpha_initial : numpy.ndarray 
            Initial values for the compressional wave velocity of each layer [m/s] 
            (array of size (n+1,)).
        nu_unsat : float
            Poisson's ratio for unsaturated soil layers.
        alpha_sat : float
            Compressional wave velocity for saturated soil layers [m/s].
        beta_initial : numpy.ndarray 
            Initial values for the shear wave velocity of each layer [m/s] 
            (array of size (n+1,)).
        rho : numpy.ndarray 
            Mass density of each layer [kg/m^3] (array of size (n+1,)).
        h_initial : numpy.ndarray 
            Initial thickness of each layer [m] (array of size (n,)).
        reversals : int
            Number of layer interfaces (counted from the surface) where 
            velocity reversals are permitted.
            reversals = 0 for a normally dispersive profile.
            reversals = 1 for allowing velocity reversals at the first layer interface.
            ...
        bs : int or float
            Shear wave velocity search-control parameter.
        bh : int or float
            Layer thickness search-control parameter.
        N_max : int
            Number of iterations.

        Returns
        -------
        beta_run : list of numpy.ndarrays
            List of sampled shear wave velocity arrays [m/s].
        alpha_run : list of numpy.ndarrays
            List of compressional wave velocity arrays [m/s].
        h_run : list of numpy.ndarrays
            List of sampled layer thickness arrays [m].
        c_t_run : list of numpy.ndarrays
            List of theoretical dispersion curves, Rayleigh wave phase velocity [m/s].
        e_run : list
            List of dispersion misfit values.
          
        beta_run[i], alpha_run[i], h_run[i], c_t_run[i] and e_run[i] result 
        from the i-th iteration. 

        """        
        # Initiation    
        beta_run = [None for i in range(N_max)]
        alpha_run = [None for i in range(N_max)]
        h_run = [None for i in range(N_max)]
        c_t_run = [None for i in range(N_max)]
        e_run = [None for i in range(N_max)]
                
        # Compute the theoretical dispersion curve and evaluate the dispersion  
        # misfit value for the initial set of model parameters
        c_t = t_dc.compute_fdma(c_vec, c_step, self.wavelength, n, alpha_initial, beta_initial, rho, h_initial, delta_c)
        e_opt = self.misfit(c_t)
        beta_opt = beta_initial
        h_opt = h_initial
        
        # Iteration
        for w in range(N_max):
            
            # Testing shear wave velocity array (sample the shear wave velocity for each layer)
            beta_test = beta_opt + np.random.uniform(low=(-(bs / 100) * beta_opt), high=((bs / 100) * beta_opt))
            while (beta_test[reversals:] != np.sort(beta_test[reversals:])).any():
                beta_test = beta_opt + np.random.uniform(low=(-(bs / 100) * beta_opt), high=((bs / 100) * beta_opt))
                
            # Testing compressional wave velocity array (reevaluate the compressional wave 
            # velocity for each layer based on its testing shear wave velocity value)
            alpha_unsat = np.sqrt((2 * (1 - nu_unsat)) / (1 - 2 * nu_unsat)) * beta_test
            alpha_test = alpha_sat * np.ones(len(beta_test), dtype=np.double)
            if (n_unsat != 0):
                alpha_test[0:n_unsat] = alpha_unsat[0:n_unsat]

            # Testing layer thickness array (sample the thickness for each layer)
            h_test = h_opt + np.random.uniform(low=(-(bh / 100) * h_opt), high=((bh / 100) * h_opt))
            
            # Compute the theoretical dispersion curve and evalute the dispersion misfit
            c_t = t_dc.compute_fdma(c_vec, c_step, self.wavelength, n, alpha_test, beta_test, rho, h_test, delta_c)
            e_test = self.misfit(c_t)
            
            beta_run[w] = beta_test
            h_run[w] = h_test
            alpha_run[w] = alpha_test
            c_t_run[w] = c_t
            e_run[w] = e_test
            
            if e_test <= e_opt:
                e_opt = e_test
                beta_opt = beta_test
                h_opt = h_test
        
        return beta_run, alpha_run, h_run, c_t_run, e_run
    
    
    def mc_inversion(self, c_test, initial, settings):
    
        """
        Monte Carlo based inversion for shear wave velocity and layer thicknesses. 
        The sampling scheme is initiated ´run´ number of times starting from the 
        initially specified layered soil model.
        
        Parameters
        ----------
        c_test : dict
            Dictionary containing information on testing Rayleigh wave phase
            velocity values.        
            
            c_test['min'] : float
                Minimum testing Rayleigh wave phase velocity [m/s].
            c_test['max'] : float
                Maximum testing Rayleigh wave phase velocity [m/s].
            c_test['step'] : float
                Testing Rayleigh wave phase velocity increment [m/s].
            c_test['delta_c'] : float
                Zero search initiation parameter [m/s].
                At wave number k_{i} the zero search is initiated at a phase velocity 
                value of c_{i-1} - c_test['delta_c'], where c_{i-1} is the theoretical  
                Rayleigh wave phase velocity value at wave number k_{i-1}.
        
        initial : dict
            Dictionary containing information on model parameterization and 
            initial values for the testing shear wave velocity profile.
        
            initial['n'] : int
                Number of finite thickness layers.
            initial['n_unsat'] : int
                Number of unsaturated (soft) soil layers. 
                initial['n_unsat'] = 0 for a fully saturated profile (gwt at surface).
                initial['n_unsat'] = 1 for an unsaturated surficial layer.
                ...
                initial['n_unsat'] = n+1 for a fully unsaturated profile.
            initial['alpha'] : numpy.ndarray 
                Initial values for the compressional wave velocity of each layer [m/s] 
                (array of size (n+1,)).
            initial['nu_unsat'] : float
                Poisson's ratio for unsaturated soil layers.
            initial['alpha_sat'] : float
                Compressional wave velocity for saturated soil layers [m/s].
            initial['beta'] : numpy.ndarray 
                Initial values for the shear wave velocity of each layer [m/s] 
                (array of size (n+1,)).
            initial['rho'] : numpy.ndarray 
                Mass density of each layer [kg/m^3] (array of size (n+1,)).
            initial['h'] : numpy.ndarray 
                Initial thickness of each layer [m] (array of size (n,)).
            initial['reversals'] : int
                Number of layer interfaces (counted from the surface) where 
                velocity reversals are permitted.
                initial['reversals'] = 0 for a normally dispersive profile.
                initial['reversals'] = 1 for allowing velocity reversals at the first layer interface.
                ...
        
        settings : dict
            Dictionary containing information on search-control parameter values
            and general settings of the sampling scheme. See further in 
            Olafsdottir et al. (2020), https://doi.org/10.3390/geosciences10080322
        
            settings['run'] : int
                Number of initiations.
            settings['bs'] : int or float
                Shear wave velocity search-control parameter. 
            settings['bh'] : int or float
                Layer thickness search-control parameter.
            settings['N_max'] : int
                Number of iterations.

        Returns
        -------
        profiles : dict 
            Dictionary containing the sampled shear wave velocity profiles 
            and the associated theoretical dispersion curves/dispersion
            misfit values.
            
            profiles['n_layers'] : int
                Number of finite thickness layers.
            profiles['beta'] : list
                Sampled shear wave velocity arrays [m/s].
                profiles['beta'][i] is a list of sampled shear wave velocity arrays
                (numpy.ndarrays) resulting from the i-th run of the algorithm.
            profiles['alpha'] : list
                Compressional wave velocity arrays [m/s].
                profiles['alpha'][i] is a list of compressional wave velocity arrays
                (numpy.ndarrays) resulting from the i-th run of the algorithm.
            profiles['h'] : list          
                Sampled layer thickness arrays [m]. 
                profiles['h'][i] is a list of sampled layer thickness arrays
                (numpy.ndarrays) resulting from the i-th run of the algorithm.
            profiles['c_t'] : list 
                Theoretical dispersion curves, Rayleigh wave phase velocity [m/s].
                profiles['c_t'][i] is a list of theoretical Rayleigh wave phase 
                velocity values (numpy.ndarrays) resulting from the i-th run 
                of the algorithm.
            profiles['misfit'] : list 
                Dispersion misfit values.
                profiles['misfit'][i] is a list of dispersion misfit values 
                resulting from the i-th run of the algorithm.
                
        """       
        # Testing range for Rayleigh wave phase velocity [m/s]
        c_vec = np.arange(c_test['min'], c_test['max'], c_test['step'], dtype = np.float64)
        
        # Initiate lists for temporarily storing sampled shear wave velocity 
        # profiles and associated dispersion curves/dispersion misfit values
        run = settings['run']
        beta_all = [None for i in range(run)]
        alpha_all = [None for i in range(run)]
        h_all = [None for i in range(run)]
        c_t_all = [None for i in range(run)]
        e_all = [None for i in range(run)]
        
        # Check input parameters
        initial['alpha'] = self._check_length(initial['n']+1, initial['alpha'], 'alpha_initial')
        initial['beta'] = self._check_length(initial['n']+1, initial['beta'], 'beta_initial')
        initial['rho'] = self._check_length(initial['n']+1, initial['rho'], 'rho')
        initial['h'] = self._check_length(initial['n'], initial['h'], 'h_initial')
        
        # Iteration (´run´ initiations)
        for i in range(run):
            print(f'Run no. {i+1}/{run} started.')
            temp = self.mc_initiation(c_vec, c_test['step'], c_test['delta_c'], 
                                initial['n'], initial['n_unsat'], initial['alpha'], 
                                initial['nu_unsat'], initial['alpha_sat'], initial['beta'],
                                initial['rho'], initial['h'], initial['reversals'], 
                                settings['bs'], settings['bh'], settings['N_max'])
            
            beta_all[i], alpha_all[i], h_all[i], c_t_all[i], e_all[i] = temp
            print(f'Run no. {i+1}/{run} completed.')
        
        # Return to inversion analysis object
        self.profiles['n_layers'] = initial['n']
        self.profiles['beta'] = beta_all
        self.profiles['alpha'] = alpha_all
        self.profiles['h'] = h_all
        self.profiles['c_t'] = c_t_all
        self.profiles['misfit'] = e_all


    @staticmethod
    def _combine_runs(list_of_sublists):
        
        """
        Create a one-dimensional list from a list of lists.
        
        Parameters
        ----------
        list_of_sublists : list
            List of sublists (a two-dimensional list). The sublists may contain  
            varying number of elements.
            
        Returns
        -------
        list
            One-dimensional list created from list_of_lists.
                    
        """
        return [item for sublist in list_of_sublists for item in sublist]
    
    
    @staticmethod
    def _sort_by(a, b, reverse=False):
        
        """
        Sort the elements of array a according to the elements of array b.
        
        Parameters
        ----------
        a : array-like
            Array to be sorted.
        b : array-like
            Reference array. 
        reverse : boolean, optional
            If set to False, the elements of array b are sorted in increasing order.
            If set to True, the elements of array b are sorted in decreasing order.
            (See Example below.)
            Default is reverse=False.
            
        Returns
        -------
        list
            Array a sorted according to the elements of array b in increasing/
            decreasing order.
            
        Example
        -------
        a = [10, 20, 30, 40, 50, 60]
        b = [4, 3, 2, 5, 6, 1]
        
        inversion.InvertDC._sort_by(a, b, reverse=False)
        [60, 30, 20, 10, 40, 50]
        
        inversion.InvertDC._sort_by(a, b, reverse=True)
        [50, 40, 10, 20, 30, 60]
        
        """
        return [x for (y,x) in sorted(zip(b,a), key=lambda pair: pair[0], reverse=reverse)]


    def _check_runs(self, runs):
        
        """
        Check parameters of plotting methods:
        - Ensure that ´runs´ is defined as 'all', an integer, or a list/array of integers. 
        - Ensure that ´runs´ does not exceed the number of saved initiations.
        - If ´runs´ is defined as 'all' or is an integer, a corresponding iterable
          is returned.
        
        Parameters
        ----------
        runs : str, int, list or numpy.ndarray
            Describes which sampling scheme initiations are to be displayed.     
        
        Returns
        -------
        runs : list or numpy.ndarray
            List of sampling scheme initiations that are to be displayed.
        
        Raises
        ------
        ValueError
            If ´runs´ does not have the correct format.
        
        """
        saved_runs = len(self.profiles['beta'])
        
        # ´runs´ is an integer
        if type(runs) is int:
            # Check if ´runs´ exceeds the number of saved algorithm initiations
            if runs >= saved_runs:
                message = '´runs´ exceeds the number of saved initiations.'
                raise ValueError(message)
            # Return ´runs´ as an iterable
            return [runs]
        # ´runs´ is a list/numpy.ndarray
        elif type(runs) is np.ndarray or type(runs) is list:
            # Check if all elements of ´runs´ are integers
            if not all(isinstance(i, int) for i in runs):
                message = 'All elements of ´runs´ must be integers.'
                raise ValueError(message)
            # Check if any element of ´runs´ exceeds the number of saved algorithm initiations    
            if any(i >= saved_runs for i in runs):
                message = '´runs´ exceeds the number of saved initiations.'
                raise ValueError(message)
            return runs
        # ´runs´ is defined as ´all´
        elif type(runs) is str and runs.lower() == 'all':
            # Return a list of all saved algorithm initiations
            return list(range(saved_runs))
        else: 
            message = '´runs´ must be defined as ´all´, an integer or a list/array of integers.'
            raise ValueError(message)    


    @staticmethod
    def _h_to_z(h, n):
        
        """
        Get locations of layer interfaces (z) from layer thicknesses (h). 
        For h = [h1, h2, h3], the returned list is z = [h1, h1+h2, h1+h2+h3].
        
        Parameters
        ----------
        h : list or numpy.ndarray
            Layer thickness array [m].
        n : int
            Number of finite thickness layers.
            
        Returns
        -------
        list
            Depth array (locations of layer interfaces) [m].
                        
        """
        return [sum(h[0:i+1]) for i in range(n)]
    
    
    @staticmethod
    def _z_to_h(z, n):
        
        """
        Get layer thicknesses (h) from locations of layer interfaces (z). 
        For z = [z1, z2, z3], the returned list is h = [z1, z2-z1, z3-z2].
        
        Parameters
        ----------
        z : list or numpy.ndarray
            Depth array (locations of layer interfaces) [m].
        n : int
            Number of finite thickness layers.
            
        Returns
        -------
        list
            Layer thickness array [m].        
                
        """
        return [z[0]] + [z[i+1]-z[i] for i in range(n-1)]
    
    
    @staticmethod
    def _z_profile(z, max_depth):
        
        """
        Format depth arrays (locations of layer interfaces) for plotting
        interval velocity profiles. For z = [z1, z2] (where z2 <= max_depth),
        the returned list is z_plot = [0, z1, z1, z2, z2, max_depth].
        
        Parameters
        ----------
        z : numpy.ndarray or list
            Depth array (locations of layer interfaces) [m].
        max_depth : int or float
            Maximum depth of shear wave velocity profile (for plotting) [m].
            
        Returns
        -------
        z_plot : numpy.ndarray
            Depth array formatted for plotting [m].
        
        """
        temp = sorted(np.concatenate((z, z))) 
        z_plot = np.concatenate(([0], temp, [max(max_depth,z[-1])]))
        return z_plot

    
    def _h_to_z_profile(self, h, max_depth):
        
        """
        Format layer thickness arrays for plotting interval velocity profiles.
        For h_vec = [h1, h2] (where h1+h2 <= max_depth), the returned list is
        z_plot = [0, h1, h1, h1+h2, h1+h2, max_depth].
        
        If h_vec is a list of layer thickness arrays, each element of 
        h_vec is formatted separately, i.e., if h_vec[i] = [h1, h2], 
        then the i-th element of the returned list is 
        z_plot[i] = [0, h1, h1, h1+h2, h1+h2, max_depth].
        
        Parameters
        ----------
        h_list : numpy.ndarray/list or list of numpy.ndarrays/lists 
            Layer thickness array(s) [m].
        max_depth : int or float
            Maximum depth of shear wave velocity profile(s) (for plotting) [m].
            
        Returns
        -------
        z_plot : numpy.ndarray or list of numpy.ndarrays
            Depth array(s) formatted for plotting [m].
        
        """
        if any(isinstance(el, list) or isinstance(el, np.ndarray) for el in h):
            no_profiles = len(h) 
            z_plot = [None] * no_profiles
            for i in range(no_profiles):
                temp = self._h_to_z(h[i], len(h[i]))
                z_plot[i] = self._z_profile(temp, max_depth)
        else:
            temp = self._h_to_z(h, len(h))
            z_plot = self._z_profile(temp, max_depth)
        
        return z_plot


    @staticmethod
    def _beta_profile(beta):
        
        """
        Format shear wave velocity arrays for plotting interval velocity profiles. 
        
        For beta = [beta1, beta2, beta3], the returned list is
        beta_plot = [beta1, beta1, beta2, beta2, beta3, beta3].
        
        If beta is a list of shear wave velocity arrays, each element 
        is formatted separately, i.e., if beta[i] = [beta1, beta2, beta3], 
        then the i-th element of the returned list is 
        beta_plot[i] = [beta1, beta1, beta2, beta2, beta3, beta3].
        
        Parameters
        ----------
        beta : numpy.ndarray/list or list of numpy.ndarrays/lists 
            Shear wave velocity array(s) [m/s]. 
        
        Returns
        -------
        beta_plot: numpy.ndarray or list of numpy.ndarrays
            Shear wave velocity array(s) formatted for plotting [m/s].
        
        """
        if any(isinstance(el, list) or isinstance(el, np.ndarray) for el in beta):
            no_profiles = len(beta)   
            beta_plot = [None] * no_profiles
            for i in range(no_profiles):
                beta_plot[i] = np.repeat(beta[i], 2)
        else:
            beta_plot = np.repeat(beta, 2)
                
        return beta_plot


    def plot_sampled(self, max_depth, runs='all', figwidth=16, figheight=12, col_map='viridis', 
                      colorbar=True, DC_yaxis='linear', return_axes=False, show_exp_dc=True):
        
        """
        Plot sampled shear wave velocity profiles (as interval velocity profiles) 
        and the corresponding theoretical dispersion curves. The shear wave 
        velocity profiles/dispersion curves are sorted based on dispersion misfit 
        values and displayed using a color scale. 
        
        Parameters
        ----------
        max_depth : int or float
            Maximum depth of shear wave velocity profiles (for plotting) [m].
        runs : str ('all'), int, list or numpy.ndarray, optional
            Sampling scheme initiations (runs) that are to be displayed. 
            Default is to combine and show all runs, i.e., runs='all'.
            Alternatively, runs can be defined as an integer or a list/array
            of integers, e.g.,
            - runs=0: Show run no. 1.
            - runs=1: Show run no. 2.
            - runs=[0,1]: Show runs no. 1 and 2 (combined).
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=16. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.
        col_map : str or Colormap, optional
            Registered colormap name or a Colormap instance.
            Default is col_map='viridis'.
        colorbar : boolean, optional
            Show colorbar (yes=True, no=False).
            Default is colorbar=True.
        DC_yaxis : {'linear', 'log'}, optional 
            Scale of dispersion curve wavelength axis. 
            - DC_yaxis='linear': Linear scale for wavelengths.
            - DC_yaxis='log': Logarithmic scale for wavelengths.
            Default is DC_yaxis='linear'.
        return_axes : boolean, optional
            If return_axes is True, the initialized figure object and set of 
            axes are returned. Default is return_axes=False.
        show_exp_dc : boolean, optional
            Show experimental dispersion curve (yes=True, no=False).
            Default is show_exp_dc=True.
            
        Returns
        -------
        fig : figure, optional
            Figure object.
        ax : axes object, optional 
            The axes of the subplots. 
        
        Raises
        ------
        AttributeError
            If no sampled profiles exist.         

        """
        if not bool(self.profiles.get('n_layers')):
            message = 'No sampled profiles exist. The dictionary ´profiles´ contains None values.' 
            raise AttributeError(message)
            
        runs = self._check_runs(runs)
        
        # Sort data by dispersion misfit values
        e_all = self._combine_runs([self.profiles['misfit'][i] for i in runs])
        c_sort = self._sort_by(self._combine_runs([self.profiles['c_t'][i] for i in runs]), e_all, reverse=True)
        beta_sort = self._sort_by(self._combine_runs([self.profiles['beta'][i] for i in runs]), e_all, reverse=True)
        h_sort = self._sort_by(self._combine_runs([self.profiles['h'][i] for i in runs]), e_all, reverse=True)
        e_sort = sorted(e_all, reverse=True)
        no_profiles = len(e_sort)      

        # Figure settings
        fig, ax = plt.subplots(1, 2, figsize=(s.cm_to_in(figwidth),s.cm_to_in(figheight)), constrained_layout=True)
        
        # Plot theoretical dispersion curves
        lc = s.plot_lines(c_sort, [self.wavelength for i in range(no_profiles)], list(range(no_profiles)), ax=ax[0], cmap=col_map, lw=0.4)  
        if show_exp_dc:
            ax[0].plot(self.c_obs, self.wavelength, '-', color='k')
            ax[0].plot(self.c_obs_low, self.wavelength, '--', color='k')
            ax[0].plot(self.c_obs_up, self.wavelength, '--', color='k')
        
        # Axes labels and limits 
        ax[0].set_xlim(s.round_down_to_nearest(np.min(c_sort)-10, 20), s.round_up_to_nearest(np.max(c_sort)+10, 20))
        ax[0].set_xlabel('Rayleigh wave velocity [m/s]', fontweight='bold')
        ax[0].set_ylabel('Wavelength [m]', fontweight='bold')
        
        # Figure appearance
        if type(DC_yaxis) is str and DC_yaxis.lower() == 'linear':
            ax[0].set_ylim(0, s.round_up_to_nearest(max(self.wavelength), 5))
            ax[0].set_yscale('linear')
        elif type(DC_yaxis) is str and DC_yaxis.lower() == 'log':
            ax[0].set_ylim(max(s.round_down_to_nearest(min(self.wavelength), 1), 0.1) , s.round_up_to_nearest(max(self.wavelength), 5))
            ax[0].set_yscale('log')
        else:
            ax[0].set_ylim(0, s.round_up_to_nearest(max(self.wavelength), 5))
            ax[0].set_yscale('linear')
            warnings.warn('Scale of wavelength axis (DC_yaxis) not recognized. Dispersion curves displayed on a linear wavelength scale (default setting).')
        ax[0].invert_yaxis() 
        ax[0].grid(color='gainsboro', linestyle=':')
        
        # Plot shear wave velocity profiles
        beta_plot = self._beta_profile(beta_sort)
        z_plot = self._h_to_z_profile(h_sort, max_depth)
        s.plot_lines(beta_plot, z_plot, list(range(no_profiles)), ax=ax[1], cmap=col_map, lw=0.4)  
        
        # Axes labels and limits 
        ax[1].set_xlim(s.round_down_to_nearest(np.min(beta_sort)-10, 20), s.round_up_to_nearest(np.max(beta_sort)+10, 20))
        ax[1].set_ylim(0, max_depth)
        ax[1].set_xlabel('Shear wave velocity [m/s]', fontweight='bold')
        ax[1].set_ylabel('Depth [m]', fontweight='bold')
                
        # Figure appearance
        ax[1].invert_yaxis() 
        ax[1].grid(color='gainsboro', linestyle=':')
        
        if colorbar:
            relative_tick_locations = [0, 0.25, 0.5, 0.75, 1]
            tick_locations = [i * (len(beta_plot)-1) for i in relative_tick_locations]
            cbar = fig.colorbar(lc, ax=ax.ravel().tolist(), location='top', aspect=50)
            cbar.set_ticks(tick_locations)
            tick_labels = [round(i,2) for i in np.quantile(e_sort, relative_tick_locations[::-1])]
            cbar.set_ticklabels(tick_labels)
            cbar.set_label('Dispersion misfit [%]', fontweight='bold')
            cbar.ax.invert_xaxis() 
        
        if return_axes: 
            return fig, ax
   

    def within_boundaries(self, runs='all'):
        
        """
        Identify sampled profiles whose theoretical dispersion curves fall 
        within the upper and lower boundaries of the experimental data.
        
        Parameters
        ----------
        runs : str ('all'), int, list or numpy.ndarray, optional
            Sampling scheme initiations (runs) that are to be displayed. 
            Default is to combine and show all runs, i.e., runs='all'.
            Alternatively, runs can be defined as an integer or a list/array
            of integers, e.g.,
            - runs=0: Show run no. 1.
            - runs=1: Show run no. 2.
            - runs=[0,1]: Show runs no. 1 and 2 (combined).
        
        Returns
        -------
        selected : dict 
            Dictionary containing sampled shear wave velocity profiles 
            whose associated theoretical dispersion curves fall within the
            upper and lower boundaries of the experimental data.
            The profiles/dispersion curves are ordered by dispersion
            misfit values.
            
            selected['n_layers'] : int
                Number of finite thickness layers.
            selected['beta'] : list of numpy.ndarrays
                Shear wave velocity arrays [m/s].
            selected['alpha'] : list of numpy.ndarrays
                Compressional wave velocity arrays [m/s].
            selected['h'] : list of numpy.ndarrays       
                Layer thickness arrays [m]. 
            selected['z'] : list of numpy.ndarrays       
                Depth of layer interfaces [m]. 
            selected['c_t'] : list  of numpy.ndarrays
                Theoretical dispersion curves, Rayleigh wave phase velocity [m/s].
            selected['misfit'] : list 
                Dispersion misfit values.

        """       
        runs = self._check_runs(runs)
        
        # Get data
        n = self.profiles['n_layers']
        e_all = self._combine_runs([self.profiles['misfit'][i] for i in runs])
        c_all = self._combine_runs([self.profiles['c_t'][i] for i in runs])
        beta_all = self._combine_runs([self.profiles['beta'][i] for i in runs])
        alpha_all = self._combine_runs([self.profiles['alpha'][i] for i in runs])
        h_all = self._combine_runs([self.profiles['h'][i] for i in runs])
        no_profiles = len(e_all)   
        
        # Find theoretical dispersion curves/interval velocity profiles that 
        # fall within the upper/lower boundaries of the experimental data
        accept = [False] * no_profiles
        for i in range(no_profiles):
            temp = [(self.c_obs_low[j] < c_all[i][j] and c_all[i][j] < self.c_obs_up[j]) for j in range(len(self.c_obs_up))]
            if all(temp) is True:
                accept[i] = True
        res = [i for i, val in enumerate(accept) if val] 
        c_temp = [c_all[i] for i in res] 
        beta_temp = [beta_all[i] for i in res] 
        alpha_temp = [alpha_all[i] for i in res] 
        h_temp = [h_all[i] for i in res] 
        e_temp = [e_all[i] for i in res] 
        
        # Sort by dispersion misfit values and return to the initialized 
        # inversion analysis object
        self.selected['n_layers'] = n
        self.selected['c_t'] = self._sort_by(c_temp, e_temp, reverse=True)
        self.selected['beta'] = self._sort_by(beta_temp, e_temp, reverse=True)
        self.selected['alpha'] = self._sort_by(alpha_temp, e_temp, reverse=True)
        self.selected['h'] = self._sort_by(h_temp, e_temp, reverse=True)
        self.selected['misfit'] = sorted(e_temp, reverse=True)  
        
        # Depth of layer interfaces
        n_selected = len(self.selected['h'])
        z_temp = [None] * n_selected
        for j in range(n_selected):
            z_temp[j] = self._h_to_z(self.selected['h'][j], n)
        self.selected['z'] = z_temp
        
    
    def plot_within_boundaries(self, max_depth, show_all=True, runs='all', figwidth=16, figheight=12, 
                               col_map='viridis', colorbar=True, DC_yaxis='linear', return_axes=False, **kwargs):
        
        """
        Plot sampled profiles whose theoretical dispersion curves fall 
        within the upper and lower boundaries of the experimental data. 
        The corresponding theoretical dispersion curves are compared to the 
        experimental data. The shear wave velocity profiles/dispersion curves 
        are sorted based on dispersion misfit values and displayed using 
        a color scale. 
        
        Parameters
        ----------
        max_depth : int or float
            Maximum depth of shear wave velocity profiles (for plotting) [m].
        show_all : boolean, optional
            Show all sampled shear wave velocity profiles/theoretical
            dispersion curves (yes=True, no=False).
        runs : str ('all'), int, list or numpy.ndarray, optional
            Sampling scheme initiations (runs) that are to be displayed. 
            Default is to combine and show all runs, i.e., runs='all'.
            Alternatively, runs can be defined as an integer or a list/array
            of integers, e.g.,
            - runs=0: Show run no. 1.
            - runs=1: Show run no. 2.
            - runs=[0,1]: Show runs no. 1 and 2 (combined).
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=16. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.
        col_map : str or Colormap, optional
            Registered colormap name or a Colormap instance.
            Default is col_map='viridis'.
        colorbar : boolean, optional
            Show colorbar (yes=True, no=False).
            Default is colorbar=True.
        DC_yaxis : {'linear', 'log'}, optional
            Scale of dispersion curve wavelength axis. 
            - DC_yaxis='linear': Linear scale for wavelengths.
            - DC_yaxis='log': Logarithmic scale for wavelengths.
            Default is DC_yaxis='linear'.
        return_axes : boolean, optional
            If return_axes is True, the initialized figure object and set of 
            axes are returned. Default is return_axes=False.
            
        Returns
        -------
        fig : figure, optional
            Figure object.
        ax : axes object, optional 
            The axes of the subplots. 
        
        Raises
        ------
        AttributeError
            If no sampled profiles exist. 
        
        Other parameters
        ----------------
        All other keyword arguments are passed on to matplotlib.collections.LineCollection. 
        See https://matplotlib.org/3.1.1/api/collections_api.html#matplotlib.collections.LineCollection
        for a list of valid kwargs.
        
        """
        # Get shear wave velocity profiles whose theoretical dispersion curves fall 
        # within the upper and lower boundaries of the experimental data. 
        self.within_boundaries(runs=runs)
        no_within = len(self.selected['c_t'])
        
        # Figure settings
        if show_all:
            # Plot all sampled profiles
            newcmp = ListedColormap(np.array([0.8, 0.8, 0.8, 1]))
            fig, ax = self.plot_sampled(max_depth, runs=runs, figwidth=figwidth, figheight=figheight, col_map=newcmp, 
                      colorbar=False, DC_yaxis=DC_yaxis, return_axes=True, show_exp_dc=False)
            ax0_x0, ax0_x1 = ax[0].get_xlim()
            ax1_x0, ax1_x1 = ax[1].get_xlim()
        else:    
            fig, ax = plt.subplots(1, 2, figsize=(s.cm_to_in(figwidth),s.cm_to_in(figheight)), constrained_layout=True)
            ax[0].grid(color='gainsboro', linestyle=':')
            ax[0].set_xlabel('Rayleigh wave velocity [m/s]', fontweight='bold')
            ax[0].set_ylabel('Wavelength [m]', fontweight='bold')
            ax[1].grid(color='gainsboro', linestyle=':')
            ax[1].set_xlabel('Shear wave velocity [m/s]', fontweight='bold')
            ax[1].set_ylabel('Depth [m]', fontweight='bold')
        
        # Plot theoretical dispersion curves
        lc = s.plot_lines(self.selected['c_t'], [self.wavelength for i in range(no_within)], list(range(no_within)), 
                        ax=ax[0], cmap=col_map, lw=0.4, **kwargs)  
        ax[0].plot(self.c_obs, self.wavelength, '-', color='k')
        ax[0].plot(self.c_obs_low, self.wavelength, '--', color='k')
        ax[0].plot(self.c_obs_up, self.wavelength, '--', color='k')

        # Plot shear wave velocity profiles
        beta_plot = self._beta_profile(self.selected['beta'])
        z_plot = self._h_to_z_profile(self.selected['h'], max_depth)
        s.plot_lines(beta_plot, z_plot, list(range(no_within)), ax=ax[1], cmap=col_map, lw=0.4, **kwargs)  
        
        # Figure appearance
        if type(DC_yaxis) is str and DC_yaxis.lower() == 'log':
            ax[0].set_ylim(max(s.round_down_to_nearest(min(self.wavelength), 1), 0.1) , s.round_up_to_nearest(max(self.wavelength), 5))
        else:
            ax[0].set_ylim(0, s.round_up_to_nearest(max(self.wavelength), 5))
        ax[0].invert_yaxis()
        ax[1].set_ylim(0, max_depth)
        ax[1].invert_yaxis() 
        if show_all:
            ax[0].set_xlim(ax0_x0, ax0_x1)
            ax[1].set_xlim(ax1_x0, ax1_x1)
        else:
            ax[0].set_xlim(s.round_down_to_nearest(min(self.c_obs_low)-10, 20), s.round_up_to_nearest(max(self.c_obs_up)+10, 20))
            ax[1].set_xlim(s.round_down_to_nearest(np.min(beta_plot)-10, 20), s.round_up_to_nearest(np.max(beta_plot)+10, 20))
        
        if colorbar:
            relative_tick_locations = [0, 0.25, 0.5, 0.75, 1]
            tick_locations = [i * (len(beta_plot)-1) for i in relative_tick_locations]
            cbar = fig.colorbar(lc, ax=ax.ravel().tolist(), location='top', aspect=50)
            cbar.set_ticks(tick_locations)
            tick_labels = [round(i,2) for i in np.quantile(self.selected['misfit'], relative_tick_locations[::-1])]
            cbar.set_ticklabels(tick_labels)
            cbar.set_label('Dispersion misfit [%]', fontweight='bold')
            cbar.ax.invert_xaxis() 

        if return_axes: 
            return fig, ax
        
        
    def _group_by_layers(self, parameter, dataset='selected'):
        
        """
        Group model parameters by layers.
        
        Parameters
        ----------
        parameter : {'beta', 'h', 'z'}
            Key of model parameter. 'beta' for shear wave velocity, 
            'h' for layer thickness or 'z' for depth of layer interfaces. 
        
        dataset : {'selected', 'sampled'}, optional
            All sampled profiles ('sampled') or profiles those theoretical
            dispersion curves fall within the boundaries of the experimental 
            data. Default is dataset='selected'.
        
        Returns
        -------
        layer : list of lists
            Model parameters grouped by layers. 
        
        Raises
        ------
        KeyError
            If ´parameter´ does not match the available options.
        ValueError
            If ´dataset´ does not match the available options.
        
        """
        # Get dataset
        if dataset.lower() == 'selected':
            data = self.selected
        elif dataset.lower() == 'sampled':
            data = self.profiles
        else: 
            message = f'Set of shear wave velocity profiles ´{dataset}´ is not recognized.'
            raise ValueError(message)
          
        # Identify model parameter
        if parameter.lower() == 'beta':
            no_groups = data['n_layers'] + 1
            key = parameter.lower()
        elif parameter.lower() in ['h', 'z']:
            no_groups = data['n_layers']
            key = parameter.lower()
        else: 
            message = f'Model parameter ´{parameter}´ is not recognized.'
            raise KeyError(message)
        
        # Group model parameters by layers
        layer = [None] * no_groups
        for i in range(no_groups):
            layer[i] = [data[key][j][i] for j in range(len(data[key]))]        
        
        return layer
    

    def mean_profile(self, stdev=True, no_stdev=1, dataset='selected'):      

        """
        Return the mean shear wave velocity profile (defined in terms of
        shear wave velocity and depth of layer interfaces) and (optional) 
        upper and lower boundaries corresponding to plus/minus no_stdev  
        standard deviation(s) from the mean value.
        
        Parameters
        ----------
        stdev : boolean, optional
            If stdev is True, the standard deviation of each layer
            parameter (shear wave velocity, depth of layer interfaces) 
            is computed. Default is stdev=True.  
        no_stdev : int or float, optional
            Number of standard deviations. Default is no_stdev=1.
        dataset : {'selected', 'sampled'}, optional
            All sampled profiles ('sampled') or profiles those theoretical
            dispersion curves fall within the boundaries of the experimental 
            data. Default is dataset='selected'.
        
        Returns
        -------
        mean_profile : dict
            Dictionary containing the mean shear wave velocity profile.
            
            mean_profile['beta'] : list 
                Shear wave velocity [m/s], mean values.
            mean_profile['z'] : list 
                Locations of layer interfaces [m/s], mean values.
            
            Returned if stdev is True.
            mean_profile['beta_low'], mean_profile['beta_up'] : list                     
                Shear wave velocity [m/s], mean +/- no_stdev * std.
            mean_profile['z_low'], mean_profile['z_up'] : list
                Depth of layer interfaces [m/s], mean +/- no_stdev * std. 
            
        """
        # Get data and sort by layers
        beta_layers = self._group_by_layers('beta', dataset=dataset)
        z_layers = self._group_by_layers('z', dataset=dataset)
        n = self.selected['n_layers']
        
        # Compute mean values
        beta_mean = [np.mean(beta_layers[i], dtype=np.float64) for i in range(n+1)]
        z_mean = [np.mean(z_layers[i], dtype=np.float64) for i in range(n)]
        
        # Return to dictionary
        mean_profile = {}
        mean_profile['beta'] = beta_mean
        mean_profile['z'] = z_mean
        
        # Compute standard deviation of layer parameters
        if stdev:
            beta_std = [np.std(beta_layers[i], ddof=1, dtype=np.float64) for i in range(n+1)]
            z_std = [np.std(z_layers[i], ddof=1, dtype=np.float64) for i in range(n)]
            
            # Return to dictionary
            mean_profile['beta_low'] = [beta_mean[i] - no_stdev * beta_std[i] for i in range(n+1)]
            mean_profile['beta_up'] = [beta_mean[i] + no_stdev * beta_std[i] for i in range(n+1)]
            mean_profile['z_low'] = [z_mean[i] - no_stdev * z_std[i] for i in range(n)]
            mean_profile['z_up'] = [z_mean[i] + no_stdev * z_std[i] for i in range(n)]
            
        return mean_profile


    def median_profile(self, perc=True, q=[10, 90], dataset='selected'):
        
        """
        Return the median shear wave velocity profile (defined in terms of
        shear wave velocity and depth of layer interfaces) and (optional) 
        the q-th percentiles of each parameter.
        
        Parameters
        ----------
        perc : boolean, optional
            If perc is True, the q-th percentiles of each layer
            parameter (shear wave velocity, depth of layer interfaces) 
            are computed and returned. Default is perc=True.    
        q : list, optional
            Pair of percentiles, which must be between 0 and 100 
            inclusive. Default is q = [10, 90] (the 10-th and 90-th
            percentiles are computed and returned).
        dataset : {'selected', 'sampled'}, optional
            All sampled profiles ('sampled') or profiles those theoretical
            dispersion curves fall within the boundaries of the experimental 
            data. Default is dataset='selected'.
        
        Returns
        -------
        median_profile : dict
            Dictionary containing the median shear wave velocity profile.
            
            median_profile['beta'] : list 
                Shear wave velocity [m/s], median values.
            median_profile['z'] : list 
                Locations of layer interfaces [m/s], median values.
            
            Returned if perc is True.
            median_profile['beta_low'], median_profile['beta_up'] : list                     
                Shear wave velocity [m/s], q-th percentiles.
            median_profile['z_low'], median_profile['z_up'] : list
                Depth of layer interfaces [m/s], q-th percentiles. 
         
        Raises 
        ------
        TypeError
            If the list of percentiles (q) does not have the correct format.
            
        """
        # Check parameters
        if len(q) != 2:
            message = 'A pair of percentiles (e.g., q = [10, 90]) must be provided.'
            raise TypeError(message)
        else:
            q = sorted(q)
        
        # Get data and sort by layers
        beta_layers = self._group_by_layers('beta', dataset=dataset)
        z_layers = self._group_by_layers('z', dataset=dataset)
        n = self.selected['n_layers']
        
        # Compute median values
        beta_median = [np.median(beta_layers[i]) for i in range(n+1)]
        z_median = [np.median(z_layers[i]) for i in range(n)]
        
        # Return to dictionary
        median_profile = {}
        median_profile['beta'] = beta_median
        median_profile['z'] = z_median
        
        # Compute q-th percentiles
        if perc:
            beta_percentile = [np.percentile(beta_layers[i], q) for i in range(n+1)]
            z_percentile = [np.percentile(z_layers[i], q) for i in range(n)]
            
            # Return to dictionary
            median_profile['beta_low'] = [beta_percentile[i][0] for i in range(n+1)]
            median_profile['beta_up'] = [beta_percentile[i][1] for i in range(n+1)]
            median_profile['z_low'] = [z_percentile[i][0] for i in range(n)]
            median_profile['z_up'] = [z_percentile[i][1] for i in range(n)]    
            
        return median_profile

    
    def plot_profile(self, profile, max_depth, c_test, initial, col='crimson', up_low=False, DC_yaxis='linear', 
                     fig=None, ax=None, return_axes=False, return_ct=False, show_legend=True, figwidth=16, figheight=12):
        
        """
        Plot the median or mean shear wave velocity profile. The associated 
        theoretical dispersion curve is computed and compared to the experimental
        data.
        
        Parameters
        ----------
        profile : dict
            Dictionary containing the median or mean shear wave velocity profile.                                
            
            profile['beta'] : list 
                Shear wave velocity [m/s].
            profile['z'] : list 
                Locations of layer interfaces [m/s].
            
            Required if up_low is True.
            profile['beta_low'], profile['beta_up'] : list                     
                Shear wave velocity [m/s], boundary values.
            profile['z_low'], profile['z_up'] : list
                Locations of layer interfaces [m/s], boundary values.            
        
        max_depth : int or float
            Maximum depth of shear wave velocity profile (for plotting) [m]. 
        c_test : dict
            Dictionary containing information on testing Rayleigh wave phase
            velocity values. For further information on c_test, please refer 
            to the documentation of mc_inversion.          
        initial : dict
            Dictionary containing information on model parameterization and 
            initial values for the testing shear wave velocity profile.
            For further information on initial, please refer to the 
            documentation of mc_inversion. 
        col : a Matplotlib color or sequence of color, optional
            Linecolor.
            Default is col='crimson'.
        up_low : boolean, optional
            Show error bars for shear wave velocity and depth of layer 
            interfaces (yes=True, no=False).
            Default is up_low=False.
        DC_yaxis : {'linear', 'log'}, optional
            Scale of dispersion curve wavelength axis. 
            - DC_yaxis='linear': Linear scale for wavelengths.
            - DC_yaxis='log': Logarithmic scale for wavelengths.
            Default is DC_yaxis='linear'.
        fig : figure, optional
            Figure object.
            Default is fig=None, a new figure object is created.
        ax : axes object, optional 
            The axes of the subplots. 
            Default is ax=None, a new set of axes is created.
        return_axes : boolean, optional
            If return_axes is True, the initialized figure object and set of 
            axis are returned. 
            Default is return_axes=False.
        return_ct : boolean, optional
            If return_ct is True, the theoretical dispersion curve associated
            with the median or mean shear wave velocity profile is returned. 
            Default is return_ct=False.
        show_legend : boolean, optional
            If show_legend is True, a legend is placed on the left-hand axis 
            of the plot (i.e., the subplot showing the dispersion curves).
            Default is show_legend=True.
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=16. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.           

        Returns
        -------
        fig : figure, optional
            Figure object.
        ax : axes object, optional 
            The axes of the subplots. 
        c_t : numpy.ndarray, optional
            Theoretical dispersion curves, Rayleigh wave phase velocity [m/s].
            The associated wavelengths are stored in self.wavelength.
            
        """
        # Model parameters
        beta = profile['beta']
        z = profile['z']
        n = initial['n']
        beta = np.array(beta)
        alpha_unsat = np.sqrt((2*(1 - initial['nu_unsat'])) / (1 - 2 * initial['nu_unsat'])) * beta
        alpha = initial['alpha_sat'] * np.ones(len(beta))
        if (initial['n_unsat'] != 0):
            alpha[0:initial['n_unsat']] = alpha_unsat[0:initial['n_unsat']]
        h = np.array(self._z_to_h(z, n))

        # Compute theoretical dispersion curve
        c_vec = np.arange(c_test['min'], c_test['max'], c_test['step'], dtype = np.float64) 
        c_t = t_dc.compute_fdma(c_vec, c_test['step'], self.wavelength, n, alpha, beta, initial['rho'], h, c_test['delta_c'])
 
    
        if ax is None:
            # Create a new figure window and set of axes
            fig, ax = plt.subplots(1, 2, figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)), constrained_layout=True)
            
            # Plot experimental dispersion curve
            ax[0].plot(self.c_obs, self.wavelength,'-',color='k', label='Mean')
            ax[0].plot(self.c_obs_low, self.wavelength, '--', color='k', label='Upper/lower')
            ax[0].plot(self.c_obs_up, self.wavelength, '--', color='k')

            ax[0].grid(color='gainsboro', linestyle=':')
            ax[1].grid(color='gainsboro', linestyle=':')
            
            # Axes labels and limits 
            ax[0].set_xlabel('Rayleigh wave velocity [m/s]', fontweight='bold')
            ax[0].set_ylabel('Wavelength [m]', fontweight='bold')
            ax[0].set_xlim(s.round_down_to_nearest(min(min(c_t), min(self.c_obs_up)), 20), s.round_up_to_nearest(max(max(c_t), max(self.c_obs_up)), 20))
            ax[0].set_ylim(0, s.round_up_to_nearest(max(self.wavelength), 5))
            ax[0].invert_yaxis() 

            ax[1].set_xlabel('Shear wave velocity [m/s]', fontweight='bold')
            ax[1].set_ylabel('Depth [m]', fontweight='bold')
            if up_low:
                ax[1].set_xlim(s.round_down_to_nearest(min(profile['beta_low'])-10, 20), s.round_up_to_nearest(max(profile['beta_up'])+10, 20))
            else:
                ax[1].set_xlim(s.round_down_to_nearest(min(beta)-10, 20), s.round_up_to_nearest(max(beta)+10, 20))
            ax[1].set_ylim(0, max_depth)
            ax[1].invert_yaxis() 
                     
        # Plot theoretical dispersion curve
        ax[0].plot(c_t, self.wavelength, c=col, label='Theoretical')
                
        # Plot shear wave velocity profile (central values)
        beta_plot = self._beta_profile(beta)
        z_plot = self._z_profile(z, max_depth)
        ax[1].plot(beta_plot, z_plot, c=col)

        if up_low:
            # Vertical error bars (boundaries for z)
            z_error_xdata = [0.5*(beta[i] + beta[i+1]) for i in range(n)]
            z_error_up = [profile['z_up'][i] - z[i] for i in range(n)]
            z_error_low = [z[i] - profile['z_low'][i] for i in range(n)]
            ax[1].errorbar(z_error_xdata, z, yerr=[z_error_low,z_error_up], c=col, barsabove=True, 
              ls='none', fmt='.', capsize=3)
            
            # Horizontal error bars (boundaries for Vs)
            z_temp =  [0] + [z[i] for i in range(n)] + [max(max_depth, z[-1])]
            beta_error_ydata = [0.5*(z_temp[i] + z_temp[i+1]) for i in range(n+1)]
            beta_error_up = [profile['beta_up'][i] - beta[i] for i in range(n+1)]
            beta_error_low = [beta[i] - profile['beta_low'][i] for i in range(n+1)]
            ax[1].errorbar(beta, beta_error_ydata, xerr=[beta_error_low,beta_error_up], c=col, barsabove=True, 
              ls='none', fmt='.', capsize=3)

        # Figure appearance
        fig.set_size_inches(s.cm_to_in(figwidth), s.cm_to_in(figheight))    
        
        if type(DC_yaxis) is str and DC_yaxis.lower() == 'linear':
            ax[0].set_yscale('linear')
        elif type(DC_yaxis) is str and DC_yaxis.lower() == 'log':
            ax[0].set_yscale('log')
        else:
            ax[0].set_yscale('linear')
            warnings.warn('Scale of wavelength axis (DC_yaxis) not recognized. Dispersion curves displayed on a linear wavelength scale (default setting).')

        # Create legend
        if show_legend:
            ax[0].legend(frameon=False)
         
        if return_axes:
            return fig, ax
        
        if return_ct:
            return c_t

    
    @staticmethod
    def _to_list(obj):
        
        """
        Return integer/float as a list. 
        
        Parameters
        ----------
        obj : int, float or list
            Object to reformat as a list. 
        
        Returns
        -------
        list
            Reformatted list.
        
        Raises 
        ------
        TypeError
            If 'obj' is not of type integer, float, or list.
        
        """
        if isinstance(obj, (int, float)):
            return [obj]
        elif isinstance(obj, list):
            return obj
        else:
            message = '´depth´ must be of type integer, float or list'
            TypeError(message)

    
    def compute_vsz(self, depth, beta_vec, layers, layer_parameter):
        
        """
        Compute the time-averaged shear wave velocity (Vsz) of the upper-most 
        z='depth' meters. If z exceeds the maximum depth of the interval 
        velocity profile described by beta_vec and layers, the profile is 
        extrapolated using the half-space velocity down to z m depth.
        
        Parameters
        ----------
        depth : int, float or list
            Depths [m]. Compute Vsz for z='depth'.
        beta_vec : list or numpy.ndarray
            Shear wave velocity array [m/s]
        layers : list or numpy.ndarray
            Layer thicknesses [m] (e.g., layers = [h1, h2, h3]) 
            or locations of layer interfaces [m] 
            (e.g., layers = [z1, z2, z3] = [h1, h1+h2, h1+h2+h3]).
        layer_parameter : {'h', 'z'}
            'h' if layer thicknesses are provided.
            'z' if locations of layer interfaces are provided.
        
        Returns
        -------
        Tuple
            Containing values of Vsz for z='depth'.
            Tuple[0] : list
                Depths [m].
            Tuple[1] : list          
                Average shear wave velocity of the top-most z meters [m/s].
        
        Raises
        ------
        ValueError
            If 'layer_parameter' is not specified as 'h' or 'z'.
            If locations of layer interfaces are not correctly specified 
            (the first element should be the depth of the first layer interface).
        
        """  
        # Check parameters
        if layer_parameter.lower() == 'h':
            h_vec = copy.deepcopy(layers)
            z_vec = self._h_to_z(h_vec, len(h_vec))
        elif layer_parameter.lower() == 'z':
            z_vec = copy.deepcopy(layers)
            h_vec = self._z_to_h(z_vec, len(z_vec))
        else:
            message = f'Layer parameter ´{layer_parameter}´ is not recognized.'
            raise ValueError(message)
        
        if z_vec[0] == 0:
            message = 'Locations of layer interfaces are not correctly specified.'
        
        # Compute the average shear wave velocity for depths 'depth'        
        depth = self._to_list(depth)
        if z_vec[-1] < depth[-1]:
            z_vec.append(depth[-1])
        vs_z = [None for i in range(len(depth))]
        for i in range(len(depth)):
            N = next(num for num, z_val in enumerate(z_vec) if z_val >= depth[i])
            if N == 0:
                vs_z[i] = beta_vec[0]
            else:
                vs_z[i] = depth[i]/(sum(np.divide(h_vec[0:N], beta_vec[0:N])) + (depth[i]-z_vec[N-1])/beta_vec[N])
               
        return (depth, vs_z)
    
    
    def save_to_pickle(self, file):
    
        """
        Pickle an inversion analysis object.
        
        Parameters
        ----------
        file : str
            Write the pickled representation of the inversion analysis object
            to the file object file.
        
        """
        if ( file[-2:] != '.p' and file[-7:] != '.pickle' ):
            file += '.p'
        
        pickle_out = open(file, "wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()    
        
