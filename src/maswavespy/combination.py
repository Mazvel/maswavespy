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
MASWavesPy combination

Combine experimental (observed) dispersion curves and assess uncertainties 
associated with the dispersion analysis process. The data processing is 
conducted as described in Olafsdottir et al. (2018b). 

References
----------
Dispersion curve combination
 - Olafsdottir, E.A., Bessason, B. and Erlingsson, S. (2018b). Combination of 
   dispersion curves from MASW measurements. Soil Dynamics and Earthquake 
   Engineering 113: 473–487. https://doi.org/10.1016/j.soildyn.2018.05.025

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from maswavespy import supplemental as s


class CombineDCs():
    
    """
    Class for creating composite dispersion curve objects.
    
    Instance attributes
    -------------------
    General site attributes:
    site : str
        Name of test site.
    profile : str 
        Identification code for measurement profile.
    metadata : str or dict
        Meta-information for object.    
    
    Experimental dispersion data:
    freq : numpy.ndarray
        Frequency array [Hz].
        (Array containing the frequency values of all observed elementary 
        dispersion curves.)
    c : numpy.ndarray
        Rayleigh wave phase velocity array [m/s].
        (Array containing the phase velocity values of all observed elementary 
        dispersion curves.)
    wavelength : numpy.ndarray
        Wavelength array [m/s]. 
        (Array containing the wavelength values of all observed 
        elementary dispersion curves.)
    cov : tuple
        Containing information on variation in estimated Rayleigh wave
        phase velocity values within each defined frequency bin.
    combined : dict 
        Dictionary containing a composite experimenal dispersion curve with 
        keys 'c_mean', 'c_std', 'c_up', 'c_low', 'wavelength' and 'no_points'.    
    resampled : dict 
        Dictionary containing the resampled composite experimenal dispersion curve 
        curve with keys 'c_mean', 'c_up', 'c_low' and 'wavelength'.    
    data_points : dict, optional             
        Dictionary containing coordinates of dispersion curve data points 
        that fall within each wavelength band with keys 'c' and 'wavelength'. 
    
    
    Instance methods
    ----------------
    dc_combination(self, a, no_std=1, return_points=False, q_min=-9, q_max=50)
        Add up elementary dispersion curves estimates obtained from multiple 
        shot gathers to obtain a composite experimental dispersion curve with
        upper/lower boundaries.
    
    dc_cov(self, binwidth=0.1)    
        Evaluate the variation in estimated Rayleigh wave phase velocity values
        in terms of COV (coefficient of variation).
        
    plot_combined_dc(self, plot_all=False, figwidth=8 , figheight=12, col='navy', pointcol='darkgray')
        Plot composite experimental dispersion curves.
    
    plot_dc_cov(self, figwidth=14, figheight=12, col='navy', alpha=0.6, **kwargs)
        Plot the variation in estimated Rayleigh wave phase velocity values  
        (COV) as a function of frequency.
    
    resample_dc(self, space='log', no_points=30, wavelength_min='default', wavelength_max='default', show_fig=True
        Resample composite experimental dispersion curves.
    
    
    Class methods
    -------------
    import_from_dict(cls, d, metadata=None)
        Initialize a composite dispersion curve object from a dictionary of 
        elementary dispersion curves.
    
    
    Static methods
    --------------    
    _check_dimensions(freq, c)
        Conduct general checks on the imported elementary dispersion curves.

    _check_identical(obj, description)
        Check if all elements of a list are equal. 

    _to_array(list_of_arrays)
        Create a 1D numpy array from a list of 1D numpy arrays.
    
    """
    
    def __init__(self, site, profile, freq, c, metadata=None):
        
        """
        Initialize a composite dispersion curve object.
        
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
        freq : list of numpy.ndarrays
            Elementary dispersion curves, list of frequency arrays.
        c : list of numpy.ndarrays
            Elementary dispersion curves, list of Rayleigh wave phase velocity arrays.
        metadata : str or dict, optional
            Meta-information for object. Default is metadata=None.
        
        Returns
        -------
        CombineDCs
            Initialized composite dispersion curve object.
        
        """
        self.site = site
        self.profile = profile
        self.metadata = metadata
        
        # Dispersion data
        freq, c = self._check_dimensions(freq, c)
        self.freq = self._to_array(freq)
        self.c = self._to_array(c)
        self.wavelengths = self.c/self.freq
        
        # Initiate dispersion processing attributes
        self.cov = None
        self.combined = {
                'c_mean' : None,
                'c_std' : None, 
                'c_up' : None,
                'c_low' : None,
                'wavelength' : None,
                'no_points' : None
                 }
        self.resampled = {
                'c_mean' : None,
                'c_up' : None,
                'c_low' : None,
                'wavelength' : None
                }
        self.data_points = {
                'c' : None,
                'wavelength' : None
                }


    @staticmethod
    def _check_dimensions(freq, c):
        
        """
        Conduct checks on inputs, i.e.
        -  Ensure that both lists (freq and c) contain the same number 
           of numpy arrays.
        -  Ensure that each pair of numpy arrays (freq[i] and c[i]) contains 
           the same number of elements.
        
        Parameters
        ----------
        freq : list of numpy.ndarrays
            Elementary dispersion curves, list of frequency arrays.
        c : list of numpy.ndarrays
            Elementary dispersion curves, list of Rayleigh wave phase velocity arrays.
        
        Returns
        -------
        Tuple
            Containing checked lists of numpy arrays.
            
        Raises
        ------
        ValueError
            If the lists freq and c do not contain the same number of elements
            or if there exists a pair of numpy arrays (freq[i] and c[i]) that
            does not contain the same number of elements.
        
        """        
        nlists = len(freq)
        if len(c) != nlists:
            message = f'Please check the imported dispersion curves. {len(freq)} frequency arrays and {len(c)} phase velocity arrays were provided.'
            raise ValueError(message)
            
        for i in range(nlists):
            if freq[i].shape[0] != c[i].shape[0]:
                message = f'Please check the imported dispersion curves (index {i}). ´freq[{i}]´ and ´c[{i}]´ do not contain the same number of elements.'
                raise ValueError(message)
        
        return (freq, c)
        

    @staticmethod
    def _to_array(list_of_arrays):
        
        """
        Create a 1D numpy array from a list of 1D numpy arrays.
        
        Parameters
        ----------
        list_of_arrays : list of numpy.ndarrays
            List of 1D numpy arrays. The numpy arrays may contain varying 
            number of elements.
            
        Returns
        -------
        array : numpy.ndarray
            1D numpy array created from list_of_arrays.
        
        """
        return np.array([i for array in list_of_arrays for i in array])
     
    
    @classmethod
    def import_from_dict(cls, d, metadata=None):
        
        """
        Initialize a composite dispersion curve object from a dictionary of 
        identified elementary dispersion curves. See also dataset.get_dcs.
        
        Parameters
        ----------
        d : dict
            Dictionary of elementdc_dicts. Required keys for each elementdc_dict
            are 'site', 'profile', 'f0' and 'c0'. See dispersion.return_to_dict for 
            further information on elementdc_dicts.
        metadata : str or dict, optional
            Meta-information for object. Default is metadata=None.
            
        Returns
        -------
        CombineDCs
            Initialized composite dispersion curve object.
        
        """
        # General information
        site_list = [d[i]['site'] for i in d]
        site = CombineDCs._check_identical(site_list, 'the site name')
        
        profile_list = [d[i]['profile'] for i in d]
        profile = CombineDCs._check_identical(profile_list, 'the profile ID')
        
        # Create lists of frequency and phase velocity arrays
        freq = [d[i]['f0'] for i in d]
        c = [d[i]['c0'] for i in d]
                
        return CombineDCs(site, profile, freq, c, metadata=metadata)
    
    
    @staticmethod
    def _check_identical(obj, description):
        
        """
        Check if all elements of a list are equal. 
        
        Parameters
        ----------
        obj : list
            List to be checked.
        description : str
            Description of the list.
        
        Returns
        -------
        obj[0] : str or num
            The first element of the checked list.
        
        Raises
        ------
        ValueError
            If the elements of obj are not all equal. 

        """
        if not obj.count(obj[0]) == len(obj):
            message = f'Please check the imported dictionary. Different values are provided for {description}.'
            raise ValueError(message)
        else:        
            return obj[0]
    
        
    def dc_cov(self, binwidth=0.1):
        
        """              
        Evaluate the variation in experimental dispersion curve estimates  
        obtained from repeated measurements (shot gathers). The variability
        is evaluated in terms of the coefficient of variation (COV=std/mean) 
        of the estimated phase velocity values within each frequency bin  
        (of width binwidth). The coefficient of variation is only computed for 
        frequency bins that contain two or more dispersion curve data points.
        
        Parameters
        ----------
        binwidth : int or float, optional
            Width of frequency bins [Hz].  
            Default is binwidth=0.1. 
        
        Returns
        -------
        Tuple
            Containing information on variation in estimated Rayleigh wave
            phase velocity values within each frequency bin.
            
            Tuple[0] : numpy.ndarray            
                Frequency, reference value (mid-point) for each frequency bin [Hz].
            Tuple[1] : numpy.ndarray
                Mean phase velocity (for each frequency bin) [m/s]
            Tuple[2] : numpy.ndarray
                Phase velocity standard deviation (for each frequency bin) [m/s]            
            Tuple[3] : numpy.ndarray
                Coefficient of variation [-].
           
        """   
        # Compute frequency values and round to the nearest binwidth
        f_vec_round = s.round_to_nearest(self.freq, binwidth)
        
        # Sort dispersion curve data points by frequency values
        f_sort = np.array(sorted(f_vec_round, reverse=False))
        c_sort = np.array([x for (y,x) in sorted(zip(f_vec_round,self.c), key=lambda pair: pair[0])])
    
        # Find the unique elements of f_sort
        # f_unique[0], sorted unique values 
        # f_unique[1], number of times each of the unique values comes up in the original array
        f_unique = np.unique(f_sort, return_counts=True)
        
        # Compute the mean phase velocity, standard deviation and COV for each frequency bin
        # that contains two or more dispersion curve data points
        mean_c = np.zeros(len(f_unique[1],))
        std_c = np.zeros(len(f_unique[1],))
        cov = np.zeros(len(f_unique[1],))
        
        loc = 0
        for i in range(len(f_unique[1])):
            c_temp = c_sort[range(loc,loc+f_unique[1][i])]
            if len(c_temp) > 1: 
                mean_c[i] = np.mean(c_temp, dtype=np.float64)
                std_c[i] = np.std(c_temp, ddof=1, dtype=np.float64)
                cov[i] = std_c[i]/mean_c[i]
            loc += f_unique[1][i]   
           
        # Remove zero values and return to CombineDCs object
        ref = np.nonzero(cov)
        self.cov = (f_unique[0][ref], mean_c[ref], std_c[ref], cov[ref])
        
        print("Mean and std DCs (frequency domain) computed.")


    def plot_dc_cov(self, figwidth=14, figheight=12, col='navy', alpha=0.6, **kwargs):
        
        """
        Visualize the variation in experimental dispersion curve estimates  
        obtained from repeated measurements (shot gathers).
        
        - Plot all (elementary) dispersion curve estimates in the frequency - 
          phase velocity domain.
        - Plot the coefficient of variation (COV) as a function of frequency.
        
        Parameters
        ----------
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=14. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.           
        col : a Matplotlib color or sequence of color, optional
            Marker color.
            Default is col='navy'.
        alpha : scalar, optional
            Alpha blending value, between 0 (transparent) and 1 (opaque).
            Default is alpha=0.6
        
        Other parameters
        ----------------
        All other keyword arguments are passed on to the matplotlib.pyplot.scatter 
        function. See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
        for a list of valid kwargs for the scatter command. 
        
        """           
        fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)))
        
        # Plot (elementary) dispersion curve estimates (f-c domain)
        ax[0].scatter(self.freq, self.c, c=col, marker='o', alpha=alpha, **kwargs)        
        
        # Axes labels and limits 
        ax[0].set_xlim(s.round_down_to_nearest(min(self.freq), 5), s.round_up_to_nearest(max(self.freq), 5))
        ax[0].set_ylim(s.round_down_to_nearest(min(self.c)-5, 10), s.round_up_to_nearest(max(self.c)+5, 10))
        ax[0].set_ylabel('Phase velocity [m/s]', fontweight='bold')    
        
        # Figure appearance
        ax[0].set_axisbelow(True)
        ax[0].grid(color='gainsboro', linestyle=':')
        
        
        # Plot COV as a function of frequency
        ax[1].plot(self.cov[0], self.cov[3], c=col)
        ax[1].scatter(self.cov[0], self.cov[3], c=col, marker='o', alpha=alpha, **kwargs)  

        # Axes labels and limits 
        ax[1].set_xlim(s.round_down_to_nearest(min(self.freq), 5), s.round_up_to_nearest(max(self.freq), 5))
        ax[1].set_ylim(0, s.round_up_to_nearest(max(self.cov[3])+0.001, 0.02))
        ax[1].set_xlabel('Frequency [Hz]', fontweight='bold')
        ax[1].set_ylabel('COV [-]', fontweight='bold')        

        # Figure appearance
        ax[1].set_axisbelow(True)
        ax[1].grid(color='gainsboro', linestyle=':')

        fig.set_tight_layout(True)


    def dc_combination(self, a, no_std=1, return_points=False, q_min=-9, q_max=50):
    
        """
        Add up elementary dispersion curves retrieved from multiple shot gathers 
        to obtain a composite experimental dispersion curve with upper/lower 
        boundaries. (A dispersion curve retrieved from a single multi-channel 
        record is here referred to as an elementary dispersion curve.)
        
        The composite dispersion curve is obtained according to the 
        computational procedure presented in Olafsdottir et al. (2018b). 
        The elementary dispersion curve data points are grouped together within 
        logarithmically (i.e., log_a) spaced wavelength intervals. The upper  
        and lower boundary curves correspond to plus/minus no_std standard 
        deviations from the mean dispersion curve. The default value for 
        no_std is 1.
        
        Parameters
        ----------
        a : int or float
            Combination parameter. The elementary dispersion curve estimates are 
            added up within log_a spaced wavelength intervals. Present experience
            indicates that values in the range of a=2.5 to a=5 are appropriate for 
            most sites. See further in Olafsdottir et al. (2018b), 
            https://doi.org/10.1016/j.soildyn.2018.05.025
        no_std : int or float, optional
            Number of standard deviations. The upper and lower boundary 
            dispersion curves are defined as plus/minus no_std standard 
            deviations from the mean dispersion curve.
            Default is no_std=1.
        return_points : boolean, optional
            Specifies whether coordinates of dispersion curve data points should 
            be returned (yes=True, no=False).
            Default is return_points=False.
        q_min : int or float, optional
            Determines the lower bound for dispersion curve wavelengths used in 
            subsequent analysis, lambda_min=2**((2q_min-3)/2a) [m].
            Default is q_min=-9.
        q_max : int or float, optional
            Determines the upper bound for dispersion curve wavelengths used in 
            subsequent analysis, lambda_max=2**((2q_max-1)/2a) [m].
            Default is q_max=50.
        
        Returns
        -------
        combined : dict 
            Dictionary containing a composite experimental dispersion curve with 
            keys 'c_mean', 'c_std', 'c_up', 'c_low', 'wavelength' and 'no_points'.
    
            combined['c_mean'] : numpy.ndarray
                Average Rayleigh wave velocity [m/s].
            combined['c_std'] : numpy.ndarray
                Rayleigh wave velocity standard deviation [m/s].
            combined['c_up'] : numpy.ndarray
                Upper boundary Rayleigh wave velocity (mean + no_std * std) [m/s]
            combined['c_low'] : numpy.ndarray
                Lower boundary Rayleigh wave velocity (mean - no_std * std) [m/s]
            combined['wavelength'] : numpy.ndarray            
                Central wavelength [m].     
                The geometric mean of each wavelength interval.     
            combined['no_points'] : numpy.ndarray
                Number of points within each wavelength interval.
                combined['no_points'][i] is the number of dispersion curve 
                data points that fall within the interval those central 
                wavelength is combined['wavelength'][i].
                
        data_points : dict, optional
            Dictionary containing coordinates of dispersion curve data points 
            with keys 'c' and 'wavelength'. 
            data_points is returned if return_points=True.

            data_points['c'] : list of numpy.ndarrays
                Rayleigh wave phase velocity [m/s] of dispersion curve data 
                points within each wavelength band. 
                data_points['c'][i] is a numpy.ndarray of Rayleigh wave phase 
                velocity values within the i-th (non-empty) interval.
            data_points['wavelength'] : list of numpy.ndarray
                Wavelength [m] of dispersion curve data points within each 
                wavelength band. 
                data_points['wavelength'][i] is a numpy.ndarray of wavelengths 
                within the i-th (non-empty) interval. 
            
        """
        # Define wavelengths intervals
        nPoints_initial = len(np.arange(q_min, q_max + 1))
        lambda_center = np.zeros(nPoints_initial)
        lambda_low = np.zeros(nPoints_initial)
        lambda_up = np.zeros(nPoints_initial)
    
        third = 2 ** (1 / a)
        sixth = 2 ** (1 / (2 * a))
    
        m = 0
        lambda_center[m] = 2 ** ((q_min - 1) / a)  # Center wavelength
        lambda_low[m] = lambda_center[m] / sixth   # Lower boundary
        lambda_up[m] = lambda_center[m] * sixth    # Upper boundary 
        for i in range((q_min + 1),(q_max + 1)):
            m += 1
            lambda_center[m] = lambda_center[m-1] * third
            lambda_low[m] = lambda_center[m] / sixth
            lambda_up[m] = lambda_center[m] * sixth
       
        # Find elementary dispersion curve data points that fall within 
        # each wavelength interval 
        c_sum  = np.zeros(nPoints_initial)
        c_sq  = np.zeros(nPoints_initial)
        no_points = np.zeros(nPoints_initial)
        if return_points is True:
            DataPoints_c = [None] * nPoints_initial
            DataPoints_lambda = [None] * nPoints_initial
    
        for i in range(len(self.wavelengths)):
            for j in range(nPoints_initial):
                # Data point within wavelength interval j
                if (lambda_low[j] <= self.wavelengths[i] and self.wavelengths[i] < lambda_up[j]):
                    c_sum[j] += self.c[i]
                    c_sq[j] += self.c[i] ** 2
                    # Coordinates of dispersion curve data points 
                    if return_points is True:
                        if (no_points[j] == 0):
                            DataPoints_c[j] = self.c[i]
                            DataPoints_lambda[j] = self.wavelengths[i]
                        else:
                            DataPoints_c[j] = np.append(DataPoints_c[j], self.c[i])
                            DataPoints_lambda[j] = np.append(DataPoints_lambda[j], self.wavelengths[i])                  
                    # Number of data points within wavelength interval j
                    no_points[j] += 1
    
        # Compute the mean phase velocity value and standard deviation for each
        # wavelength interval that contains at least two data points 
        c_mean = np.zeros(nPoints_initial)
        c_std = np.zeros(nPoints_initial)
        for i in range(nPoints_initial):
            if no_points[i] > 1: # Wavelength intervals that include at least two data points
                c_mean[i] = c_sum[i] / no_points[i]
                c_std[i] =  np.sqrt((c_sq[i]- no_points[i] * c_mean[i] ** 2) / (no_points[i] - 1))
            elif no_points[i] == 1:
                c_mean[i] = 0
                c_std[i] = 0
    
        for i in range(nPoints_initial):
            if c_std[i] == 0 and c_mean[i] != 0:
                c_mean[i] = 0
    
        # Remove zero values and return composite dispersion curve  
        ref = np.nonzero(c_mean)
        self.combined['c_mean'] = c_mean[ref]
        self.combined['c_std'] = c_std[ref]
        self.combined['c_up'] = c_mean[ref] + no_std * c_std[ref]
        self.combined['c_low'] = c_mean[ref] - no_std * c_std[ref]
        self.combined['wavelength'] = lambda_center[ref]
        self.combined['no_points'] = no_points[ref]
        
        # Return data points  
        if return_points is True:
            self.data_points['c'] = [i for i in DataPoints_c if i is not None]
            self.data_points['wavelength'] = [i for i in DataPoints_lambda if i is not None]
    
        print("Composite DC (wavelength domain) computed with a = " + str(a) + ".")


    def plot_combined_dc(self, plot_all=False, figwidth=8 , figheight=12, col='navy', pointcol='darkgray'):
        
        """
        Plot the composite experimental dispersion curve and its upper/lower 
        boundary curves. In addition, the elementary dispersion curve data points  
        can be shown (optional setting). 
        
        Parameters
        ----------
        plot_all : boolean, optional
            Plot elementary dispersion curve data points (yes=True, no=False).
            Default is plot_all=False.
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=8. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.
        col : a Matplotlib color or sequence of color, optional
            Linecolor.
            Default is col='navy'.
        pointcol : a Matplotlib color or sequence of color, optional
            Marker color.
            Default is col='darkgray'.
            
        """
        # Figure settings
        fig = plt.figure(figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)))
        ax = fig.add_subplot(1, 1, 1)
    
        # Plot elementary dispersion curve data points
        if plot_all == 1:
            ax.plot(self.c, self.wavelengths, 'o', ms=4, color=pointcol, label='Data points')
    
        # Plot the composite experimental dispersion curve (mean \pm no_std * std)
        ax.plot(self.combined['c_mean'], self.combined['wavelength'],'-',
                color=col, lw=1, label='Mean')   
        ax.plot(self.combined['c_up'], self.combined['wavelength'], '-.', 
                color=col, lw=1, label='Upper/lower')
        ax.plot(self.combined['c_low'], self.combined['wavelength'], '-.', 
                color=col, lw=1)
    
        # Axis limits and axis labels 
        ax.set_xlim(s.round_down_to_nearest(min(self.combined['c_low']), 25), 
                    s.round_up_to_nearest(max(self.combined['c_up']), 25))
        ax.set_ylim(s.round_down_to_nearest(min(self.combined['wavelength']), 5), 
                    s.round_up_to_nearest(max(self.combined['wavelength']), 5))
        plt.xlabel('Rayleigh wave velocity [m/s]', fontweight='bold')
        plt.ylabel('Wavelength [m]', fontweight='bold')
        ax.invert_yaxis() 
    
        # Figure appearance
        ax.set_axisbelow(True)
        ax.grid(color='gainsboro', linestyle=':')
        plt.legend(loc='lower left', frameon=False)
        fig.set_tight_layout(True) 
    
    
    def resample_dc(self, space='log', no_points=30, wavelength_min='default', wavelength_max='default', show_fig=True):
        
        """
        Resample the composite experimental dispersion curve and its upper and 
        lower boundary curves at no_points logarithmically or linearly spaced points 
        within the interval of [wavelength_min, wavelength_max]. By default, 
        wavelength_min = combined['wavelength'][0] and wavelength_max = combined['wavelength'][-1]. 
        Visually compare the original and resampled dispersion curves (optional).
        
        Parameters
        ----------
        space : {'log', 'linear'}, optional
            - If space='log', the dispersion curve is resampled at wavelength points 
              that are spaced evenly on a logarithmic scale (a geometric progression).
            - If space='linear', the dispersion curve is resampled at wavelength points 
              that are spaced evenly on a linear scale (an arithmetic progression).        
            Default is space='log'.
        wavelength_min : 'default', float or int
            - If wavelength_min = 'default' the minimum wavelength of the resampled DC
              is set equal to combined['wavelength'][0]
            - Else, the minimum wavelength is set as wavelength_min. An error is returned
              if wavelength_min < combined['wavelength'][0].
            Default is wavelength_min = 'default'
        wavelength_max : 'default', float or int
            - If wavelength_max = 'default' the maximum wavelength of the resampled DC
              is set equal to combined['wavelength'][-1].
            - Else, the maximum wavelength is set as wavelength_max. An error is returned
              if wavelength_max > combined['wavelength'][-1].
            Default is wavelength_max = 'default'
        no_points : int, optional
            Number of sample points. 
            Default is no_points=30.
        show_fig : boolean, optional
            Plot the original and resampled dispersion curves (yes=True, no=False).
            Default is show_fig=True.
            
        Returns
        -------
        resampled : dict 
            Dictionary containing the resampled composite experimental dispersion 
            curve with keys 'c_mean', 'c_up', 'c_low' and 'wavelength'.
    
            resampled['c_mean'] : numpy.ndarray
                Average Rayleigh wave velocity [m/s].
            resampled['c_up'] : numpy.ndarray
                Upper boundary Rayleigh wave velocity [m/s].
            resampled['c_low'] : numpy.ndarray
                Lower boundary Rayleigh wave velocity [m/s].
            resampled['wavelength'] : numpy.ndarray            
                Wavelength [m].

        Raises
        ------
        ValueError
            If 'space' is not specified as 'log' or 'linear'
            
        """
        spaces = ['log', 'linear']
        if space.lower() not in spaces:
            message = f'space must be specified as ´log´ or ´linear´, not as ´{space}´'
            raise ValueError(message)

        f_dc_mean = interp1d(np.round(self.combined['wavelength'],4), self.combined['c_mean'], kind='linear')
        f_dc_low = interp1d(np.round(self.combined['wavelength'],4), self.combined['c_low'], kind='linear')
        f_dc_up = interp1d(np.round(self.combined['wavelength'],4), self.combined['c_up'], kind='linear')
        
        wavelength_min = self._check_wavelength(wavelength_min, 'min')
        wavelength_max = self._check_wavelength(wavelength_max, 'max')
        if space.lower() == 'log':
            lambda_interp = np.round(np.geomspace(wavelength_min, wavelength_max, num=no_points, endpoint=True),4)
            space_print = ' logarithmically spaced'
        elif space.lower() == 'linear':
            lambda_interp = np.round(np.linspace(wavelength_min, wavelength_max, num=no_points, endpoint=True),4)
            space_print = ' linearly spaced'

        self.resampled['wavelength'] = lambda_interp
        self.resampled['c_mean'] = f_dc_mean(lambda_interp)
        self.resampled['c_low'] = f_dc_low(lambda_interp)
        self.resampled['c_up'] = f_dc_up(lambda_interp)
        
        # Plot - visually compare original and resampled dispersion curves    
        if show_fig:
            self.plot_combined_dc()
            plt.plot(self.resampled['c_mean'], lambda_interp, 'o', ms=4, 
                     c='red', label='Mean (resampled)')
            plt.plot(self.resampled['c_low'], lambda_interp, 'o', ms=4, 
                     markerfacecolor='None', markeredgecolor='red', label='Upper/lower (resampled)')
            plt.plot(self.resampled['c_up'], lambda_interp, 'o', ms=4, 
                     markerfacecolor='None', markeredgecolor='red')
            plt.legend(loc='lower left', frameon=False)
    
        print('Composite DC (wavelength domain) resampled at ' + str(no_points) + space_print + ' points between wavelengths of ' + str(wavelength_min) + ' m and ' + str(wavelength_max) + ' m.')

   
    def _check_wavelength(self, wavelength, boundary):
        
        """
        Check boundary (min and max) wavelength values for resampled DC.

        Parameters
        ----------
        wavelength : 'default', float or int
            Wavelength value to be checked/retrieved.
        boundary : 'min' or 'max'
            - boundary = 'min' to check the lower boundary wavelength value of the resampled DC.
            - boundary = 'max' to check the upper boundary wavelength value of the resampled DC.     

        Returns
        -------
        wavelength_value : float or int
            Checked wavelength value.

        Raises
        ------
        ValueError
            If wavelength is not specified as 'default', float or int.
            If the specified wavelength value falls outside the range of the calculated composite DC.
        
        """
        if isinstance(wavelength,str) and wavelength.lower() == 'default':
            if boundary == 'min':
                wavelength_value = min(np.round(self.combined['wavelength'],4))
            elif boundary == 'max':
                wavelength_value = max(np.round(self.combined['wavelength'],4))
        elif isinstance(wavelength,float) or isinstance(wavelength,int):
            wavelength_value = wavelength
        else:
            message = 'Please check the wavelength boundaries (wavelength_min and wavelength_max). They must be specified as ´default´ or be given a numerical value.'
            raise ValueError(message)

        # Ensure that numerically specified wavelength_values fall within the range of 
        # [combined['wavelength'][0], combined['wavelength'][-1]].
        if boundary == 'min':
            if wavelength_value < min(np.round(self.combined['wavelength'],4)):
                min_value = min(np.round(self.combined['wavelength'],4))
                message = f'The value specified for ´wavelength_min´ is outside the wavelength range of the calculated composite DC. Its minimum wavelength is {min_value} m.'
                raise ValueError(message)
        if boundary == 'max':
            if wavelength_value > max(np.round(self.combined['wavelength'],4)):
                max_value = max(np.round(self.combined['wavelength'],4))
                message = f'The value specified for ´wavelength_max´ is outside the wavelength range of the calculated composite DC. Its maximum wavelength is {max_value} m.'
                raise ValueError(message)               

        return wavelength_value