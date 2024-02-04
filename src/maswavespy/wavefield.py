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
MASWavesPy wavefield

Import recorded multi-channel time series and transform the recorded data 
into the frequency-phase velocity domain. The data processing is conducted by 
application of the phase-shift method (Park et al. 1998). The dispersion 
analysis computational tools are based on those presented in 
Olafsdottir et al. (2018).

References
----------
Phase-shift method
 - Park, C.B., Miller, R.D. and Xia J. (1998). Imaging dispersion curves 
   of surface waves on multi-channel record. In SEG technical program 
   expanded abstracts 1998, New Orleans, LA, pp. 1377–1380. 
   https://doi.org/10.1190/1.1820161
MASWaves (MATLAB implementation)
 - Olafsdottir, E.A., Erlingsson, S. and Bessason, B. (2018). 
   Tool for analysis of multichannel analysis of surface waves (MASW) 
   field data and evaluation of shear wave velocity profiles of soils. 
   Canadian Geotechnical Journal 55(2): 217–233. 
   https://doi.org/10.1139/cgj-2016-0302
   
"""

import numpy as np
import cmath
import matplotlib.pyplot as plt
import obspy
import copy

from maswavespy import supplemental as s
from maswavespy.dispersion import ElementDC

# Dispersion imaging (cython optimization)
from maswavespy import cy_dispersion_imaging as cy_di


class RecordMC():
   
    """
    Class for creating multi-channel record objects.
   
    Instance attributes
    -------------------
    General site attributes:
        site : str
            Name of test site.
        profile : str
            Identification code for measurement profile.
        file_name : str
            Path of file with recorded seismic data.
        metadata : str or dict
            Meta-information for object.
       
    Data acquisition parameters:        
        traces : numpy.ndarray
            Multi-channel surface wave record consisting of n traces.
            Time-histories of ground motion recorded at n equally spaced points
            on the ground surface.
        n : int
            Number of receivers/geophones.
        direction : str
            Direction of measurement, 'forward' or 'reverse'.
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        fs : float
            Sampling frequency [Hz].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve [Hz].


    Instance methods
    ----------------  
    dispersion_imaging(self, cT_min, cT_max, cT_step)
        Transform the recorded wavefield into the frequency-phase velocity domain. 
        The transformation visualizes the energy density of the acquired data 
        from which modal dispersion curves can be identified.  
        
    dispersion_imaging_cy(self, cT_min, cT_max, cT_step)
        Cython optimization of dispersion_imaging.
            
    element_dc(self, cT_min, cT_max, cT_step, cython_optimization=True)
        Conduct the dispersion imaging (by using a phase shift transform) of 
        the multi-channel record and initialize an elementary dispersion curve 
        (ElementDC) object.

    plot_data(self, du=1.25, filled=True, normalized=True, translated=False, figwidth=9, figheight=12)
        Visualize the acquired wavefield as a wiggle trace display.  
   
    _direction_record(self)
        Ensure that acquired surface wave registrations have the correct form
        for dispersion processing.         
 
    
    Class methods
    -------------
    import_from_textfile(cls, site, profile, file_name, header_lines, n, direction, dx, x1, fs, f_pick_min, metadata=None, **kwargs)
        Initialize a multi-channel record object from a text file.
       
    import_from_waveform(cls, site, profile, file_name, n, direction, dx, x1, fs, f_pick_min, metadata=None, **kwargs)
        Initialize a multi-channel record object from a waveform file.
   
       
    Static methods
    --------------    
    _check_direction_str(direction)
        Ensure that the direction of measurement is specified as 'forward'
        or 'reverse'.

    _check_number_of_channels(traces, n)
        Ensure that the number of surface wave traces is equal to the
        specified number of receivers.

    """

    def __init__(self, site, profile, traces, n, direction, dx, x1, fs, f_pick_min, file_name=None, metadata=None):      
       
        """
        Initialize a multi-channel record object.
       
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str
            Identification code for measurement profile.
        traces : numpy.ndarray
            Multi-channel surface wave record consisting of n traces, time domain.
            Each trace must be stored in a separate column, i.e.,
             - traces[:,0] time-domain registrations from receiver 1 (channel 1)
             - traces[:,1] time-domain registrations from receiver 2 (channel 2)
             - ...
             - traces[:,(n-1)] time-domain registrations from receiver n (channel n)
            All traces must be of equal length.            
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'}
            Direction of measurement.            
            - 'forward': Forward measurement.
               Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse': Reverse (backward) measurement.
               Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        fs : float
            Sampling frequency [Hz].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve (depends on
            the geophones' natural frequency) [Hz]. Spectral maxima
            (experimental dispersion curves) are only identified at
            frequencies higher than f_pick_min.
        file_name : str, optional
            Path of file with recorded seismic data.
            Default is file_name=None.
        metadata : str or dict, optional
            Meta-information for object. Additional information about recorded data.
            Default is None.
           
        Returns
        -------
        RecordMC
            Initialized multi-channel record object.
       
        """        
        # General information on test site and recorded data
        self.site = site
        self.profile = profile
        self.file_name = file_name
        self.metadata = metadata
       
        # Acquisition parameters
        self.traces = traces
        self.n =self._check_number_of_channels(traces, n)
        self.direction = self._check_direction_str(direction)
        self.dx = dx
        self.x1 = x1
        self.fs = fs
        self.f_pick_min = f_pick_min


    @staticmethod
    def _check_direction_str(direction):
       
        """
        Ensure that the direction of measurement is specified as 'forward'
        or 'reverse'.
       
        Parameters
        ----------
        direction : str
            Specified direction of measurement.
           
        Returns
        -------
        direction : str
            Checked direction. If the string 'direction' contains upper case
            letters they are converted to lower case.  
       
        Raises
        ------
        ValueError
            If 'direction' is not specified as 'forward' or 'reverse'
       
        """
        directions = ['forward', 'reverse']
        if direction.lower() not in directions:
            message = f'direction must be specified as ´forward´ or ´reverse´, not as ´{direction}´'
            raise ValueError(message)
        else:
            return direction.lower()


    @staticmethod
    def _check_number_of_channels(traces, n):
       
        """
        Ensure that the number of surface wave traces is equal to the
        specified number of receivers.
       
        Parameters
        ----------
        traces : numpy.ndarray
            Multi-channel surface wave record.
        n : int
            Specified number of receivers.
           
        Returns
        -------
        n : int
            Checked number of receivers (number of imported traces).  

        Raises
        ------
        ValueError
            If the number of imported surface wave traces is not equal to the
            specified number of receivers (n).
                 
        """                      
        if traces.shape[1] != int(n):
            message = f'{traces.shape[1]} traces were imported but the number of receivers is specified as {n}'
            raise ValueError(message)
        else:
            return int(n)      


    @classmethod
    def import_from_textfile(cls, site, profile, file_name, header_lines, n, direction, dx, x1, fs, f_pick_min, metadata=None, **kwargs):  
       
        """
        Initialize a multi-channel record object from a text file.
       
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str
            Identification code for measurement profile.
        file_name : str
            Path of file with recorded seismic data.
            Requirements:            
            - By default, it is assumed that the recorded data is stored in
              a whitespace/tab-delimited text file. If a different string is
              used to separate values, it can be passed to the loadtxt command
              as an additional keyword argument (e.g., delimiter=',' for
              a comma-separated textfile).
            - The number of header lines, including comments, is header_lines.
            - Each trace must be stored in a single column.
            - All traces must be of equal length and missing values are not
              allowed (each row in the text file must include the same number
              of values).
        header_lines : int
            Number of header lines, including comments.
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'}
            Direction of measurement.
            - 'forward': Forward measurement.
               Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse': Reverse (backward) measurement.
               Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        fs : float
            Sampling frequency [Hz].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve (depends on
            the geophones' natural frequency) [Hz]. Spectral maxima
            (experimental dispersion curves) are only identified at
            frequencies higher than f_pick_min.
        metadata : str or dict, optional
            Meta-information for object. Additional information about recorded data.
            Default is metadata=None.
       
        Returns
        -------
        RecordMC
            Initialized multi-channel record object.
           
        Other parameters
        ----------------
        All other keyword arguments are passed on to numpy.loadtxt.
        See https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html
        for a list of valid kwargs for the loadtxt command.
       
        """        
        # Import data from text file
        traces = np.loadtxt(file_name, skiprows=header_lines, **kwargs)
           
        # Initialize the multi-channel record object.
        return RecordMC(site, profile, traces, n, direction, dx, x1, fs, f_pick_min, file_name, metadata)  
   
   
    @classmethod
    def import_from_waveform(cls, site, profile, file_name, n, direction, dx, x1, fs, f_pick_min, metadata=None, **kwargs):
       
        """
        Initialize a multi-channel record object from a waveform file.
       
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str
            Identification code for measurement profile.
        file_name : str
            Path of file with recorded seismic data.  
            See https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
            for a list of supported file formats.
        n : int
            Number of receivers.
        direction : {'forward', 'reverse'}  
            Direction of measurement.
            - 'forward': Forward measurement.
               Seismic source is applied next to receiver 1 (channel 1).
            - 'reverse': Reverse (backward) measurement.
               Seismic source is applied next to receiver n (channel n).
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        fs : float
            Sampling frequency [Hz].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve (depends on
            the geophones' natural frequency) [Hz]. Spectral maxima
            (experimental dispersion curves) are only identified at
            frequencies higher than f_pick_min.
        metadata : str or dict, optional
            Meta-information for object. Additional information about recorded data.
            Default is metadata=None.
       
        Returns
        -------
        RecordMC
            Initialized multi-channel record object.
           
        Other parameters
        ----------------
        All other keyword arguments are passed on to obspy.core.stream.read.
        See https://docs.obspy.org/packages/autogen/obspy.core.stream.read.html
        for a list of valid kwargs for the obspy.read command.
       
        """    
        # Import data from file
        record = obspy.read(file_name, **kwargs)
        traces = np.column_stack([trace.data for trace in record])
       
        # Initialize the multi-channel record object.
        return RecordMC(site, profile, traces, n, direction, dx, x1, fs, f_pick_min, file_name, metadata)  
       
           
    def plot_data(self, du=1.25, filled=True, normalized=True, translated=False, figwidth=9, figheight=12):

        """
        Visualize the acquired wavefield as a wiggle trace display.  
               
        Parameters
        ----------
        du : float, optional
            Scale factor for offset between wiggles.
            Default is du=1.25.
        filled : boolean, optional
            - For variable area wiggle traces, filled=True.
              The positive part of the wiggle curve (i.e., to the right of
              the zero line) is filled.
            - For wiggle traces, filled=False.
              The area under the wiggle trace is unfilled.    
            Default is filled=True.      
        normalized : boolean, optional
            - To normalize the amplitude of each trace, normalized=True.
            - No normalization, normalized=False.
            Default is normalized=True.
        translated : boolean, optional
            - To translate each seismic trace so that its average amplitude  
              equals zero, translated=True.
            - No translation, translated=False.
            Default is translated=False.
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=9.
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=12.
       
        """      
        # Figure settings
        fig = plt.figure(figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)))
        ax = fig.add_subplot(1, 1, 1)
       
        # Acquired time-series and array of receivers
        u_plot = copy.deepcopy(self.traces)
        receiver = np.linspace(0, self.n-1, self.n).astype('int')
       
        # Recording time (pre-trigger + post-trigger recording length) [s]
        t_max = len(self.traces[:,1]) / self.fs - 1 / self.fs
        t = np.linspace(0, t_max, num=len(self.traces[:,1]))
       
        # Normalization
        if normalized:
            for i in range(self.n):
                u_plot[:,i] = u_plot[:,i] / np.amax(u_plot[:,i])
       
        # Translation
        if translated:
            for i in range(self.n):
                u_plot[:,i] = u_plot[:,i] - np.mean(u_plot[:,i])
         
        # Plot surface wave traces
        offsets = du * np.amax(u_plot[:,1]) * np.linspace(0, self.n-1, self.n)
        for i in range(self.n):
            trace = offsets[i] * np.ones(len(u_plot[:,1])) + u_plot[:,i]
            ax.plot(trace, t, 'k-', linewidth = 0.5)
            if filled:
                ax.fill_betweenx(t, offsets[i], trace, where=(trace > offsets[i]), color='k')
   
        # Axis limits and axis labels
        plt.xlim(-3 * np.amax(u_plot[:,1]) * du, (self.n + 2) * np.amax(u_plot[:,1]) * du)
        # Reverse shot
        if self.direction == 'reverse':
            plt.xticks(offsets[-1::-5], self.x1 + self.dx * receiver[0::5])                    
        # Forward shot
        else:    
            plt.xticks(offsets[0::5], self.x1 + self.dx * receiver[0::5])    
        plt.xlabel('Distance from impact load point [m]', fontweight='bold')
       
        plt.ylim(0, t[-1])
        plt.ylabel('Time [s]', fontweight='bold')
        ax.invert_yaxis()        
 
        # Figure appearance
        plt.tight_layout()
             
        
    def _direction_record(self):
        
        """
        Ensure that acquired surface wave registrations have the correct form 
        for dispersion processing. 
        
        Returns
        -------
        u : numpy.ndarray
            Checked multi-channel surface wave record, i.e.
            
            - u[:,0] time-domain registrations from the receiver that is closest 
              to the impact load point (channel 1 for 'forward' measurements and 
              channel n for 'reverse' measurements)
            - ...
            - u[:,(n-1)] time-domain registrations from the receiver that is 
              furthest away from the impact load point (channel n for 'forward'
              measurements and channel 1 for 'reverse' measurements).
              
        """
        # Reverse shot
        if self.direction == 'reverse':
            u = np.zeros([len(self.traces), self.n])
            for i in range(self.n):
                u[:,i] = self.traces[:,(self.n - (i + 1))]
            return u.astype(np.double)
        # Forward shot (no change required)
        else:
            return self.traces.astype(np.double)
    
    
    def dispersion_imaging(self, cT_min, cT_max, cT_step):
        
        """
        Transform the recorded wavefield into the frequency - phase velocity 
        domain. The transformation visualizes the energy density of 
        the acquired data from which modal dispersion curves can be identified.
        
        Parameters
        ----------
        cT_min : float
            Minimum testing Rayleigh wave phase velocity [m/s].
        cT_max : float
            Maximum testing Rayleigh wave phase velocity [m/s].
        cT_step : float
            Testing Rayleigh wave phase velocity increment [m/s].
        
        Returns
        ------- 
        f : numpy.ndarray
            Frequency array [Hz].
        c : numpy.ndarray 
            Rayleigh wave phase velocity array [m/s].
        A : numpy.ndarray
            Slant-stacked amplitude matrix. Summed (slant-stacked) amplitudes 
            corresponding to different couples (f,c).
        
        """          
        # Import and check acquired surface wave registrations
        u = self._direction_record()
        
        # Location of receivers, distance from impact load point [m]
        x = np.arange(self.x1, self.x1 + self.n * self.dx, self.dx)
        
        # Converting measuring frequency (Hz to rad/sec)
        omega_fs = 2 * np.pi * float(self.fs) 
       
        # Apply discrete Fourier transform to the time axis of u
        U = np.fft.fft(u, axis=0)
    
        # Number of samples in each transformed trace
        u_len = U.shape[0]
    
        # Compute the phase spectrum of U
        i = cmath.sqrt(-1)
        P = np.exp(i * (-1) * np.angle(U))
    
        # Frequency range for U
        omega = np.arange(0, omega_fs, (omega_fs / u_len))
        f = omega / (2 * np.pi) # [Hz]
        
        # Rayleigh wave phase velocity testing range
        c = np.arange(cT_min, cT_max + cT_step, cT_step, dtype='float64')
        c_len = c.shape[0]
        
        # Compute the slant-stack (summed) amplitude corresponding to each set of
        # omega and cT, A(omega,cT).
        A = np.zeros((u_len, c_len), dtype='float64')   
        for k in range(u_len): # Frequency component k
            for m in range(c_len): # Testing phase velocity component m
                # Determining the amount of phase shifts required to counterbalance
                # the time delay corresponding to specific offsets for a given set 
                # of omega and c
                delta = omega[k] / c[m]
                # Obtaining the (normalized) slant-stack amplitude corresponding
                # to each set of omega and c
                A[k,m] = (abs(sum(np.exp((-1) * i * delta * x) * P[k,:])) / self.n).real
    
        return f,c,A 

    def dispersion_imaging_cy(self, cT_min, cT_max, cT_step):
        
        """
        Cython optimization of dispersion_imaging (recommended).
        
        Transform the recorded wavefield into the frequency - phase velocity 
        domain. The transformation visualizes the energy density of 
        the acquired data from which modal dispersion curves can be identified.
        
        Parameters
        ----------
        cT_min : float
            Minimum testing Rayleigh wave phase velocity [m/s].
        cT_max : float
            Maximum testing Rayleigh wave phase velocity [m/s].
        cT_step : float
            Testing Rayleigh wave phase velocity increment [m/s].
        
        Returns
        ------- 
        f : numpy.ndarray
            Frequency array [Hz].
        c : numpy.ndarray 
            Rayleigh wave phase velocity array [m/s].
        A : numpy.ndarray
            Slant-stacked amplitude matrix. Summed (slant-stacked) amplitudes 
            corresponding to different couples (f,c).
        
        """  
        # Import and check acquired surface wave registrations
        u = self._direction_record()
        
        # Dispersion imaging
        f,c,A = cy_di.dispersion_imaging_cy(u, self.n, self.dx, self.x1, self.fs, cT_min, cT_max, cT_step)
        
        return f,c,A 
        
               
    def element_dc(self, cT_min, cT_max, cT_step, cython_optimization=True):
        
        """
        Conduct the dispersion imaging (by using a phase shift transform) of 
        the multi-channel record and initialize an elementary dispersion curve 
        (ElementDC) object.
        
        Parameters
        ----------
        cT_min : float
            Minimum testing Rayleigh wave phase velocity [m/s].
        cT_max : float
            Maximum testing Rayleigh wave phase velocity [m/s].
        cT_step : float
            Testing Rayleigh wave phase velocity increment [m/s].
        cython_optimization : boolean, optional
            - To use a cython optimization of dispersion_imaging, 
              cython_optimization=True (recommended).
            - No optimization, cython_optimization=False.
            Default is cython_optimization=True.
        
        Returns
        ------- 
        ElementDC
            Initialized elementary dispersion curve object.
        
        """  
        # Dispersion imaging (phase shift transform)
        if cython_optimization:    
            f,c,A = self.dispersion_imaging_cy(cT_min, cT_max, cT_step)
        else:
            f,c,A = self.dispersion_imaging(cT_min, cT_max, cT_step)
        
        return ElementDC(self.site, self.profile, self.direction, self.dx, self.x1, self.f_pick_min, f, c, A)