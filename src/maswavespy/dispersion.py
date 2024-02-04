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
MASWavesPy dispersion

Identify (elementary) experimental dispersion curves from multi-channel
surface wave registrations. The data processing is conducted by 
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
import matplotlib.pyplot as plt
import tkinter as tk

from maswavespy import supplemental as s
from maswavespy.select_dc import SelectDC
        
        
class ElementDC():

    """
    Class for creating elementary dispersion curve objects.
    
    Instance attributes
    -------------------
    General site and record identification attributes:
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
        direction : str
            Direction of measurement, 'forward' or 'reverse'.
        dx : float
            Receiver spacing [m].
        x1 : float
            Source offset [m].
        f_pick_min : int or float
            Lower boundary of the geophones' response curve [Hz].

    Dispersion processing attributes:
        f : numpy.ndarray
            Frequency array [Hz].
        c : numpy.ndarray
            Rayleigh wave phase velocity array [m/s].
        A : numpy.ndarray
            Slant-stacked amplitude matrix. Summed (slant-stacked) amplitudes 
            corresponding to different couples (f,c).
        fplot : numpy.ndarray
            Frequency range of the dispersion image [Hz].
        cplot : numpy.ndarray
            Velocity range of the dispersion image [m/s].
        Aplot : numpy.ndarray
            Summed (slant-stack) amplitudes within the frequency range of
            [f_min, f_max].
        Amax : tuple
            Containing identified spectral maxima.
        f0 : numpy.ndarray
            Fundamental mode dispersion curve, frequency array [Hz].
        c0 : numpy.ndarray
            Fundamental mode dispersion curve, Rayleigh wave phase velocity 
            array [m/s].


    Instance methods
    ----------------    
    find_spectral_maxima(self)
        Identify spectral maxima at frequencies higher than f_pick_min.
        
    label_spectral_maxima(self, Amax, ax=None, return_text=False, **kwargs)
        Number spectral maxima (superimpose on dispersion image).
    
    pick_dc(self, f_min, f_max)
        Identify (elementary) dispersion curves from multi-channel 
        surface wave registrations. Opens a GUI for dispersion curve selection.
    
    plot_dispersion_image(self, f_min, f_max, col_map='jet', resolution=100, figwidth=14, figheight=8, fig=None, ax=None, tight_layout=True) 
        Plot the two-dimensional dispersion image (phase velocity spectrum) 
        of the recorded seismic wavefield.
    
    plot_spectral_maxima(self, Amax, ax=None, color='k', edgecolor='face', point_size=9, return_paths=False, **kwargs)
        Plot spectral maxima (superimpose on dispersion image).

    return_to_dict(self)
        Return an identified elementary dispersion curve as a dictionary.
     
            
    """
    
    def __init__(self, site, profile, direction, dx, x1, f_pick_min, f, c, A):      
        
        """
        Initialize an elementary dispersion curve object.
        
        Parameters
        ----------
        site : str
            Name of test site.
        profile : str 
            Identification code for measurement profile.
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
        f_pick_min : int or float
            Lower boundary of the geophones' response curve (depends on 
            the geophones' natural frequency) [Hz]. Spectral maxima 
            (experimental dispersion curves) are only identified at 
            frequencies higher than f_pick_min.
        f : numpy.ndarray
            Frequency array [Hz].
        c : numpy.ndarray 
            Rayleigh wave phase velocity array [m/s].
        A : numpy.ndarray
            Slant-stacked amplitude matrix. Summed (slant-stacked) amplitudes 
            corresponding to different couples (f,c).
            
        Returns
        ------- 
        ElementDC
            Initialized elementary dispersion curve object.
        
        """        
        # General information on test site and recorded data
        self.site = site
        self.profile = profile
        self.direction = direction
        self.dx = dx
        self.x1 = x1
        self.f_pick_min = f_pick_min
       
        self.f = f
        self.c = c
        self.A = A
        self.fplot = None
        self.cplot = None
        self.Aplot = None
        self.Amax = None
        self.f0 = None
        self.c0 = None


    def plot_dispersion_image(self, f_min, f_max, col_map='jet', resolution=100, figwidth=14, figheight=8, fig=None, ax=None, tight_layout=True):
    
        """           
        Plot the two-dimensional dispersion image (phase velocity spectrum) 
        of the recorded seismic wavefield. The slant-stacked amplitude (A) 
        is presented in the frequency - phase velocity - normalized amplitude 
        domain using a color scale.   
        
        Parameters
        ----------
        f_min : int or float
            Lower limit of frequency axis [Hz].
        f_max : int or float
            Upper limit of frequency axis [Hz].
        col_map : str or Colormap, optional
            Registered colormap name or a Colormap instance.
            Default is col_map='jet'.
        resolution : int, optional
            Number of contour lines.
            Default is resolution=100.
        figwidth : int or float, optional
            Width of figure in centimeters [cm].
            Default is figwidth=14. 
        figheight : int or float, optional
            Height of figure in centimeters [cm].
            Default is figheight=8.
        fig : figure, optional
            A figure object.
            Default is fig=None (a new figure object will be created).
        ax : axes object, optional 
            The axes of the subplot. 
            Default is ax=None (current pyplot axes will be used).
        tight_layout : boolean, optional
            Use matplotlib.pyplot.tight_layout() to adjust subplot params 
            (yes=True, no=False).
            Default is tight_layout=True.           
        
        Returns
        -------
        fplot : numpy.ndarray
            Frequency range of the dispersion image [Hz]. 
        cplot : numpy.ndarray
            Velocity range of the dispersion image [m/s]. 
        Aplot : numpy.ndarray
            Summed (slant-stack) amplitudes within the frequency range of
            [f_min, f_max].
        
        """      
        # Construct frequency and phase velocity matrices
        f2 = np.array([self.f,]*self.A.shape[1]).transpose()
        c2 = np.array([self.c,]*self.A.shape[0])
             
        # Limits of frequency axis
        delta_fmin = abs(f2[:,0] - f_min)
        no_fmin = np.where(delta_fmin == delta_fmin.min())
        delta_fmax = abs(f2[:,0] - f_max)
        no_fmax = np.where(delta_fmax == delta_fmax.min())
    
        # Get data for plotting
        self.Aplot = self.A[no_fmin[0][0]:(no_fmax[0][0] + 1), :] 
        self.fplot = f2[no_fmin[0][0]:(no_fmax[0][0] + 1), :]
        self.cplot = c2[no_fmin[0][0]:(no_fmax[0][0] + 1), :]
        
        # Figure settings
        if fig is None:   
            fig = plt.figure(figsize=(s.cm_to_in(figwidth), s.cm_to_in(figheight)))
            fig.add_subplot(1, 1, 1)  
        if ax is None:
            ax = plt.gca()
        
        # Plot spectral image
        cf = ax.contourf(self.fplot, self.cplot, self.Aplot, resolution, cmap=col_map)
        cb = fig.colorbar(cf, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], orientation='vertical')
        cb.set_label('Normalized amplitude', fontweight='bold')
        
        # Axis limits and axis labels 
        ax.set_xlim(round(self.fplot[0,1],0), round(self.fplot[-1,1],0))   
        ax.set_xlabel('Frequency [Hz]', fontweight='bold')
        ax.set_ylim(self.cplot[0,0], self.cplot[0,-1])
        ax.set_ylabel('Phase velocity [m/s]', fontweight='bold')
        
        # Figure appearance
        ax.grid(color='lightgray', linestyle=':')
        fig.set_tight_layout(tight_layout)


    def find_spectral_maxima(self):
    
        """
        Identify spectral maxima at frequencies higher than f_pick_min.
            
        Returns
        -------
        Tuple
            Containing identified spectral maxima.
            
            Tuple[0] : numpy.ndarray
                Frequency values [Hz]. 
            Tuple[1] : numpy.ndarray
                Phase velocity values [m/s].
        
        """           
        # Normalize slant-stacked amplitudes at each frequency
        Anorm = np.zeros(np.shape(self.Aplot))
        for i in range(self.Aplot.shape[0]):
            Anorm[i,:] = self.Aplot[i,:] / np.amax(self.Aplot[i,:])
        
        # Identify spectral maxima
        f_loc, c_loc = np.where(Anorm == 1)        
        Amax_fvec = np.zeros(len(f_loc))
        Amax_cvec = np.zeros(len(c_loc))
        for i in range(len(f_loc)):
            Amax_fvec[i] = self.fplot[f_loc[i],1]
            Amax_cvec[i] = self.cplot[1,c_loc[i]]
        
        # Locate the spectral maximum at each frequency higher then f_pick_min
        ii = np.where(Amax_fvec > self.f_pick_min) 
        Amax_fvec = Amax_fvec[ii]
        Amax_cvec = Amax_cvec[ii]
        
        # Sort in order of increasing frequency
        index_array = np.argsort(Amax_fvec)
        Amax_fvec = Amax_fvec[index_array]
        Amax_cvec = Amax_cvec[index_array]
        
        return (Amax_fvec, Amax_cvec)


    def plot_spectral_maxima(self, Amax, ax=None, color='k', edgecolor='face', point_size=9, return_paths=False, **kwargs):
        
        """
        Plot spectral maxima (superimpose on dispersion image).
    
        Parameters
        ----------
        Amax : tuple
            Containing identified spectral maxima.
            Amax[0] : numpy.ndarray
                Frequency values [Hz]. 
            Amax[1] : numpy.ndarray
                Phase velocity values [m/s]. 
        ax : axes object, optional 
            The axes of the subplot.
            Default is ax=None (current pyplot axes will be used).
        color : {'none'} or a Matplotlib color/sequence of color, optional
            The marker color.
            Default is color='k' (black).
        edgecolor : {'face', 'none'} or a Matplotlib color/sequence of color, optional
            The edge color of the marker.
            Default is edgecolor='face' (the edge color is the same as the face color).
        point_size : scalar, optional
            The marker size in points**2.
            Default is point_size=9.
        return_paths : boolean, optional
            If return_paths is True, the PathCollection Amax_paths is returned.  
            Default is return_paths=False, i.e., Amax_paths is not returned.
        
        Returns
        -------
        Amax_paths : PathCollection, optional
            See https://matplotlib.org/api/collections_api.html#matplotlib.collections.PathCollection
            for information on PathCollections.
        
        Other parameters
        ----------------
        All other keyword arguments are passed on to the matplotlib.pyplot.scatter 
        function. See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html
        for a list of valid kwargs for the scatter command. 
        
        """      
        # Initiation
        if ax is None:
            ax = plt.gca()

        # Plot spectral maxima
        Amax_paths = ax.scatter(Amax[0], Amax[1], s=point_size, color=color, edgecolor=edgecolor, **kwargs)

        if return_paths:
            return Amax_paths


    def label_spectral_maxima(self, Amax, ax=None, return_text=False, **kwargs):
        
        """
        Number spectral maxima (superimpose on dispersion image).
    
        Parameters
        ----------
        Amax : tuple
            Containing identified spectral maxima.
            Amax[0] : numpy.ndarray
                Frequency values [Hz]. 
            Amax[1] : numpy.ndarray
                Phase velocity values [m/s]. 
        ax : axes object, optional 
            The axes of the subplot.
            Default is ax=None (current pyplot axes will be used).  
        return_text : boolean, optional
            If return_text is True, a list of matplotlib.text.Text instances, 
            labels_text, is returned. 
            Default is return_text=False, i.e., labels_text is not returned.           

        Returns
        -------
        labels_text : list of text instances, optional
            See https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text
            for information on matplotlib.text.Text instances.
        
        Other parameters
        ----------------
        All other keyword arguments are passed on to the matplotlib.text.Text
        class. See https://matplotlib.org/3.1.1/api/text_api.html#matplotlib.text.Text
        for a list of valid kwargs for the Text class.     
        
        """   
        # Initiation
        if ax is None:
            ax = plt.gca()        
        
        # Label spectral maxima
        labels = [str(i) for i in range(len(Amax[0]))]
        labels_text = []
        for i in range(len(Amax[0])):
            labels_text.append(ax.text(Amax[0][i], Amax[1][i]+1, labels[i], **kwargs))
            
        if return_text:
            return labels_text
        
    
    def pick_dc(self, f_min, f_max):
        
        """
        Identify (elementary) dispersion curves from multi-channel 
        surface wave registrations. Opens a GUI for dispersion curve selection.
        
        Parameters
        ----------
        f_min : int or float
            Lower limit of frequency axis [Hz].
        f_max : int or float
            Upper limit of frequency axis [Hz].   

        """          
        print("A GUI for dispersion curve selection has been opened in a separate window. \n")
        # Open GUI
        app = SelectDC(self, f_min, f_max, master=tk.Tk())
        app.mainloop()
        
    
    def return_to_dict(self):
        
        """
        Return an identified elementary dispersion curve as a dictionary.
        
        Returns
        -------
        elementdc_dict : dict
            Dictionary containing an elementary dispersion curve and general
            information on the measurement site and profile configuration.
            The keys of elementdc_dict are the following: 'site', 'profile',
            'filename', 'direction', 'dx', 'x1', 'f0' and 'c0'.
        
        """
        elementdc_dict = {
                'site' : self.site,
                'profile' : self.profile,
                'direction' : self.direction,
                'dx' : self.dx,
                'x1' : self.x1,
                'f0' : self.f0,
                'c0' : self.c0
                }
                
        return elementdc_dict
    
         