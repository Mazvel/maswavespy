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
MASWavesPy

Supplemental routines for MASWavesPy Wavefield, Dispersion, Combination and Inversion.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def cm_to_in(x):
    
    """
    Convert length values expressed in centimeters [cm] to their equivalent in 
    inches [in].
    
    Parameters
    ----------
    x : int or float
        Length in centimeters [cm].
        
    Returns
    -------       
    float
        Length x expressed in inches [in].
    
    """
    return (x / 2.54)


def round_to_nearest(x, base):
    
    """
    Round x to the nearest base. Both integer and non-integer (i.e., fractional)
    bases are allowed. If x is an array, each element of x will be rounded 
    to the nearest base. 
    
    Parameters
    ----------
    x : int, float, list or numpy.ndarray
        Number or array.
    base : int or float
        Round to the nearest base.    
    
    Returns
    ------- 
    x_round : float or numpy.ndarray
        x (or each element of x) rounded to the nearest base. 
    
    """
    # Count the number of decimal digits in base
    if isinstance(base, int):
        prec = 0
    else:
        prec = len(str(base).split(".")[1])
    
    # Round x to the nearest base. The returned value is expressed with 
    # a precision of prec decimals. 
    x_round = (base * (np.array(x) / base).round()).round(prec)

    return x_round


def round_up_to_nearest(x, base):
    
    """
    Round x up to the nearest base. Both integer and non-integer (i.e., fractional)
    bases are allowed. If x is an array, each element of x will be rounded 
    up to the nearest base. 
    
    Parameters
    ----------
    x : int, float, list or numpy.ndarray
        Number or array.
    base : int or float
        Round up to the nearest base.     
    
    Returns
    ------- 
    x_round : float or numpy.ndarray
        x (or each element of x) rounded up to the nearest base. 
    
    """
    # Count the number of decimal digits in base
    if isinstance(base, int):
        prec = 0
    else:
        prec = len(str(base).split(".")[1])
    
    # Round x up to the nearest base. The returned value is expressed with 
    # a precision of prec decimals. 
    x_round = (base * np.ceil((np.array(x) / base))).round(prec)

    return x_round


def round_down_to_nearest(x, base):
    
    """
    Round x down to the nearest base. Both integer and non-integer (i.e., fractional)
    bases are allowed. If x is an array, each element of x will be rounded 
    down to the nearest base. 
    
    Parameters
    ----------
    x : int, float, list or numpy.ndarray
        Number or array.
    base : int or float
        Round down to the nearest base.
        
    Returns
    ------- 
    x_round : float or numpy.ndarray
        x (or each element of x) rounded down to the nearest base. 
    
    """
    # Count the number of decimal digits in base
    if isinstance(base, int):
        prec = 0
    else:
        prec = len(str(base).split(".")[1])
    
    # Round x down to the nearest base. The returned value is expressed with 
    # a precision of prec decimals. 
    x_round = (base * np.floor((np.array(x) / base))).round(prec)

    return x_round


def plot_lines(xs, ys, z, ax=None, **kwargs):
    
    """
    Plot multiple lines (ys versus xs) in a single set of axes using
    Matplotlib' Line Collection. The line color(s) are set according 
    to the list of scalars z.

    Parameters
    ----------
    xs : list or numpy.ndarray
        x-coordinates of all line segments.
        
        xs[i] : list or numpy.ndarray
            Line no. i, array of x-coordinates.
    
    ys : list or numpy.ndarray
        y-coordinates of all line segments.
        
        ys[i] : list or numpy.ndarray
            Line no. i, array of y-coordinates.
    
    z : numpy.ndarray or range
        Determines the color of each line segment. 
        The color of the line segment defined in terms of xs[i] and ys[i]  
        is set by z[i]. The values of z are mapped to RGBA colors from the 
        given colormap.
    ax : axes object, optional 
        The axes of the subplot. 
        Default is ax=None (current pyplot axes will be used).

    Returns
    -------
    line_segments : LineCollection
        An initialized LineCollection instance. See
        https://matplotlib.org/3.1.1/api/collections_api.html#matplotlib.collections.LineCollection
        for information on LineCollections.
    
    Other parameters
    ----------------
    All other keyword arguments are passed on to matplotlib.collections.LineCollection. 
    See https://matplotlib.org/3.1.1/api/collections_api.html#matplotlib.collections.LineCollection
    for a list of valid kwargs. 
    
    """
    # Initiation
    if ax is None:
        ax = plt.gca()
        
    # Reformat lists of coordinates and create a LineCollection 
    coordinates = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    line_segments = LineCollection(coordinates, **kwargs)

    # Set the coloring of each line according to z
    line_segments.set_array(np.asarray(z))

    # Add lines to axes and autoscale axes limits 
    ax.add_collection(line_segments)
    ax.autoscale()

    return line_segments

