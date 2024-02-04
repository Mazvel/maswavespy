# distutils: language = c++

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
#
"""
MASWavesPy Dispersion - dispersion imaging

Transform multi-channel surface wave recordings from the space-time domain 
to the frequency domain by appication of the phase-shift method (Park et al. 1998).

References
----------
Phase-shift method
 - Park, C.B., Miller, R.D. and Xia J. (1998). Imaging dispersion curves 
   of surface waves on multi-channel record. In SEG technical program 
   expanded abstracts 1998, New Orleans, LA, pp. 1377–1380. 
   doi:10.1190/1.1820161.
   
"""

import numpy as np
cimport numpy as np

cdef extern from "<math.h>" namespace "std":
    double atan(double)

cdef extern from "<complex.h>" namespace "std":
    double complex exp(double complex)
    double abs(double complex)
    double arg(double complex)
    
cimport cython

ctypedef np.double_t DTYPE_t
ctypedef np.complex128_t DTYPE_c

cdef double complex I = -1j


@cython.boundscheck(False) # Turn off bounds-checking
@cython.wraparound(False) # Turn off negative index wrapping
def dispersion_imaging_cy(np.ndarray[DTYPE_t, ndim=2] u, int n, double dx, double x1, 
                          double fs, double cT_min, double cT_max, double cT_step):
    
    """
    Transform the recorded wavefield into the frequency - phase velocity domain. 
    The transformation visualizes the energy density of the acquired data 
    from which modal dispersion curves can be identified. 
    Cython optimization of dispersion_imaging for enhanced computational speed.
        
    Parameters
    ----------
    u : numpy.ndarray
        Multi-channel surface wave record.
    n : int
        Number of receivers/geophones.
    dx : float
        Receiver spacing [m].
    x1 : float
        Source offset [m].
    fs : float
        Sampling frequency [Hz].
    cT_min : float
        Minimum testing Rayleigh wave phase velocity [m/s].
    cT_max : float
        Maximum testing Rayleigh wave phase velocity [m/s].
    cT_step : float
        Testing Rayleigh wave phase velocity increment [m/s].
    
    Returns
    ------- 
    f : numpy.ndarray
        Frequency vector [Hz].
    c : numpy.ndarray 
        Rayleigh wave phase velocity vector [m/s].
    A : numpy.ndarray
        Slant-stacked amplitude matrix. Summed (slant-stacked) amplitudes 
        corresponding to different couples (f,c).
    
    """   
    cdef:
        int k, m, q
        double omega_fs, delta, n_inv
        double complex add, temp
    
    # Location of receivers, distance from seismic source [m]
    cdef np.ndarray[double, ndim=1] x
    x = np.arange(x1, x1 + n * dx, dx, dtype=np.double)
    
    # Converting measuring frequency (Hz to rad/sec)
    omega_fs = 8 * atan(1) * fs 
   
    # Apply discrete Fourier transform to the time axis of u
    cdef np.ndarray[DTYPE_c, ndim=2] U
    U = np.fft.fft(u,axis=0)
    cdef np.int u_len = U.shape[0]
    
    # Phase angle (angular component) of U
    cdef np.ndarray[double, ndim=2] Uarg
    Uarg = np.empty((u_len,n), dtype=np.double)
    for m in range(u_len):
        for q in range(n):
            Uarg[m,q] = arg(U[m,q])
            
    # Frequency range for U
    cdef np.ndarray[DTYPE_t, ndim=1] f
    f = np.fft.fftfreq(u_len, d=1./fs)
    cdef np.int f_len = f.shape[0]
    
    cdef np.ndarray[DTYPE_t, ndim=1] omega
    omega = np.empty((f_len,), dtype=np.double)
    for k in range(f_len):
        omega[k] = 8 * atan(1) * f[k]
    
    # Rayleigh wave phase velocity testing range
    cdef np.ndarray[DTYPE_t, ndim=1] c 
    c = np.arange(cT_min, cT_max + cT_step, cT_step, dtype=np.double)
    cdef np.int c_len = c.shape[0]
    
    # Compute the slant-stack (summed) amplitude corresponding to each set of
    # omega and cT, A(omega,cT).
    cdef np.ndarray[DTYPE_t, ndim=2] A 
    A = np.empty((u_len, c_len), dtype=np.double)
    n_inv = 1.0/n   
    for m in range(u_len): # Frequency component j
        for k in range(c_len): # Testing phase velocity component k
            temp = 0.0+0.0j
            # Determining the amount of phase shifts required to counterbalance
            # the time delay corresponding to specific offsets for a given set 
            # of omega and c
            delta = omega[m] / c[k]
            # Obtaining the (normalized) slant-stack amplitude corresponding
            # to each set of omega and c            
            for q in range(n):
                add = exp(I * (Uarg[m,q] + delta * x[q]))
                temp = temp + add
            A[m,k] = abs(temp) * n_inv
            
    return f, c, A
