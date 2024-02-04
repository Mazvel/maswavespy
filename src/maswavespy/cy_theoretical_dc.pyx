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
MASWavesPy Inversion - computation of theoretical dispersion curves

Compute theoretical fundamental mode Rayleigh wave dispersion curves for 
a linear elastic semi-infinite layered medium. Each layer (including the 
half-space) is assumed to be flat and have homogeneous and isotropic properties. 
Computations are based on the fast delta matrix algorithm (Buchen and 
Ben-Hador, 1996).

References
----------
Fast delta matrix algorithm
 - Buchen, P.W. & Ben-Hador, R. (1996). Free-mode surface-wave computations. 
   Geophysical Journal International, 124(3), 869–887. 
   doi:10.1111/j.1365-246X.1996.tb05642.x. 

"""

import numpy as np
cimport numpy as np

cdef extern from "<math.h>" namespace "std":
    double fabs(double)
    double pow(double, double)
    double atan(double)

cdef extern from "<complex.h>" namespace "std":
    double real(double complex)
    double imag(double complex)
    double complex sqrt(double complex)
    double complex cosh(double complex)
    double complex sinh(double complex)

cimport cython

ctypedef np.double_t DTYPE_t


@cython.boundscheck(False) # Turn off bounds-checking
@cython.wraparound(False) # Turn off negative index wrapping
def compute_fdma(np.ndarray[DTYPE_t, ndim=1] c_test, double c_step, np.ndarray[DTYPE_t, ndim=1] wavelength,
                 int n, np.ndarray[DTYPE_t, ndim=1] alpha, np.ndarray[DTYPE_t, ndim=1] beta, 
                 np.ndarray[DTYPE_t, ndim=1] rho, np.ndarray[DTYPE_t, ndim=1] h, double delta_c):

    """
    Compute the fundamental mode Rayleigh wave dispersion curve for 
    the stratified soil model defined by n, alpha, beta, rho and h 
    at wavelengths wavelength. Computations are conducted
    using the fast delta matrix algorithm (fdma). 
    
    Parameters
    ----------
    c_test : numpy.ndarray of type double
        Rayleigh wave phase velocity, testing values [m/s].
    c_step : double
        Testing Rayleigh wave phase velocity increment [m/s].
    wavelength : numpy.ndarray of type double
        Theoretical dispersion curve, wavelength array [m].
    n : int
        Number of finite thickness layers.
    alpha : numpy.ndarray of type double
        Compressional wave velocity vector [m/s] (array of size (n+1,)).
    beta : numpy.ndarray of type double
        Shear wave velocity vector [m/s] (array of size (n+1,)).
    rho : numpy.ndarray of type double
        Mass density vector [kg/m^3] (array of size (n+1,)).
    h : numpy.ndarray of type double
        Layer thickness vector [m] (array of size (n,)).
    delta_c : double
        Zero search initiation parameter [m/s].
        At wave number k_{i} the zero search is initiated at a phase velocity 
        value of c_{i-1} - delta_c, where c_{i-1} is the theoretical Rayleigh 
        wave phase velocity value at wave number k_{i-1}.
           
    Returns
    ------- 
    c_t : numpy.ndarray of type double
        Theoretical dispersion curve, Rayleigh wave phase velocity array [m/s].

    """
    cdef:
        int i, m, q, m_loc, delta_m
        double c2, G02, eta, a, ak, b, bk, ki, D, signD, sign_old, tol
        double complex X0, X1, X2, X3, X4, r, s, krh, ksh, C_alpha, C_beta, S_alpha, S_beta, p1, p2, p3, p4, q1, q2, q3, q4, y1, y2, z1, z2

    cdef np.int c_test_len = c_test.shape[0]
    cdef np.int wavelength_len = wavelength.shape[0]
    cdef np.int alpha_len = alpha.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=1] c_t
    c_t = np.empty((wavelength_len,), dtype=np.double)

    cdef np.ndarray[DTYPE_t, ndim=1] alpha2
    alpha2 = np.empty((n+1,), dtype=np.double)

    cdef np.ndarray[DTYPE_t, ndim=1] beta2
    beta2 = np.empty((n+1,), dtype=np.double)

    cdef np.ndarray[DTYPE_t, ndim=1] epsilon
    epsilon = np.empty((n,), dtype=np.double)

    for i in range(n):
        epsilon[i] = rho[i+1] / rho[i]

    for i in range(alpha_len):
        alpha2[i] = pow(alpha[i], 2)
        beta2[i] = pow(beta[i], 2)

    # For each wave number ki, compute the dispersion function using
    # increasing values of c_test until its value has a sign change.
    # At wave number ki the search is initiated at c_test = c_t[i-1] - delta_c
    m_loc = 0
    delta_m = round(delta_c / c_step)
    signD = 0
    tol = 0.001
    
    for i in range(wavelength_len):
        ki = (8 * atan(1)) / wavelength[i]
    
        for m in range(m_loc, c_test_len):
            c2 = pow(c_test[m], 2)

            # Initialize the layer recursion vector
            G02 = pow(rho[0] * beta2[0], 2)
            X0 = G02 * (2 * (2 - c2 / beta2[0]))
            X1 = G02 * (-(2 - c2 / beta2[0]) ** 2)
            X2 = 0
            X3 = 0
            X4 = -4 * G02

            # Conduct the layer recursion
            for q in range(n):

                # Compute layer parameters
                eta = 2 / c2 * (beta2[q] - epsilon[q] * beta2[q+1])
                a = epsilon[q] + eta
                ak = a - 1
                b = 1 - eta
                bk = b - 1

                # Compute layer eigenfunctions
                r = sqrt(1 - c2 / alpha2[q])
                krh = ki * r * h[q]
                C_alpha = cosh(krh)
                S_alpha = sinh(krh)

                s = sqrt(1 - c2 / beta2[q])
                ksh = ki * s * h[q]
                C_beta = cosh(ksh)
                S_beta = sinh(ksh)

                # Update the elements of the layer recursion vector
                p1 = C_beta * X1 + s * S_beta * X2
                p2 = C_beta * X3 + s * S_beta * X4
                p3 = (1 / s) * S_beta * X1 + C_beta * X2
                p4 = (1 / s) * S_beta * X3 + C_beta * X4

                q1 = C_alpha * p1 - r * S_alpha * p2
                q2 = -(1 / r) * S_alpha * p3 + C_alpha * p4

                y1 = ak * X0 + a * q1
                y2 = a * X0 + ak * q2

                z1 = b * X0 + bk * q1
                z2 = bk * X0 + b * q2

                X0 = bk * y1 + b * y2
                X1 = a * y1 + ak * y2
                X2 = epsilon[q] * (C_alpha * p3 - r * S_alpha * p4)
                X3 = epsilon[q] * (-(1 / r) * S_alpha * p1 + C_alpha * p2)
                X4 = bk * z1 + b * z2
            
            r = sqrt(1.0 - c2 / alpha2[n])
            s = sqrt(1.0 - c2 / beta2[n])

            # Compute the value of the dispersion function and extract its sign
            D = csgn(X1 + s * X2 - r * (X3 + s * X4))

            # Determine whether the sign of the dispersion function has changed
            if m == m_loc:
                sign_old = fdma(c_test[0], ki, n, alpha, beta, rho, h, alpha2, beta2)
            else:
                sign_old = signD
            signD = D
            if -1 - tol < sign_old*signD < -1 + tol:
                c_t[i] = c_test[m]
                m_loc = m - delta_m
                if m_loc < 0:
                    m_loc = 0
                break

    return c_t


@cython.cdivision(True) # Turn off zero division checking
cdef double sgn(double num):

    """
    Sign function (signum function). The function sgn extracts the sign of 
    the real number num. sgn(num) is defined as 0 if num equals 0. 
    
    Parameters
    ----------
    num : double
        Real number.
          
    Returns
    -------       
    1 if num is greater than 0.
    0 if num equals 0.
    -1 if num is less than 0.

    """
    if num == 0:
        return 0
    return num/fabs(num)


cdef double csgn(double complex num):

    """
    Sign function (signum function) for complex numbers. The function csgn 
    determines in which half-plane ("left" or "right") the complex number 
    num is. sgn(num) is defined as 0 if num equals 0+0j
    
    Parameters
    ----------
    num : complex double
        Complex number.
    
    Returns
    -------       
    1 if (i) real(num) > 0 OR (ii) real(num) = 0 and imag(num) > 0 
    -1 if (i) real(num) < 0 OR (ii) real(num) = 0 and imag(num) < 0 
    0 if num = 0 + 0j
    
    """
    if real(num) > 0:
        return 1
    elif real(num) < 0:
        return -1
    return sgn(imag(num))


@cython.boundscheck(False) # Turn off bounds-checking
@cython.wraparound(False) # Turn off negative index wrapping
def fdma(double c, double k, int n, np.ndarray[DTYPE_t, ndim=1] alpha, np.ndarray[DTYPE_t, ndim=1] beta,
         np.ndarray[DTYPE_t, ndim=1] rho, np.ndarray[DTYPE_t, ndim=1] h, 
         np.ndarray[DTYPE_t, ndim=1] alpha2, np.ndarray[DTYPE_t, ndim=1] beta2):

    """
    Construct the Rayleigh wave dispersion function.
    
    Computes the value of the Rayleigh wave dispersion function for the ordered 
    couple (c,k) and returns its sign. Computations are based on
    the fast delta matrix algorithm (fdma).
    
    Parameters
    ----------
    c : double
        Rayleigh wave phase velocity [m/s].
    k : double
        Wave number.
    n : int
        Number of finite thickness layers.
    alpha : numpy.ndarray of type double
        Compressional wave velocity vector [m/s] (array of size (n+1,)).
    beta : numpy.ndarray of type double
        Shear wave velocity vector [m/s] (array of size (n+1,)).
    rho : numpy.ndarray of type double
        Mass density vector [kg/m^3] (array of size (n+1,)).
    h : numpy.ndarray of type double
        Layer thickness vector [m] (array of size (n,)).
    alpha2 : numpy.ndarray of type double
        Compressional wave velocity vector [m] with each element raised to 
        the power of 2 (array of size (n+1,)).
    beta2 : numpy.ndarray of type double
        Shear wave velocity vector [m] with each element raised to 
        the power of 2 (array of size (n+1,)).
        
    Returns
    -------  
    signD : double
        The sign of the dispersion function value for the ordered couple (c,k).
    
    """
    cdef:
        int i
        double c2, G02, eta, a, ak, b, bk, epsilon
        double complex X0, X1, X2, X3, X4, r, s, krh, ksh, C_alpha, C_beta, S_alpha, S_beta, p1, p2, p3, p4, q1, q2, q3, q4, y1, y2, z1, z2

    c2 = c**2

    # Initialize the layer recursion vector
    G02 = pow(rho[0] * beta2[0], 2)
    X0 = G02 * (2 * (2 - c2 / beta2[0]))
    X1 = G02 * (-(2 - c2 / beta2[0]) ** 2)
    X2 = 0
    X3 = 0
    X4 = -4 * G02

    # Conduct the layer recursion
    for i in range(n):

        # Compute layer parameters
        epsilon = rho[i+1] / rho[i]
        eta = 2 / c2 * (beta2[i] - epsilon * beta2[i+1])
        a = epsilon + eta
        ak = a - 1
        b = 1 - eta
        bk = b - 1

        # Compute layer eigenfunctions
        r = sqrt(1 - c2 / alpha2[i])
        krh = (k * h[i]) * r
        C_alpha = cosh(krh)
        S_alpha = sinh(krh)

        s = sqrt(1 - c2 / beta2[i])
        ksh = (k * h[i]) * s
        C_beta = cosh(ksh)
        S_beta = sinh(ksh)

        # Update the elements of the layer recursion vector
        p1 = C_beta * X1 + s * S_beta * X2
        p2 = C_beta * X3 + s * S_beta * X4
        p3 = (1 / s) * S_beta * X1 + C_beta * X2
        p4 = (1 / s) * S_beta * X3 + C_beta * X4

        q1 = C_alpha * p1 - r * S_alpha * p2
        q2 = -(1 / r) * S_alpha * p3 + C_alpha * p4
        q3 = C_alpha * p3 - r * S_alpha * p4
        q4 = -(1 / r) * S_alpha * p1 + C_alpha * p2

        y1 = ak * X0 + a * q1
        y2 = a * X0 + ak * q2

        z1 = b * X0 + bk * q1
        z2 = bk * X0 + b * q2

        X0 = bk * y1 + b * y2
        X1 = a * y1 + ak * y2
        X2 = epsilon * q3
        X3 = epsilon * q4
        X4 = bk * z1 + b * z2

    r = sqrt(1 - c2 / alpha2[n])
    s = sqrt(1 - c2 / beta2[n])

    # Compute the value of the dispersion function and return its sign
    return csgn(X1 + s * X2 - r * (X3 + s * X4))