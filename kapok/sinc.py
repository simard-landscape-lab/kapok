# -*- coding: utf-8 -*-
"""Sinc Forest Model Module

    Forest height estimation using sinc coherence model and combined sinc
    coherence and phase difference model.  This is a basic model which
    relates coherence magnitude and phase to forest height.
    
    For reference on the model, see:
    
    S. R. Cloude, "Polarization coherence tomography," Radio Science,
    41, RS4017, 2006.  doi:10.1029/2005RS003436.
    
    
    Author: Michael Denbina
    
    Copyright 2016 California Institute of Technology.  All rights reserved.
    United States Government Sponsorship acknowledged.

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import numpy as np


def sincinv(gamma, kz, tdf=None, mask=None):
    """Sinc Forest Model Inversion
    
    Function to estimate tree heights using the inverse sinc function of the
    estimated volume coherence (e.g., HV, or gamma high from phase diversity 
    coherence optimization).
    
    The model equation is hv = (2*sincinv(|gammav|))/(kz), where hv is
    volume/forest height, gammav is the volume coherence, and kz is the
    interferometric vertical wavenumber.
    
    We assume that the inverse sinc function values for low coherence are
    within the central peak of the sinc function.
    
    The input arguments can be for a single pixel or the entire image,
    provided they either have the same dimensions or one of them is a scalar.
    
    Arguments:
        gamma (array): The estimated volumetric coherence.
        kz (array): The vertical wavenumber, in radians/meter.
        tdf (array): The estimated real-valued temporal decorrelation factor,
            if needed. The input gammav value will be corrected by assuming
            that the observed coherence is equal to the true volumetric
            coherence times tdf.  Default: No temporal decorrelation.
    
    Returns:
        hv (array): The estimated volume/forest height, in meters.

    """
    # If no temporal decorrelation factor input, tdf = 1:
    if tdf is None:
        tdf = np.ones(gamma.shape)

    # Build a LUT of sinc function values for the inversion.
    # Get a vector of values (used as input to the np.sinc() function)
    # from 1 down to 0, because np.sinc() is the normalized sinc
    # function and we want to start at the first zero of the sinc function.
    # We go in decreasing order because in order for np.interp() to work
    # properly the LUT values must be increasing.
    LUTreturn = np.linspace(1,0,num=201)
    LUT = np.sinc(LUTreturn)
    
    gamma = np.abs(gamma/tdf)
    gamma[gamma > 1] = 1
    gamma[gamma < 0] = 0

        
    # Use np.interp to linearly interpolate the LUT to the input gammav value.
    # We multiply by pi because numpy uses the normalized sinc function,
    # sinc(x) = sin(pi*x)/(pi*x), and we need a phase from the inverse.
    hv = (2*np.pi*np.interp(gamma,LUT,LUTreturn)) / np.abs(kz)
    
    if mask is not None:
        hv[np.invert(mask)] = -1
        
    return hv

    
def sincfwd(hv, kz):
    """ sincfwd
    
    Sinc coherence amplitude forward model.  See sincinv() for more details.
    This function can be used, for example, to calculate the expected volume
    decorrelation for this model for a given set of forest heights.
    
    Arguments:
        hv (array): The volume/forest height, in meters.
        kz (array): The vertical wavenumber, in radians/meter.
        
    Returns:
        gammav (array): The estimated volume coherence magnitude.
    
    """
    gammav = np.sinc((hv*np.abs(kz))/(2*np.pi))
    
    return gammav

    

def sincphaseinv(gamma, phi, kz, epsilon=None, tdf=None, mask=None):
    """ sincphaseinv
    
    Vegetation height inversion using a combination of sinc coherence
    estimation and the height of the estimated volume phase center above
    the ground.
       
    We mask out pixels with low phase separation between gamma high and
    gamma low, and set the vegetation height for these pixels to zero.
    
    The parameter epsilon is used to weight the second coherence amplitude
    term of the inversion.  In the zero extinction case, epsilon = 0.5, while
    for the infinite extinction case, epsilon = 0.  The suggested
    value in the Cloude (2006) paper for moderate extinction is 0.4.
    
    Arguments:
        gamma (array): Array of the estimated volumetric coherence for each
            pixel.
        phi (array): Estimated ground phases (e.g., from the functions in the
            kapok.topo module).
        kz (array): The vertical wavenumbers, in radians/meter.
        epsilon (float): The weighting factor of the coherence amplitude
            sinc inversion term.  Default: 0.4.
        tdf (array): The estimated temporal decorrelation factor, if needed.
            The input gammav value will be corrected by assuming that the
            observed coherence is equal to the true volumetric coherence times
            tdf.  Default: No temporal decorrelation.
        
    Returns:
        hv (array): The estimated forest height, in meters.
    
    """
    # Default epsilon value.
    if epsilon is None:
        epsilon = 0.4
    
    # If temporal decorrelation factor is not provided, assume no temporal
    # decorrelation:
    if tdf is None:
        tdf = np.ones(gamma.shape)
   
    # Phase difference:
    hp = np.angle((gamma/tdf)*np.exp(-1j*phi))/kz
    hp[hp < 0] += 2*np.pi*np.abs(kz[hp < 0])
       
    # Add sinc inversion term:
    hv = hp + (epsilon*sincinv(gamma,kz,tdf))
    
    if mask is not None:
        hv[np.invert(mask)] = -1
          
    return hv