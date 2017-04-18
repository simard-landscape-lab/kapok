# -*- coding: utf-8 -*-
"""Topography Estimation Module

    Estimating ground coherence and ground topographic phase using line
    fitting of the observed PolInSAR coherences.  For reference, see the
    paper:
    
    S. R. Cloude and K. P. Papathanassiou, "Three-stage inversion process
    for polarimetric SAR interferometry," IEE Proceedings - Radar, Sonar
    and Navigation, vol. 150, no. 3, pp. 125-134, 2 June 2003.
    doi: 10.1049/ip-rsn:20030449
    
    
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
import collections
import time

import numpy as np


def groundsolver(gamma, kz=None, groundmag=None, gammavol=None,
                 returnall=False, silent=False):
    """Solve for the ground coherence using a line fit of the observed
        coherences.
    
        Arguments:
            gamma (array): Array containing the complex coherences to line
                fit.  Should have dimensions (2, azimuth, range).
                gamma[0,:,:] should contain one of the coherences to line fit,
                and gamma[1,:,:] should contain the other (e.g., from phase
                diversity coherence optimization).
            kz: The kz values.  If specified, the algorithm will choose
                between the two ground coherence solutions by assuming that
                the phase difference between the high coherence and the
                ground is less than pi.  If this behaviour is not
                desired, see the gammavol argument for another method for
                choosing between the ground coherences.  Note that only the
                sign of kz, not the magnitude, matters here.  The input can
                therefore either be the 2D array of kz values, or simply a
                scalar with the same sign as the actual kz.
            groundmag (array): Magnitude of the ground coherence.
            gammavol (array): Array containing coherences that are assumed
                to be closer to the volume-dominated coherence than the
                ground-dominated coherence.  If specified, the ground
                solver will always choose the ground solution that is
                farther (on the complex plane) from gammavol.  Note:
                Selection of the ground solution through the kz is
                recommended (see above), as it is generally more
                robust.
            returnall (bool): If set to True, function will return the
                ground solutions, as well as the other ground solution
                not chosen, and an array identifying which of the two
                input coherences in gamma are the volume-dominated
                coherence.  If False, only the chosen ground solution
                will be returned.
            silent (bool): If set to True, no output status will be
                printed.
            
        Returns:
            ground (array): Array of complex-valued ground coherences.
            groundalt (array): The other ground solutions which were not
                chosen.  Only returned if returnall == True.
            volindex (array): 2D array containing the index of the high/volume
                (smallest ground contribution) coherence for each pixel.
                If gamma[0,azimuth,range] is closer to the vol coherence
                than gamma[1,azimuth,range], volindex[azimuth,range] will
                equal 0.  Only returned if returnall == True.
    
    """
    if not silent:
        print('kapok.topo.groundsolver | Solving for ground coherence. ('+time.ctime()+')')
    # Get the two possible ground coherence solutions.
    solutions = linefit(gamma, groundmag)
    
    if kz is not None:
        if not isinstance(kz, (collections.Sequence, np.ndarray)):
            kz = np.ones((gamma.shape[1],gamma.shape[2]),dtype='float32') * kz
            
        # Get the volume-dominated coherences corresponding to each ground solution. (Observed coherence farthest from ground.)
        gammav = gamma.copy()
        gammav[0] = np.where(np.abs(solutions[0] - gamma[0]) > np.abs(solutions[0] - gamma[1]), gamma[0], gamma[1])
        gammav[1] = np.where(np.abs(solutions[1] - gamma[0]) > np.abs(solutions[1] - gamma[1]), gamma[0], gamma[1])
        
        # Angular separation between volume coherence and ground -- is it same sign as kz?
        sep = np.angle(gammav*np.conj(solutions))*np.sign(kz)
        
        ground = np.where(sep[0] >= 0, solutions[0], solutions[1])
        groundalt = np.where(sep[0] >= 0, solutions[1], solutions[0])
        volindex = (np.abs(gamma[1] - ground) > np.abs(gamma[0] - ground))
    elif gammavol is not None:
        # Of the two observed coherences, assume the volume-dominated coherence is the one
        # which is closer to the input gammavol array.
        volindex = (np.abs(gamma[1] - gammavol) < np.abs(gamma[0] - gammavol))
        gammav = np.where(volindex, gamma[1], gamma[0])
        
        # Choose the ground that is farther from gammav.
        ground = np.where(np.abs(gammav - solutions[0]) > np.abs(gammav - solutions[1]),solutions[0],solutions[1])
        groundalt = np.where(np.abs(gammav - solutions[0]) > np.abs(gammav - solutions[1]),solutions[1],solutions[0])
    else:
        print('kapok.topo.groundsolver | Neither kz or estimated volume coherence specified.  Unable to choose between ambiguous ground solutions.  Aborting.')
        ground = None
        groundalt = None
        volindex = None
    
    if not silent:
        print('kapok.topo.groundsolver | Complete. ('+time.ctime()+')')
        
    if returnall:
        return ground, groundalt, volindex
    else:
        return ground


def linefit(gamma, groundmag=None):
    """Fit a line through two observed complex coherences and return
        the two possible ground coherence solutions.
        
        Arguments:
            gamma (array):  Array with dimensions (2, azimuth, range)
                containing the observed coherences to fit a line
                through.
            groundmag (array): The ground coherence magnitude.  If not
                specified, the function will assume the ground
                coherence magnitude is equal to one.  If specified,
                this function will find the intersections between
                the fitted line and a circle with radius equal to
                gammag.  If these intersections are within the
                observed coherences, the solutions will be moved
                along the line until they are on top of the closest
                observed coherence.  There will always be one
                solution on either side of the observed coherence
                region.  If the value of gammag is such that
                both intersections are on one side of the observed
                coherences, then the coherence farther from the
                observed coherences will be chosen, and the other
                ground solution will be equal to the observed
                coherence farthest from the first ground solution.
                In general, this should be a rare situation to
                occur, unless gammag is set to an unreasonably low
                value.
                
        Returns:
            solutions (array): Array with the same dimensions as
                gamma containing the two ground complex coherence
                solutions for each pixel.
        
    """
    if groundmag is None:
        groundmag = np.ones((gamma.shape[1],gamma.shape[2]),dtype='float32')
    elif not isinstance(groundmag, (collections.Sequence, np.ndarray)):
        groundmag = np.ones((gamma.shape[1],gamma.shape[2]),dtype='float32') * groundmag
        
    groundmag[groundmag > 1] = 1.0
    
    solutions = np.zeros(gamma.shape,dtype='complex64')        
        
    # Intersections between line through gamma and circle with radius groundmag:
    a = np.square(np.abs(gamma[0] - gamma[1]))
    b = 2*np.real(gamma[0]*np.conj(gamma[1])) - 2*np.square(np.abs(gamma[1]))
    c = np.square(np.abs(gamma[1])) - np.square(np.abs(groundmag))
    
    xa = (-1*b - np.sqrt(np.square(b) - 4*a*c))/(2*a)
    xb = (-1*b + np.sqrt(np.square(b) - 4*a*c))/(2*a)
    
    solutions[0] = xa*gamma[0] + (1-xa)*gamma[1]
    solutions[1] = xb*gamma[0] + (1-xb)*gamma[1]
    
    # Is the coherence magnitude given by groundmag lower than both observed
    # coherences? (e.g., no valid intersection)
    ind = (groundmag < np.abs(gamma[0])) & (groundmag < np.abs(gamma[1]))
    if np.any(ind):
        solutions[0][ind] = np.nan
        solutions[1][ind] = np.nan
         
    # Are any of the solutions within the observed coherence region?       
    ind = np.sign(np.angle(solutions*np.conj(gamma[1]))) == np.sign(np.angle(gamma[0]*np.conj(solutions)))
    if np.any(ind):
        solutions[ind] = np.nan

        
    # Both solutions invalid:
    ind = np.isnan(solutions[0]) & np.isnan(solutions[1])
    if np.any(ind):
        solutions[0][ind] = gamma[0][ind]
        solutions[1][ind] = gamma[1][ind]
    
    # First solution invalid.
    ind = np.isnan(solutions[0]) & np.isfinite(solutions[1])
    if np.any(ind):
        gammareplace = np.where(np.abs(solutions[1]-gamma[0]) > np.abs(solutions[1]-gamma[1]),gamma[0],gamma[1])
        solutions[0][ind] = gammareplace[ind]
        
    # Other solution invalid.
    ind = np.isnan(solutions[1]) & np.isfinite(solutions[0])
    if np.any(ind):
        gammareplace = np.where(np.abs(solutions[0]-gamma[0]) > np.abs(solutions[0]-gamma[1]),gamma[0],gamma[1])
        solutions[1][ind] = gammareplace[ind]
    
    return solutions