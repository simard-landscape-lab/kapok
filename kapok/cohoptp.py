# -*- coding: utf-8 -*-
# cython: language_level=3
"""Coherence Optimization Native Python Functions.

    Currently contains an implementation of the phase diversity coherence
    optimization algorithm which finds the two coherences with the largest
    separation in the complex plane.  This is Python code which the cohopt.py
    wrapper module defaults to when the Cython import fails.

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
import time

import numpy as np
import numpy.linalg as linalg



def pdopt(tm, om, numph=30, step=50, reg=0.0, returnall=False):
    """Phase diversity coherence optimization.
    
    Solves an eigenvalue problem in order to find the complex coherences with
    maximum separation (|a - b|) in the complex plane.  Of these two
    coherences, one should in theory represent the coherence with the
    smallest ground contribution present in the data (the 'high' coherence).
    The other then represents the coherence with the largest ground
    contribution present in the data (the 'low' coherence).
    
    Arguments:
        tm (array): The polarimetric covariance (T) matrix of the data,
            with dimensions: [az, rng, num_pol, num_pol].  Note that in the
            HDF5 file, covariance matrix elements below the diagonal are
            zero-valued, in order to save disk space.  The (j,i) elements
            should therefore calculated from the complex conjugate of the
            (i,j) elements using the kapok.lib.makehermitian() function before
            the matrix is passed to this function.  Note: This should be the
            average matrix of the two tracks forming the baseline, assuming
            polarimetric stationarity.
        om (array): The polarimetric interferometric (Omega) matrix of the
            data, with dimensions [az, rng, num_pol, num_pol].
        numph (int): The number of phase shifts to calculate coherences for.
            The higher the number, the smaller the spacing of the coherences
            around the coherence region perimeter.  The smaller the number,
            the faster the computation time.  Default: 30.
        step (int): Block size (in pixels) used for linalg.eig.  Higher values
            will use more memory but can run a little faster.
            Default: 50.
        reg (float): Regularization factor.  The tm matrix is added to
            the matrix reg*Tr(tm)*I, where Tr(tm) is the trace of tm, and I
            is the identity matrix.  Similarly, the omega matrix is added to
            the matrix reg*Tr(om)*I.  This regularization reduces the spread
            of the coherence region for pixels where the backscatter is
            highly polarization dependent.
        returnall (bool): True/False flag.  Set to true to return the
            weight vectors for the optimized coherences, as well as the
            pair of minor axis coherences (optimized coherence pair with
            minimum separation in the complex plane).  Default: False.
          
    Returns:
        gammamax (array): The optimized coherence with the max eigenvalue.
        gammamin (array): The optimized coherence with the min eigenvalue.
        gammaminormax (array): Of the coherences with the minimum separation
            in the complex plane (e.g., along the minor axis of a elliptical
            coherence region), this will be the one with the max eigenvalue.
            Only returned if returnall == True.
        gammaminormin (array): Of the coherences with the maximum separation
            in the complex plane (e.g., along the minor axis of a elliptical
            coherence region), this will be the one with the min eigenvalue.
            Only returned if returnall == True.
        wmax (array): The weight vector for the max eigenvalue coherence, if
            returnall == True.
        wmin (array): The weight vector for the min eigenvalue coherence, if
            returnall == True.
    
    """
    dim = np.shape(tm)    
    
    # Matrix regularization: 
    if reg > 0:
        regmat = np.zeros(dim, dtype='complex64')
        regmat[:,:] = np.eye(dim[2])
        regmat = regmat * reg * np.trace(tm, axis1=2, axis2=3)[:,:,np.newaxis,np.newaxis]
        tm = tm + regmat
        
        regmat = np.zeros(dim, dtype='complex64')
        regmat[:,:] = np.eye(dim[2])
        regmat = regmat * reg * np.trace(om, axis1=2, axis2=3)[:,:,np.newaxis,np.newaxis]
        om = om + regmat
        del regmat
    
    
    # Arrays to store coherence separation, and the two complex coherence values.
    cohsize = (dim[0],dim[1]) # number of az, rng pixels
    cohdiff = np.zeros(cohsize,dtype='float32')
    gammamax = np.zeros(cohsize,dtype='complex64')
    gammamin = np.zeros(cohsize,dtype='complex64')
    
    # Arrays to store minor axis coherences.
    mincohdiff = np.ones(cohsize,dtype='float32') * 99
    gammaminormax = np.zeros(cohsize,dtype='complex64')
    gammaminormin = np.zeros(cohsize,dtype='complex64')
    
    # Arrays to store polarimetric weighting vectors for the optimized coherences.
    weightsize = (dim[0],dim[1],dim[3])
    wmax = np.zeros(weightsize,dtype='complex64')
    wmin = np.zeros(weightsize,dtype='complex64')

    # Main Loop
    for Ph in np.arange(0,numph): # loop through rotation angles
        Pr = Ph * np.pi / numph # phase shift to be applied
        
        print('kapok.cohopt.pdopt | Current Progress: '+str(np.round(Pr/np.pi*100,decimals=2))+'%. ('+time.ctime()+')     ', end='\r')
        
        for az in range(0,dim[0],step):
            azend = az + step
            if azend > dim[0]:
                azend = dim[0]
            
            for rng in range(0,dim[1],step):
                rngend = rng + step
                if rngend > dim[1]:
                    rngend = dim[1]
                
                omblock = om[az:azend,rng:rngend]
                tmblock = tm[az:azend,rng:rngend]
                z12 = omblock.copy()
                
                # Apply phase shift to omega matrix:
                z12 = z12*np.exp(1j*Pr)
                z12 = 0.5 * (z12 + np.rollaxis(np.conj(z12),3,start=2))
                
                # Check if any pixels have singular covariance matrices.
                # If so, set those matrices to the identity, to keep an
                # exception from being thrown by linalg.inv().
                det = linalg.det(tmblock)
                ind = (det == 0)
                if np.any(ind):
                    tmblock[ind] = np.eye(dim[3])

                
                # Solve the eigenvalue problem:
                nu, w = linalg.eig(np.einsum('...ij,...jk->...ik', linalg.inv(tmblock), z12))
                
                wH = np.rollaxis(np.conj(w),3,start=2)
                
                Tmp = np.einsum('...ij,...jk->...ik', omblock, w)
                Tmp12 = np.einsum('...ij,...jk->...ik', wH, Tmp)
                
                Tmp = np.einsum('...ij,...jk->...ik', tmblock, w)
                Tmp11 = np.einsum('...ij,...jk->...ik', wH, Tmp)
                
                azind = np.tile(np.arange(0,w.shape[0]),(w.shape[1],1)).T
                rngind = np.tile(np.arange(0,w.shape[1]),(w.shape[0],1))
                
                lmin = np.argmin(nu,axis=2)
                gmin = Tmp12[azind,rngind,lmin,lmin] / np.abs(Tmp11[azind,rngind,lmin,lmin])
                
                lmax = np.argmax(nu,axis=2)
                gmax = Tmp12[azind,rngind,lmax,lmax] / np.abs(Tmp11[azind,rngind,lmax,lmax])
                
                ind = (np.abs(gmax-gmin) > cohdiff[az:azend,rng:rngend])
                
                
                # If we've found the coherences with the best separation
                # so far, save them.
                if np.any(ind):
                    (azupdate, rngupdate) = np.where(ind)
                    
                    cohdiff[az+azupdate,rng+rngupdate] = np.abs(gmax-gmin)[azupdate,rngupdate]
                    gammamax[az+azupdate,rng+rngupdate] = gmax[azupdate,rngupdate]
                    gammamin[az+azupdate,rng+rngupdate] = gmin[azupdate,rngupdate]
                    
                    if returnall:
                        wmax[az+azupdate,rng+rngupdate,:] = np.squeeze(w[azupdate,rngupdate,:,lmax[azupdate,rngupdate]])
                        wmin[az+azupdate,rng+rngupdate,:] = np.squeeze(w[azupdate,rngupdate,:,lmin[azupdate,rngupdate]])
                
                
                # If returnall is True, also check if this coherence pair
                # has the smallest separation found so far.
                if returnall:
                    ind = (np.abs(gmax-gmin) < mincohdiff[az:azend,rng:rngend])
                    
                    if np.any(ind):
                        (azupdate, rngupdate) = np.where(ind)
                        
                        mincohdiff[az+azupdate,rng+rngupdate] = np.abs(gmax-gmin)[azupdate,rngupdate]
                        gammaminormax[az+azupdate,rng+rngupdate] = gmax[azupdate,rngupdate]
                        gammaminormin[az+azupdate,rng+rngupdate] = gmin[azupdate,rngupdate]
    
    
    print('kapok.cohopt.pdopt | Optimization complete. ('+time.ctime()+')          ')
    if returnall:
        return gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin
    else:
        return gammamax, gammamin


def pdopt_pixel(tm, om, numph=60, reg=0.0):
    """Phase diversity coherence optimization for a single pixel.
    
    Same functionality as the pdopt function above, but for a single pixel
    only.  This is the function called when plotting a coherence region.
    
    Arguments:
        tm (array): The polarimetric covariance (T) matrix of the data,
            with dimensions: [num_pol, num_pol].  Note that in the
            HDF5 file, covariance matrix elements below the diagonal are
            zero-valued, in order to save disk space.  The (j,i) elements
            should therefore calculated from the complex conjugate of the
            (i,j) elements using the kapok.lib.makehermitian() function before
            the matrix is passed to this function.  Note: This should be the
            average matrix of the two tracks forming the baseline, assuming
            polarimetric stationarity.
        om (array): The polarimetric interferometric (Omega) matrix of the
            data, with dimensions [num_pol, num_pol].
        numph (int): The number of phase shifts to calculate coherences for.
            The higher the number, the smaller the spacing of the coherences
            around the coherence region perimeter.  The smaller the number,
            the faster the computation time.  Default: 30.
        reg (float): Regularization factor.  The tm matrix is added to
            the matrix reg*Tr(tm)*I, where Tr(tm) is the trace of tm, and I
            is the identity matrix.  Similarly, the omega matrix is added to
            the matrix reg*Tr(om)*I.  This regularization reduces the spread
            of the coherence region for pixels where the backscatter is
            highly polarization dependent.
    
    Returns:
        gammamax (complex): the optimized coherence with the max eigenvalue.
        gammamin (complex): the optimized coherence with the min eigenvalue.
        gammaregion (array): Every coherence from the solved eigenvalue
            problems.  These coherences will lie around the edge of the
            coherence region.
    
    """   
    cohdiff = 0
    gammaregion = np.empty((numph*2 + 1),dtype='complex')
    
    # Matrix regularization: 
    if reg > 0:
        tm = tm + reg*np.trace(tm)*np.eye(3)
        om = om + reg*np.trace(om)*np.eye(3)


    for Ph in range(0,numph): # loop through rotation angles
        Pr = Ph * np.pi / numph # phase shift to be applied
        
        
        # Apply phase shift to omega matrix:
        z12 = om.copy()*np.exp(1j*Pr)
        z12 = 0.5 * (z12 + np.transpose(np.conj(z12)))
        

        # Solve the eigenvalue problem:                
        nu, w = linalg.eig(np.dot(linalg.inv(tm),z12))
                
        wH = np.transpose(np.conj(w))
        
        Tmp = np.dot(om,w)
        Tmp12 = np.dot(wH,Tmp)
        
        Tmp = np.dot(tm,w)
        Tmp11 = np.dot(wH,Tmp)
        
        l = np.argmin(nu)
        gmin = Tmp12[l,l] / np.abs(Tmp11[l,l]) # min eigenvalue coherence
        
        l = np.argmax(nu)
        gmax = Tmp12[l,l] / np.abs(Tmp11[l,l]) # max eigenvalue coherence
        
        gammaregion[Ph] = gmin
        gammaregion[Ph+numph] = gmax
        
        if (np.abs(gmax-gmin) > cohdiff):
            cohdiff = np.abs(gmax-gmin)
            gammamax = gmax
            gammamin = gmin

    gammaregion[-1] = gammaregion[0] # copy the first coherence to the end of the array, for a continuous coherence region plot
    
    return gammamax, gammamin, gammaregion