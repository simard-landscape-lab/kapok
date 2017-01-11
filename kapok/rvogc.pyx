# -*- coding: utf-8 -*-
# cython: language_level=3
"""Random Volume Over Ground (RVoG) Forest Model Inversion

    Contains functions for the forward RVoG model, and inversion.  Functions
    written in Cython for increased speed.  Imported by main rvog module.
    
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

cimport numpy as np
cimport cython
np.import_array()
np.import_ufunc()



def rvogfwdvol(hv, ext, inc, kz):
    """RVoG forward model volume coherence.
    
    For a given set of model parameters, calculate the RVoG model coherence.
    
    Arguments:
        hv: Height of the forest volume, in meters.
        ext: Wave extinction within the forest volume, in Np/m.
        inc: Incidence angle, in radians.
        kz: Interferometric vertical wavenumber, in radians/meter.
            
    Returns:
        gamma: Modelled complex coherence.

    """
    p1 = 2*ext*np.cos(inc)
    p2 = p1 + 1j*kz
    
    gammav = (p1 / p2) * (np.exp(p2*hv)-1) / (np.exp(p1*hv)-1)
    
    # Check if scalar or array hv.
    if isinstance(hv, (collections.Sequence, np.ndarray)):
        hvscalar = False
    else:
        hvscalar = True
    
    # Check for Zero Forest Height
    if hvscalar and (hv <= 0):
        gammav[:] = 1.0
    else:
        ind = (hv <= 0)
        if np.any(ind):
            gammav[ind] = 1.0
    
    # Check for Non-Finite Volume Coherence (very large p1--extinction essentially infinite)    
    ind = ~np.isfinite(gammav)
    if np.any(ind):
        if hvscalar:
            gammav[ind] = np.exp(1j*hv*kz[ind])
        else:
            gammav[ind] = np.exp(1j*hv[ind]*kz[ind])
    
    return gammav
    
    
def rvoginv(gamma, phi, inc, kz, ext=None, tdf=None, mu=0, mask=None,
            limit2pi=True, hv_min=0, hv_max=50, hv_step=0.01, ext_min=0.0115,
            ext_max=0.115):
    """RVoG model inversion.
    
        Calculate the RVoG model parameters which produce a modelled coherence
        closest to a set of observed coherences.  The model is formulated
        using real-valued volumetric temporal decorrelation factor (tdf), with
        physical parameters representing forest height (hv), extinction of
        the radar waves within the forest canopy (ext), and the coherence of
        the ground surface (phi), where arg(phi) is equal to the topographic
        phase.  In addition, the ground-to-volume amplitude ratio (mu) varies
        as a function of the polarization.
        
        In the single-baseline case, in order to reduce the number of unknowns
        and ensure the model has a unique solution, we assume that mu for the
        high coherence (from the phase diversity coherence optimization) is
        fixed.  By default it is set to zero, that is, we assume the high
        coherence has no ground scattering component.  We must then fix either
        the extinction value, or the temporal decorrelation.
        
        This function therefore requires either the ext or td keyword arguments
        to be provided.  The function will then optimize whichever of those two
        parameters is not provided, plus the forest height.  If neither
        parameter is provided, tdf will be fixed to a value of 1.0 (no temporal
        decorrelation).
        
        Note that the ext, tdf, and mu keyword arguments can be provided as
        a fixed single value (e.g., mu=0), as an array with the same
        dimensions as gamma, or as a LUT of parameter values as a function
        of the forest height parameter.  In this case, a dict should be given
        where dict['x'] contains the forest height values for each LUT bin,
        and dict['y'] contains the parameter values.  This LUT will then be
        interpolated using numpy.interp to the forest height values by the
        function.
        
        Note that one cannot fix both ext and tdf using this function.  The
        function will always try to solve for one of these two parameters.
        
        Arguments:
            gamma (array): 2D complex-valued array containing the 'high'
                coherences from the coherence optimization.
            phi (array): 2D complex-valued array containing the ground
                coherences (e.g., from kapok.topo.groundsolver()).
            inc (array): 2D array containing the master track incidence
                angle, in radians.
            kz (array): 2D array containing the kz values, in radians/meter.
            ext: Fixed values for the extinction parameter, in Nepers/meter.
                If not specified, function will try to optimize the values of
                ext and hv for fixed tdf.  Default: None.
            tdf: Fixed values for the temporal decorrelation factor, from 0
                to 1.  If not specified, the function will try to optimize
                the values of tdf and hv.  If both ext and tdf are left empty,
                function will fix tdf to 1.  Default: None.
            mu: Fixed values for the ground-to-volume scattering ratio of
                gamma.  Default: 0.
            mask (array): Boolean array.  Pixels where (mask == True) will be
                inverted, while pixels where (mask == False) will be ignored,
                and hv set to -1.
            limit2pi (bool): If True, function will not allow hv to go above
                the 2*pi (ambiguity) height (as determined by the kz values).
                If False, no such restriction.  Default: True.
            hv_min (float): Minimum allowed hv value, in meters.
                Default: 0.
            hv_max (float): Maximum allowed hv value, in meters.
                Default: 50.
            hv_step (float): Function will perform consecutive searches with
                progressively smaller step sizes, until the step size
                reaches a value below hv_step.  Default: 0.01 m.
            ext_min (float): Minimum extinction value, in Np/m.
                Default: 0.0115 Np/m (~0.1 dB/m).
            ext_max (float): Maximum extinction value, in Np/m.
                Default: 0.115 Np/m (~1 dB/m).
            
        Returns:
            hvmap (array): Array of inverted forest height values, in meters.
            extmap/tdfmap (array): If ext was specified, array of inverted tdf
                values will be returned here.  If tdf was specified, array
                of inverted ext values will be returned.
            converged (array): A 2D boolean array.  For each pixel, if
                |observed gamma - modelled gamma| <= 0.01, that pixel is
                marked as converged.  Otherwise, converged will be False for
                that pixel.  Pixels where converged == False suggest that the
                RVoG model could not find a good fit for that pixel, and the
                parameter estimates may be invalid.
    
    """
    print('kapok.rvog.rvoginv | Beginning RVoG model inversion. ('+time.ctime()+')')
    dim = np.shape(gamma)
    
    if mask is None:
        mask = np.ones(dim, dtype='bool')
        
    if np.all(limit2pi) or (limit2pi is None):
        limit2pi = np.ones(dim, dtype='bool')
    elif np.all(limit2pi == False):
        limit2pi = np.zeros(dim, dtype='bool')

       
    hv_samples = int((hv_max-hv_min+1)*3) # Initial Number of hv Bins in Search Grid
    hv_vector = np.linspace(hv_min, hv_max, num=hv_samples)
    
    if tdf is not None:
        ext_samples = 60
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    elif ext is None:
        tdf = 1.0
        ext_samples = 60
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    else:
        ext_vector = [-1]
 
        
    # Use mask to clip input data.
    gammaclip = gamma[mask]
    phiclip = phi[mask]
    incclip = inc[mask]
    kzclip = kz[mask]
    limit2piclip = limit2pi[mask]
    
    if isinstance(mu, (collections.Sequence, np.ndarray)):
        muclip = mu[mask]
    elif isinstance(mu, dict):
        print('kapok.rvog.rvoginv | Using LUT for mu as a function of forest height.')
        muclip = None
    else:
        muclip = np.ones(gammaclip.shape, dtype='float32') * mu
           
    if isinstance(ext, (collections.Sequence, np.ndarray)):
        extclip = ext[mask]
    elif isinstance(ext, dict):
        print('kapok.rvog.rvoginv | Using LUT for extinction as a function of forest height.')
        extclip = None
    elif ext is not None:
        extclip = np.ones(gammaclip.shape, dtype='float32') * ext
    elif isinstance(tdf, (collections.Sequence, np.ndarray)):
        tdfclip = tdf[mask]
    elif isinstance(tdf, dict):
        print('kapok.rvog.rvoginv | Using LUT for temporal decorrelation magnitude as a function of forest height.')
        tdfclip = None
    elif tdf is not None:
        tdfclip = np.ones(gammaclip.shape, dtype='float32') * tdf
        

    # Arrays to store the fitted parameters:
    hvfit = np.zeros(gammaclip.shape, dtype='float32')
    
    if ext is None:
        extfit = np.zeros(gammaclip.shape, dtype='float32')
        print('kapok.rvog.rvoginv | Solving for forest height and extinction, with fixed temporal decorrelation.')
    else:
        tdffit = np.zeros(gammaclip.shape, dtype='float32')
        print('kapok.rvog.rvoginv | Solving for forest height and temporal decorrelation magnitude, with fixed extinction.')
    
    
    # Variables for optimization:
    mindist = np.ones(gammaclip.shape, dtype='float32') * 1e9
    convergedclip = np.ones(gammaclip.shape,dtype='bool')
    threshold = 0.01 # threshold for convergence
    
    print('kapok.rvog.rvoginv | Performing repeated searches over smaller parameter ranges until hv step size is less than '+str(hv_step)+' m.')
    print('kapok.rvog.rvoginv | Beginning pass #1 with hv step size: '+str(np.round(hv_vector[1]-hv_vector[0],decimals=3))+' m. ('+time.ctime()+')')
    

    for n, hv_val in enumerate(hv_vector):
        print('kapok.rvog.rvoginv | Progress: '+str(np.round(n/hv_vector.shape[0]*100,decimals=2))+'%. ('+time.ctime()+')     ', end='\r')
        for ext_val in ext_vector:
            if isinstance(mu, dict):
                muclip = np.interp(hv_val, mu['x'], mu['y'])
                
            if ext is None:
                if isinstance(tdf, dict):
                    tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])
                    
                gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip)
                gamma_model = phiclip * (muclip + tdfclip*gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)
            else:
                if isinstance(ext, dict):
                    extclip = np.interp(hv_val, ext['x'], ext['y'])
                    
                gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip)
                tdf_val = np.abs((gammaclip*(muclip+1) - phiclip*muclip)/(phiclip*gammav_model))
                gamma_model = phiclip * (muclip + tdf_val*gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)

            # If potential vegetation height is greater than
            # 2*pi ambiguity height, and the limit2pi option
            # is set to True, remove these as potential solutions:               
            ind_limit = limit2piclip & (hv_val > np.abs(2*np.pi/kzclip))
            if np.any(ind_limit):
                dist[ind_limit] = 1e10

            # Best solution so far?                 
            ind = np.less(dist,mindist)
            
            # Then update:
            if np.any(ind):
                mindist[ind] = dist[ind]
                hvfit[ind] = hv_val
                if ext is None:
                    extfit[ind] = ext_val
                else:
                    tdffit[ind] = tdf_val[ind]
                    
                    
                    
    hv_inc = hv_vector[1] - hv_vector[0]
    if ext is None:
        ext_inc = ext_vector[1] - ext_vector[0]
    else:
        ext_inc = 1e-10

    
    itnum = 1
    while (hv_inc > hv_step):
        itnum += 1
        hv_low = hvfit - hv_inc
        hv_high = hvfit + hv_inc
        hv_val = hv_low.copy()
        hv_inc /= 10
               
        if ext is None:
            ext_low = extfit - ext_inc
            ext_high = extfit + ext_inc
            ext_val = ext_low.copy()
            ext_inc /= 10
        else:
            ext_low = np.array(ext_min)
            ext_high = np.array(ext_max)
            ext_val = ext_low.copy()
            ext_inc = 1e10
        
        print('kapok.rvog.rvoginv | Beginning pass #'+str(itnum)+' with hv step size: '+str(np.round(hv_inc,decimals=3))+' m. ('+time.ctime()+')')
        while np.all(hv_val < hv_high):
            print('kapok.rvog.rvoginv | Progress: '+str(np.round((hv_val-hv_low)/(hv_high-hv_low)*100,decimals=2)[0])+'%. ('+time.ctime()+')     ', end='\r')
            
            while np.all(ext_val < ext_high):
                if isinstance(mu, dict):
                    muclip = np.interp(hv_val, mu['x'], mu['y'])
                
                if ext is None:
                    if isinstance(tdf, dict):
                        tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])
                        
                    gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip)
                    gamma_model = phiclip * (muclip + tdfclip*gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
                else:
                    if isinstance(ext, dict):
                        extclip = np.interp(hv_val, ext['x'], ext['y'])
                        
                    gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip)
                    tdf_val = np.abs((gammaclip*(muclip+1) - phiclip*muclip)/(phiclip*gammav_model))
                    gamma_model = phiclip * (muclip + tdf_val*gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
    
                # If potential vegetation height is greater than
                # 2*pi ambiguity height, and the limit2pi option
                # is set to True, remove these as potential solutions:               
                ind_limit = limit2piclip & (hv_val > np.abs(2*np.pi/kzclip))
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10
    
                # Best solution so far? 
                ind = np.less(dist,mindist)
                
                # Then update:
                if np.any(ind):
                    mindist[ind] = dist[ind]
                    hvfit[ind] = hv_val[ind]
                    if ext is None:
                        extfit[ind] = ext_val[ind]
                    else:
                        tdffit[ind] = tdf_val[ind]
                        

                # Increment the extinction:        
                ext_val += ext_inc

            # Increment the forest height:                
            hv_val += hv_inc
            ext_val = ext_low.copy()

    
    # Check convergence rate.
    ind = np.less(mindist,threshold)
    convergedclip[ind] = True
    num_converged = np.sum(convergedclip)
    num_total = len(convergedclip)
    rate = np.round(num_converged/num_total*100,decimals=2)
    
    print('kapok.rvog.rvoginv | Completed.  Convergence Rate: '+str(rate)+'%. ('+time.ctime()+')')
    
    # Rebuild masked arrays into original image size.
    hvmap = np.ones(dim, dtype='float32') * -1
    hvmap[mask] = hvfit
    
    converged = np.ones(dim, dtype='float32') * -1
    converged[mask] = convergedclip
    
    if ext is None:
        extmap = np.ones(dim, dtype='float32') * -1
        extmap[mask] = extfit
        return hvmap, extmap, converged
    else:
        tdfmap = np.ones(dim, dtype='float32') * -1
        tdfmap[mask] = tdffit
        return hvmap, tdfmap, converged