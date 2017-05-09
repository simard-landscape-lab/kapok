# -*- coding: utf-8 -*-
"""Random Volume Over Ground (RVoG) Forest Model Inversion

    Contains functions for the forward RVoG model, and inversion.  This is
    Python code which the rvog.py wrapper module falls back to when the Cython
    import fails.
    
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


def rvogfwdvol(hv, ext, inc, kz, rngslope=0.0):
    """RVoG forward model volume coherence.
    
    For a given set of model parameters, calculate the RVoG model coherence.
    
    Note that all input arguments must be arrays (even if they are one element
    arrays), so that they can be indexed to check for nan or infinite values
    (due to extreme extinction or forest height values).  All input arguments
    must have the same shape.
    
    Arguments:
        hv (array): Height of the forest volume, in meters.
        ext (array): Wave extinction within the forest volume, in Np/m.
        inc (array): Incidence angle, in radians.
        kz (array): Interferometric vertical wavenumber, in radians/meter.
        rngslope (array): Range-facing terrain slope angle, in radians.  If not
            specified, flat terrain is assumed.
            
    Returns:
        gamma: Modelled complex coherence.

    """
    # Calculate the propagation coefficients.
    p1 = 2*ext*np.cos(rngslope)/np.cos(inc-rngslope)    
    p2 = p1 + 1j*kz
    
    # Check for zero or close to zero hv (or kz) values (e.g., negligible
    # volume decorrelation).
    gammav = kz*hv
    ind_novolume = np.isclose(np.abs(gammav), 0)
       
    # Check for zero or close to zero extinction values (e.g., uniform
    # vertical structure function).
    gammav = p2 * (np.exp(p1*hv) - 1)
    ind_zeroext = np.isclose(np.abs(gammav), 0) & ~ind_novolume
    
    # Check for infinite numerator of the volume coherence equation (e.g.,
    # extremely high extinction value).
    gammav = p1 * (np.exp(p2*hv) - 1)
    ind_nonfinite = ~np.isfinite(gammav) & ~ind_novolume & ~ind_zeroext
    
    # The remaining indices are where the standard equation should be valid:
    ind = ~ind_zeroext & ~ind_novolume & ~ind_nonfinite
    
    if np.any(ind_novolume):
        gammav[ind_novolume] = 1
        
    if np.any(ind_zeroext):
        gammav[ind_zeroext] = ((np.exp(1j*kz*hv) - 1) / (1j*kz*hv))[ind_zeroext]
        
    if np.any(ind_nonfinite):
        gammav[ind_nonfinite] = np.exp(1j*hv*kz)[ind_nonfinite]
    
    if np.any(ind):
        gammav[ind] = ((p1 / p2) * (np.exp(p2*hv)-1) / (np.exp(p1*hv)-1))[ind]
    
    
    return gammav    


def rvoginv(gamma, phi, inc, kz, ext=None, tdf=None, mu=0.0, rngslope=0.0,
            mask=None, limit2pi=True, hv_min=0.0, hv_max=60.0, hv_step=0.01,
            ext_min=0.0, ext_max=0.115, silent=False):
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
                the gamma input argument.  Default: 0.
            rngslope (array): Terrain slope angle in the ground range
                direction, in radians.  Default: 0 (flat terrain).
            mask (array): Boolean array.  Pixels where (mask == True) will be
                inverted, while pixels where (mask == False) will be ignored,
                and hv set to -1.
            limit2pi (bool): If True, function will not allow hv to go above
                the 2*pi (ambiguity) height (as determined by the kz values).
                If False, no such restriction.  Default: True.
            hv_min (float or array): Minimum allowed hv value, in meters.
                Default: 0.
            hv_max (float or array): Maximum allowed hv value, in meters.
                Default: 50.
            hv_step (float): Function will perform consecutive searches with
                progressively smaller step sizes, until the step size
                reaches a value below hv_step.  Default: 0.01 m.
            ext_min (float): Minimum extinction value, in Np/m.
                Default: 0.00115 Np/m (~0.01 dB/m).
            ext_max (float): Maximum extinction value, in Np/m.
                Default: 0.115 Np/m (~1 dB/m).
            silent (bool): Set to True to suppress status updates.  Default:
                False.
            
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
    if not silent:
        print('kapok.rvog.rvoginv | Beginning RVoG model inversion. ('+time.ctime()+')')
    dim = np.shape(gamma)
    
    if mask is None:
        mask = np.ones(dim, dtype='bool')
        
    if np.all(limit2pi) or (limit2pi is None):
        limit2pi = np.ones(dim, dtype='bool')
    elif np.all(limit2pi == False):
        limit2pi = np.zeros(dim, dtype='bool')
        
    if isinstance(hv_max, (collections.Sequence, np.ndarray)):
        hv_max_clip = hv_max.copy()[mask]
        hv_max = np.nanmax(hv_max)
    else:
        hv_max_clip = None
        
    if isinstance(hv_min, (collections.Sequence, np.ndarray)):
        hv_min_clip = hv_min.copy()[mask]
        hv_min = np.nanmin(hv_min)
    else:
        hv_min_clip = None
    
    
    hv_samples = int((hv_max-hv_min)*2 + 1) # Initial Number of hv Bins in Search Grid
    hv_vector = np.linspace(hv_min, hv_max, num=hv_samples)
    
    
    if tdf is not None:
        ext_samples = 40
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    elif ext is None:
        tdf = 1.0
        ext_samples = 40
        ext_vector = np.linspace(ext_min, ext_max, num=ext_samples)
    else:
        ext_vector = [-1.0]
 
        
    # Use mask to clip input data.
    gammaclip = gamma[mask]
    phiclip = phi[mask]
    incclip = inc[mask]
    kzclip = kz[mask]
    limit2piclip = limit2pi[mask]
    
    if isinstance(mu, (collections.Sequence, np.ndarray)):
        muclip = mu[mask]
    elif isinstance(mu, dict):
        if not silent:        
            print('kapok.rvog.rvoginv | Using LUT for mu as a function of forest height.')
        muclip = None
    else:
        muclip = np.ones(gammaclip.shape, dtype='float32') * mu
        
    if isinstance(rngslope, (collections.Sequence, np.ndarray)):
        rngslopeclip = rngslope[mask]
    else:
        rngslopeclip = np.ones(gammaclip.shape, dtype='float32') * rngslope
    
    if isinstance(ext, (collections.Sequence, np.ndarray)):
        extclip = ext[mask]
    elif isinstance(ext, dict):
        if not silent:
            print('kapok.rvog.rvoginv | Using LUT for extinction as a function of forest height.')
        extclip = None
    elif ext is not None:
        extclip = np.ones(gammaclip.shape, dtype='float32') * ext
    elif isinstance(tdf, (collections.Sequence, np.ndarray)):
        tdfclip = tdf[mask]
    elif isinstance(tdf, dict):
        if not silent:
            print('kapok.rvog.rvoginv | Using LUT for temporal decorrelation magnitude as a function of forest height.')
        tdfclip = None
    elif tdf is not None:
        tdfclip = np.ones(gammaclip.shape, dtype='float32') * tdf
        

    # Arrays to store the fitted parameters:
    hvfit = np.zeros(gammaclip.shape, dtype='float32')
    
    if ext is None:
        extfit = np.zeros(gammaclip.shape, dtype='float32')
        if not silent:
            print('kapok.rvog.rvoginv | Solving for forest height and extinction, with fixed temporal decorrelation.')
    else:
        tdffit = np.zeros(gammaclip.shape, dtype='float32')
        if not silent:
            print('kapok.rvog.rvoginv | Solving for forest height and temporal decorrelation magnitude, with fixed extinction.')
    
    
    # Variables for optimization:
    mindist = np.ones(gammaclip.shape, dtype='float32') * 1e9
    convergedclip = np.ones(gammaclip.shape,dtype='bool')
    threshold = 0.01 # threshold for convergence
    
    if not silent:
        print('kapok.rvog.rvoginv | Performing repeated searches over smaller parameter ranges until hv step size is less than '+str(hv_step)+' m.')
        print('kapok.rvog.rvoginv | Beginning pass #1 with hv step size: '+str(np.round(hv_vector[1]-hv_vector[0],decimals=3))+' m. ('+time.ctime()+')')
    

    for n, hv_val in enumerate(hv_vector):
        if not silent:
            print('kapok.rvog.rvoginv | Progress: '+str(np.round(n/hv_vector.shape[0]*100,decimals=2))+'%. ('+time.ctime()+')     ', end='\r')
        for ext_val in ext_vector:
            if isinstance(mu, dict):
                muclip = np.interp(hv_val, mu['x'], mu['y'])
                
            if ext is None:
                if isinstance(tdf, dict):
                    tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])
                    
                gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslope=rngslopeclip)
                gamma_model = phiclip * (muclip + tdfclip*gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)
            else:
                if isinstance(ext, dict):
                    extclip = np.interp(hv_val, ext['x'], ext['y'])
                    
                gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslope=rngslopeclip)
                tdf_val = np.abs((gammaclip*(muclip+1) - phiclip*muclip)/(phiclip*gammav_model))
                gamma_model = phiclip * (muclip + tdf_val*gammav_model) / (muclip + 1)
                dist = np.abs(gammaclip - gamma_model)

            # If potential vegetation height is greater than
            # 2*pi ambiguity height, and the limit2pi option
            # is set to True, remove these as potential solutions:               
            ind_limit = limit2piclip & (hv_val > np.abs(2*np.pi/kzclip))
            if np.any(ind_limit):
                dist[ind_limit] = 1e10
            
            # If hv_min and hv_max were set to arrays,
            # ensure that solutions outside of the bounds are excluded.
            if hv_min_clip is not None:
                ind_limit = (hv_val < hv_min_clip)
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10
            
            if hv_max_clip is not None:
                ind_limit = (hv_val > hv_max_clip)
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10
            
            
            # Best solution so far?                 
            ind = dist < mindist
            
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
            ext_low[ext_low < ext_min] = ext_min
            ext_high = extfit + ext_inc
            ext_high[ext_high > ext_max] = ext_max
            ext_val = ext_low.copy()
            ext_inc /= 10
        else:
            ext_low = np.array(ext_min,dtype='float32')
            ext_high = np.array(ext_max,dtype='float32')
            ext_val = ext_low.copy()
            ext_inc = 10.0
        
        if not silent:
            print('kapok.rvog.rvoginv | Beginning pass #'+str(itnum)+' with hv step size: '+str(np.round(hv_inc,decimals=3))+' m. ('+time.ctime()+')')
        while np.all(hv_val < hv_high):
            if not silent:
                print('kapok.rvog.rvoginv | Progress: '+str(np.round((hv_val-hv_low)/(hv_high-hv_low)*100,decimals=2)[0])+'%. ('+time.ctime()+')     ', end='\r')
            
            while np.all(ext_val < ext_high):
                if isinstance(mu, dict):
                    muclip = np.interp(hv_val, mu['x'], mu['y'])
                
                if ext is None:
                    if isinstance(tdf, dict):
                        tdfclip = np.interp(hv_val, tdf['x'], tdf['y'])
                        
                    gammav_model = rvogfwdvol(hv_val, ext_val, incclip, kzclip, rngslope=rngslopeclip)
                    gamma_model = phiclip * (muclip + tdfclip*gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
                else:
                    if isinstance(ext, dict):
                        extclip = np.interp(hv_val, ext['x'], ext['y'])
                        
                    gammav_model = rvogfwdvol(hv_val, extclip, incclip, kzclip, rngslope=rngslopeclip)
                    tdf_val = np.abs((gammaclip*(muclip+1) - phiclip*muclip)/(phiclip*gammav_model))
                    gamma_model = phiclip * (muclip + tdf_val*gammav_model) / (muclip + 1)
                    dist = np.abs(gammaclip - gamma_model)
    
                # If potential vegetation height is greater than
                # 2*pi ambiguity height, and the limit2pi option
                # is set to True, remove these as potential solutions:               
                ind_limit = limit2piclip & (hv_val > np.abs(2*np.pi/kzclip))
                if np.any(ind_limit):
                    dist[ind_limit] = 1e10
                
                # If hv_min and hv_max were set to arrays,
                # ensure that solutions outside of the bounds are excluded.
                if hv_min_clip is not None:
                    ind_limit = (hv_val < hv_min_clip)
                    if np.any(ind_limit):
                        dist[ind_limit] = 1e10
                
                if hv_max_clip is not None:
                    ind_limit = (hv_val > hv_max_clip)
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
    
    if not silent:
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


def rvogblselect(gamma, kz, method='prod', minkz=0.0314, gammaminor=None):
    """From a multi-baseline dataset, select the baseline for each pixel that
        we expect to produce the best forest height estimate using the RVoG
        model.
        
        There are multiple different methods implemented here for ranking
        the baselines.  These are chosen using the method keyword argument.
        The default is method='prod', which selects the baseline with the
        highest product between the coherence region major axis line length
        and the magnitude of the complex average of the high and low
        coherences.  Essentially, this method prefers baselines which
        have both a long coherence region (e.g., a large phase separation
        between the high and low coherences) as well as a high overall
        coherence magnitude.
        
        The second method is 'line', which takes the product between
        the coherence region major axis separation times the minimum distance
        between the origin of the complex plane and the line segment fitted
        to the optimized coherences.  This is fairly similar to the previous
        method, but will produce different results in some cases.  This
        criteria and the previous option were suggested by Marco Lavalle.
        
        The second method is 'ecc', which selects the baseline with the
        highest coherence region eccentricity, favoring baselines
        with coherence regions that have a large axial ratio (major axis
        divided by minor axis).
        
        The last option is method='var', which favours the baseline with
        the smallest expected height variance, which is calculated using
        the Cramer-Rao Lower Bound for the phase variance.  For details on
        these last two selection criteria, see the paper:
        
        S. K. Lee, F. Kugler, K. P. Papathanassiou, I. Hajnsek,
        "Multibaseline polarimetric SAR interferometry forest height
        inversion approaches", POLinSAR ESA-ESRIN, 2011-Jan. 
        
        Arguments:
            gamma (array): Array of coherence values for the
                multi-baseline dataset.  Should have dimensions
                (bl, coh, azimuth, range).  bl is the baseline index, and coh
                is the coherence index (e.g., the high and low optimized
                coherences).  Note that if you are using the eccentricity
                selection method, gamma needs to contain the high and low
                coherences, as the shape of the coherence region is used
                in the selection process.  If you are using the height
                variance selection method, however, the input coherences
                can be any coherences you wish to use.  The mean coherence
                magnitude of the input coherences for each pixel will be
                used to calculate the height variance.
            kz (array): Array of kz values for the multi-baseline dataset.
                Should have shape (baseline, azimuth, range).
            gammaminor (array): If using the eccentricity selection method,
                this keyword argument needs to be given an array with the
                same shape as gamma, containing the two coherences along the
                minor axis of the coherence region (e.g., the optimized
                coherences with the smallest separation).  These can be
                calculated and saved by calling kapok.Scene.opt(saveall=True).
                See the documentation for kapok.Scene.opt for more details.
                Default: None.
            method (str): String which determines the method to use for
                selecting the baselines.  Options are 'ecc'
                (for eccentricity),  'var' (for height variance), or
                'prod' (for product of coherence region major axis and
                coherence magnitude).  See main function description above
                for more details.  Default: 'prod'.
            minkz (float): For a baseline to be considered, the absolute
                value of kz must be at least this amount.  This keyword
                argument allows baselines with zero spatial separation to be
                excluded.  Default: 0.0314 (e.g., if pi height is greater
                than 100m, that baseline will be excluded).
        
        Returns:
            gammasel (array): Array of coherences, for the selected
                baselines only.  Has shape (2, azimuth, range).
            kzsel (array): The kz values of the selected baselines for each
                pixel.  Has shape (azimuth, range).
            blsel (array): For each pixel, an array containing the baseline index
                of the baseline that was chosen.
    
    """
    if 'line' in method: # Line Length * Separation Product Method
        from kapok.lib import linesegmentdist
        print('kapok.rvog.rvogblselect | Performing incoherent multi-baseline RVoG inversion.  Selecting baselines using product of fitted line distance from origin and coherence separation. ('+time.ctime()+')')
        sep = np.abs(gamma[:,0] - gamma[:,1])
        dist = linesegmentdist(0, gamma[:,0], gamma[:,1])
        criteria = sep * dist
    elif 'var' in method: # Height Variance Method
        # Note: We don't include the number of looks in the equation, as we
        # assume the coherence for all of the baselines have been estimated
        # using the same number of looks, so it does not affect the
        # selection.
        print('kapok.rvog.rvogblselect | Performing incoherent multi-baseline RVoG inversion.  Selecting baselines using height variance. ('+time.ctime()+')')
        criteria = np.abs(gamma[:,0]) ** 2
        criteria = -1*np.sqrt((1-criteria)/2/criteria)/np.abs(kz)
    elif 'ecc' in method: # Eccentricity Method
        if gammaminor is not None:
            print('kapok.rvog.rvogblselect | Performing incoherent multi-baseline RVoG inversion.  Selecting baselines using coherence region eccentricity. ('+time.ctime()+')')
            criteria = (np.abs(gammaminor[:,0] - gammaminor[:,1])/np.abs(gamma[:,0] - gamma[:,1])) ** 2
            criteria = np.sqrt(1 - criteria)
        else:
            print('kapok.rvog.rvogblselect | Using eccentricity method for baseline selection, but gammaminor keyword has not been set.  Aborting.')
            return None
    else: # Default to Coherence Magnitude * Separation Product Method
        print('kapok.rvog.rvogblselect | Performing incoherent multi-baseline RVoG inversion.  Selecting baselines using product of average coherence magnitude and separation. ('+time.ctime()+')')
        sep = np.abs(gamma[:,0] - gamma[:,1])
        mag = np.abs(gamma[:,0] + gamma[:,1])
        criteria = sep * mag
    
    
    # Remove too small baselines.
    criteria[np.abs(kz) < minkz] = -1e6
    
    # Now shuffle the coherences and kz values around to return the baselines
    # with the highest criteria value for each pixel.
    blsel = np.argmax(criteria, axis=0)
    az = np.tile(np.arange(gamma.shape[2]),(gamma.shape[3],1)).T
    rng = np.tile(np.arange(gamma.shape[3]),(gamma.shape[2],1))
       
    gammasel = gamma[blsel,:,az,rng]
    gammasel = np.rollaxis(gammasel, 2)
    kzsel = kz[blsel,az,rng]
    
    return gammasel, kzsel, blsel