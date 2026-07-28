# -*- coding: utf-8 -*-
"""Neumann et al. Structure Model

    Contains functions for the forward Neumann structure model.  The model
    is implemented as described in:
    
    Neumann, M., Ferro-Famil, L., & Reigber, A. (2010). Estimation of forest
    structure, ground, and canopy layer characteristics from multibaseline
    polarimetric interferometric SAR data. IEEE Transactions on Geoscience
    and Remote Sensing. http://doi.org/10.1109/TGRS.2009.2031101

    At the moment, these functions are primarily used by the simulation
    module in kapok.sim.  Model inversion is not yet implemented.
    
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

import numpy as np
from scipy import special

from kapok.rvog import rvogfwdvol



def converttau(tau):
    """Given a value for the degree of orientation randomness paramter, tau,
        calculate the value of the degree of concentration, kappa, then
        calculate the values of the g and gc parameters from that.
        
        The following equations are used:
        
        tau = I0(kappa) * np.exp(-kappa)    
        g = I2(kappa) / I0(kappa)    
        gc = I1(kappa) / I0(kappa)
        
        I0 is the modified Bessel function of order 0, I1 is the modified
        Bessel function of order 1, and I2 is the modified Bessel function of
        order 2.
        
        Arguments:    
            tau: The degree of orientation randomness, ranging from 0 to 1.
        
        Returns:    
            g: The value of the g parameter.
            gc: The value of the gc parameter.
        
    """
    kappa_vec = np.linspace(0,100,num=1001)
    kappa_vec = np.flipud(np.sort(np.append(kappa_vec,[0,200,300,400,500,600,700])))
    tau_vec = special.i0e(kappa_vec)
    
    kappa = np.interp(tau, tau_vec, kappa_vec)
    
    g = special.iv(2,kappa) / special.i0(kappa)
    gc = special.i1(kappa) / special.i0(kappa)
    
    
    if isinstance(tau, (collections.Sequence, np.ndarray)):
        ind = (~np.isfinite(g))
        g[ind] = 1
        
        ind = (~np.isfinite(gc))
        gc[ind] = 1
    else:
        if ~np.isfinite(g):
            g = 1
        
        if ~np.isfinite(gc):
            gc = 1

    
    return g, gc
    
    
def volumecoherency(delta, g, gc):
    """Given values for the delta (particle scattering anisotropy), g, and gc
        parameters, calculate the coherency matrix of the volume layer.
    
        Arguments:    
            delta: The complex-valued particle scattering anisotropy.
            g: g parameter calculated from tau (the degree of orientation
                randomness).
            gc: gc parameter calculated from tau (the degree of orientation
                randomness).
        
        Returns:    
            tv: The 3x3 reflection symmetric coherency matrix for the volume
                layer with dimensions (3, 3, delta.shape).
    
    """
    tvshape = np.append([3,3],delta.shape)
    tv = np.zeros(tvshape, dtype='complex64')
       
    tv[0,0] = 1
    tv[0,1] = gc*delta
    tv[1,0] = gc*np.conj(delta)
    tv[1,1] = 0.5*(1+g)*np.square(np.abs(delta))
    tv[2,2] = 0.5*(1-g)*np.square(np.abs(delta))
    
    return tv
    
    
def groundcoherency(beta, beta22, beta33):
    """Given values for the beta, beta22, and beta33 parameters, calculate the
        coherency matrix for the ground layer.
        
        Arguments:    
            beta: The beta parameter values of the ground surface layer.
            beta22: The beta22 parameter values (the values in the (2,2)
                element of the ground coherency matrix).
            beta33: The beta33 parameter values (the values in the (3,3)
                element of the ground coherency matrix).
        
        Returns:   
            tg: The 3x3 reflection symmetric coherency matrix for the ground
                layer, with dimensions of (3, 3, beta.shape).
    
    """
    tgshape = np.append([3,3],beta.shape)
    tg = np.zeros(tgshape, dtype='complex64')
    
    tg[0,0] = 1
    tg[0,1] = beta
    tg[1,0] = np.conj(beta)
    tg[1,1] = beta22
    tg[2,2] = beta33
    
    return tg
           
    
def sbcoherency(fv, fg, g, gc, delta, beta, beta22, beta33, hv=None, ext=None,
                gammaphi=None, alphav=None, inc=None, kz=None):
    """Calculate the two layer coherency (including both ground and volume
        layers) for a single baseline (e.g., this produces a 3x3 matrix).
        
        If you specify the parameters hv, ext, gammaphi, alphav, inc, and kz,
        this function will return an interferometric Omega matrix.  If these
        parameters are not specified, this function will return a polarimetric
        T matrix.
        
        Arguments:    
            fv: The intensity normalization factor for the volume layer.
            fg: The intensity normalization factor for the ground layer.
            g: The value of the g parameter for the volume coherency.
            gc: The value of the gc parameter for the volume coherency.
            delta: The particle scattering anisotropy.
            beta: The complex-valued ground coherency parameter in the (1,1)
                element of the ground coherency matrix.
            beta22: The real-valued ground coherency parameter in the (2,2)
                element of the ground coherency matrix.
            beta33: The real-valued ground coherency parameter in the (3,3)
                element of the ground coherency matrix.
            hv:  The vegetation heights, in meters.  This, and the following
                arguments, are only required if you wish to calculate an Omega
                matrix.  Note that if you specify hv, you must also specify
                all of the following parameters as well.
            ext: The extinction parameter, in Np/m.
            gammaphi: The complex-valued ground coherence.
            alphav: The decorrelation factor of the volume coherence.
            inc: The incidence angle, in radians.
            kz: The vertical wavenumber, in rad/m.
        
        Returns:
            m: the modelled 3x3 coherency matrix (either T or Omega) for the
                given input parameters.  Will have dimensions of (3, 3,
                par.shape) where (par.shape) is the dimensions of the input
                parameter arrays (which should all be equal).
    
    """
    tg = fg * groundcoherency(beta, beta22, beta33)
    tv = fv * volumecoherency(delta, g, gc)    
    
    if hv is None:
        m = tg + tv
    else:
        m = (tg * gammaphi) + (tv * rvogfwdvol(hv, ext, inc, kz) * alphav * np.exp(1j*np.angle(gammaphi)))
        
    return m