# -*- coding: utf-8 -*-
"""Windowing library functions (rebinning and smoothing).

    Author: Maxim Neumann
	
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
import scipy.ndimage

def mlook(data, mlwin):
    """Multilook/rebin image to smaller image by averaging.
    
    Arguments:
        data: Array (up to 4D) containing data to multilook.
        mlwin (tuple, int): Tuple of ints containing the smoothing window
            sizes in each dimension.
            
    Returns:
        mldata: Array containing multilooked data.

    """
    if mlwin == (1,1):
        return data
    
    data = np.asarray(data)
    n_dim = len(data.shape)
        
    nshape = np.array(data.shape) // np.array(list(mlwin) + [1]*(n_dim-len(mlwin)))

    sh = np.array([ [nshape[i], data.shape[i]//nshape[i]] for i,x in enumerate(nshape) ]).flatten()
        
    if n_dim == 2:
        if not any(np.mod(data.shape, nshape)):
            return data.reshape(sh).mean(-1).mean(1)
        else:
            return data[0:sh[0]*sh[1],0:sh[2]*sh[3]].reshape(sh).mean(-1).mean(1)
    elif n_dim == 3:
        if not any(np.mod(data.shape, nshape)):
            return data.reshape(sh).mean(1).mean(2).mean(3)
        else:
            return data[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5]]\
                .reshape(sh).mean(1).mean(2).mean(3)
    elif n_dim == 4:
        if not any(np.mod(data.shape, nshape)):
            return data.reshape(sh).mean(1).mean(2).mean(3).mean(4)
        else:
            return data[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5],0:sh[6]*sh[7]]\
                .reshape(sh).mean(1).mean(2).mean(3).mean(4)
    else:
        print('Error in mlook: given number of dimensions not considered.')


def smooth(data, smwin, **kwargs):
    """Smoothing with a boxcar moving average. Uses
    scipy.ndimage.uniform_filter.
    
    Arguments:
        data: Array containing data to smooth.
        smwin (tuple, int): Tuple of ints containing the smoothing window
            sizes in each dimension.
        
    Returns:
        smdata: Array containing boxcar averaged data.
    
    """
    if smwin == (1,1):
        return data
    elif type(data[0,0]) in [complex,np.complex64,np.complex128]:
        res = np.empty(data.shape, dtype=complex)
        res.real = scipy.ndimage.uniform_filter(np.real(data), smwin, **kwargs)
        res.imag = scipy.ndimage.uniform_filter(np.imag(data), smwin, **kwargs)
        return res
    else:
        return scipy.ndimage.uniform_filter(data, smwin, **kwargs)
