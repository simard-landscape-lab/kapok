# -*- coding: utf-8 -*-
"""Windowing library functions (rebinning and smoothing).

    Authors: Maxim Neumann (mlook and smooth), Michael Denbina
        (nanmlook and nansmooth)
    
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
        print('kapok.lib.mlook | Given number of dimensions not considered.  Aborting.')


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
        
        
def gaussiansmooth(data, std, truncate=3.0):
    """Smoothing with a 2D Gaussian filter. Uses
        scipy.ndimage.filters.gaussian_filter.  Non-finite values anywhere
        in the smoothing window will result in that pixel in the filtered
        result also containing a nan value.
    
    Arguments:
        data: Array containing data to smooth.
        std (float): Standard deviation of the Gaussian filter, in pixel
            units.
        truncate (float): The number of standard deviations at which to
            truncate the filter.  Default: 3.0
        
    Returns:
        filtdata: Array containing Gaussian filtered data.
    
    """    
    filtdata = scipy.ndimage.filters.gaussian_filter(data, std, truncate=truncate)
    
    return filtdata
        
        
def nanmlook(data, mlwin):
    """Multilook/rebin image to smaller image by averaging.  Ignores nan values
    in input data.
    
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
            return np.nanmean(np.nanmean(data.reshape(sh),axis=-1),axis=1)
        else:
            return np.nanmean(np.nanmean(data[0:sh[0]*sh[1],0:sh[2]*sh[3]].reshape(sh),axis=-1),axis=1)
    elif n_dim == 3:
        if not any(np.mod(data.shape, nshape)):
            return np.nanmean(np.nanmean(np.nanmean(data.reshape(sh),axis=1),axis=2),axis=3)
        else:
            return np.nanmean(np.nanmean(np.nanmean(data[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5]]\
                .reshape(sh),axis=1),axis=2),axis=3)
    elif n_dim == 4:
        if not any(np.mod(data.shape, nshape)):
            return np.nanmean(np.nanmean(np.nanmean(np.nanmean(data.reshape(sh),axis=1),axis=2),axis=3),axis=4)
        else:
            return np.nanmean(np.nanmean(np.nanmean(np.nanmean(data[0:sh[0]*sh[1],0:sh[2]*sh[3],0:sh[4]*sh[5],0:sh[6]*sh[7]]\
                .reshape(sh),axis=1),axis=2),axis=3),axis=4)
    else:
        print('kapok.lib.mlooknan | Given number of dimensions not considered.  Aborting.')
        
        
def nansmooth(data, smwin, **kwargs):
    """Smoothing with a boxcar moving average. Uses
        scipy.ndimage.uniform_filter.  Ignores non-finite values in input
        data (these elements will not be included in the boxcar average).
        Note that if the window contains no finite values, the smoothed
        data will also contain a nan value for that element.
    
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
        valid = np.isfinite(data)
        data_nonan = data.copy()
        data_nonan[~valid] = 0
        winsize = np.prod(smwin)
        
        res = np.empty(data.shape, dtype=complex)
        res.real = scipy.ndimage.uniform_filter(np.real(data_nonan), smwin, **kwargs)
        res.imag = scipy.ndimage.uniform_filter(np.imag(data_nonan), smwin, **kwargs)
        
        num_valid = np.round(scipy.ndimage.uniform_filter(valid.astype('float32'), smwin, **kwargs) * winsize)
        res.real[num_valid > 0] = res.real[num_valid > 0] * winsize / num_valid[num_valid > 0]
        res.real[num_valid <= 0] = np.nan
        res.imag[num_valid > 0] = res.imag[num_valid > 0] * winsize / num_valid[num_valid > 0]
        res.real[num_valid <= 0] = np.nan
        return res
    else:
        valid = np.isfinite(data)
        data_nonan = data.copy()
        data_nonan[~valid] = 0
        winsize = np.prod(smwin)
        
        res = scipy.ndimage.uniform_filter(data_nonan, smwin, **kwargs)
        
        num_valid = np.round(scipy.ndimage.uniform_filter(valid.astype('float32'), smwin, **kwargs) * winsize)
        res[num_valid > 0] *= winsize / num_valid[num_valid > 0]
        res[num_valid <= 0] = np.nan
        return res


def nangaussiansmooth(data, std, truncate=3.0):
    """Smoothing with a 2D Gaussian filter. Uses
        scipy.ndimage.filters.gaussian_filter.  Ignores non-finite values in
        input data.  Note that if the window contains no finite values, the
        smoothed data will also contain a nan value for that element.
    
    Arguments:
        data: Array containing data to smooth.
        std (float): Standard deviation of the Gaussian filter, in pixel
            units.
        truncate (float): The number of standard deviations at which to
            truncate the filter.  Default: 3.0
        
    Returns:
        filtdata: Array containing Gaussian filtered data.
    
    """
    # Replace nan-values with zeros.
    datamasked = data.copy()
    datamasked[~np.isfinite(data)] = 0
    
    weights = np.ones(data.shape,dtype='float32')
    weights[~np.isfinite(data)] = 0
    
    datamasked = scipy.ndimage.filters.gaussian_filter(datamasked, std, truncate=truncate)
    weights = scipy.ndimage.filters.gaussian_filter(weights, std, truncate=truncate)   
    weights[np.isclose(weights,0)] == -1
    
    filtdata = datamasked / weights
    filtdata[weights < 0] = np.nan
    
    return filtdata