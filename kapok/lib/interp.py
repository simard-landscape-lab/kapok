# -*- coding: utf-8 -*-
"""Interpolation functions.

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
from scipy import ndimage as nd


def bilinear_interpolate(data, x, y):
    """Function to perform bilinear interpolation on the input array data, at
    the image coordinates given by input arguments x and y.
    
    Arguments
        data (array): 2D array containing raster data to interpolate.
        x (array): the X coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that X refers to the second
            dimension of data (e.g., the columns).
        y (array): the Y coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that Y refers to the first
            dimension of data (e.g., the rows).
            
    Returns:
        intdata (array): The 2D interpolated array, with same dimensions as
            x and y.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Get lower and upper bounds for each pixel.
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # Clip the image coordinates to the size of the input data.
    x0 = np.clip(x0, 0, data.shape[1]-1);
    x1 = np.clip(x1, 0, data.shape[1]-1);
    y0 = np.clip(y0, 0, data.shape[0]-1);
    y1 = np.clip(y1, 0, data.shape[0]-1);

    data_ll = data[ y0, x0 ] # lower left corner image values
    data_ul = data[ y1, x0 ] # upper left corner image values
    data_lr = data[ y0, x1 ] # lower right corner image values
    data_ur = data[ y1, x1 ] # upper right corner image values

    w_ll = (x1-x) * (y1-y) # weight for lower left value
    w_ul = (x1-x) * (y-y0) # weight for upper left value
    w_lr = (x-x0) * (y1-y) # weight for lower right value
    w_ur = (x-x0) * (y-y0) # weight for upper right value
    
    # Where the x or y coordinates are outside of the image boundaries, set one
    # of the weights to nan, so that these values are nan in the output array.
    w_ll[np.less(x,0)] = np.nan
    w_ll[np.greater(x,data.shape[1]-1)] = np.nan
    w_ll[np.less(y,0)] = np.nan
    w_ll[np.greater(y,data.shape[0]-1)] = np.nan
    
    intdata = w_ll*data_ll + w_ul*data_ul + w_lr*data_lr + w_ur*data_ur

    return intdata
    
    
def nnfill(data):
    """Function to fill nan values in a 2D array using nearest neighbor
    interpolation.
    
    Arguments:
        data (array): A 2D array containing the data to fill.  Void elements
            should have values of np.nan.
    
    Returns:
        filled (array): The filled data.
    
    """
    ind = nd.distance_transform_edt(np.isnan(data), return_distances=False, return_indices=True)
    return data[tuple(ind)]
    