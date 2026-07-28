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

    # Where the x or y coordinates are outside of the image boundaries, set all
    # of the weights to nan, so that these values are nan in the output array.
    ind = (x < 0) | (x > data.shape[1]-1) | (y < 0) | (y > data.shape[0]-1)
    if np.any(ind):
        w_ll[ind] = np.nan
        w_ul[ind] = np.nan
        w_lr[ind] = np.nan
        w_ur[ind] = np.nan
        data_ll[ind] = np.nan
        data_ul[ind] = np.nan
        data_lr[ind] = np.nan
        data_ur[ind] = np.nan

    # Handle void data.
    ind_all_void = np.isnan(data_ll) & np.isnan(data_ul) & np.isnan(data_lr) & np.isnan(data_ur)
    ind_some_void = np.isnan(data_ll) & (~ind_all_void)
    if np.any(ind_some_void):
        w_ll[ind_some_void] = 0
        data_ll[ind_some_void] = 0
    ind_some_void = np.isnan(data_ul) & (~ind_all_void)
    if np.any(ind_some_void):
        w_ul[ind_some_void] = 0
        data_ul[ind_some_void] = 0
    ind_some_void = np.isnan(data_lr) & (~ind_all_void)
    if np.any(ind_some_void):
        w_lr[ind_some_void] = 0
        data_lr[ind_some_void] = 0
    ind_some_void = np.isnan(data_ur) & (~ind_all_void)
    if np.any(ind_some_void):
        w_ur[ind_some_void] = 0
        data_ur[ind_some_void] = 0

    intdata = w_ll*data_ll + w_ul*data_ul + w_lr*data_lr + w_ur*data_ur

    if np.any(ind_all_void):
        intdata[ind_all_void] = np.nan
    ind_low_weight = (w_ll + w_ul + w_lr + w_ur) < 0.99
    if np.any(ind_low_weight):
        intdata[ind_low_weight] = np.nan

    return intdata


def nearest_interpolate(data, x, y, nodataval=np.nan):
    """Function to perform nearest neighbor interpolation on the input array
        data, at the image coordinates given by input arguments x and y.

    Arguments
        data (array): 2D array containing raster data to interpolate.
        x (array): the X coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that X refers to the second
            dimension of data (e.g., the columns).
        y (array): the Y coordinate values at which to interpolate (in array
            indices, starting at zero).  Note that Y refers to the first
            dimension of data (e.g., the rows).
        nodataval: No data value for points outside of the data bounds.

    Returns:
        intdata (array): The 2D interpolated array, with same dimensions as
            x and y.

    """
    x = np.asarray(x)
    y = np.asarray(y)

    x = np.round(x).astype('int32')
    y = np.round(y).astype('int32')

    ind = (x < 0) | (y < 0) | (x >= data.shape[1]) | (y >= data.shape[0])

    if np.any(ind):
        x[ind] = 0
        y[ind] = 0

    intdata = data[y, x]

    if np.any(ind):
        intdata[ind] = nodataval

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
