# -*- coding: utf-8 -*-
"""Various helper functions for dealing with GDAL datasets, and resampling
    them.

    Author: Michael Denbina

    Copyright 2016-2024 California Institute of Technology.  All rights reserved.
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
import subprocess

import numpy as np
from osgeo import gdal, osr

from .interp import bilinear_interpolate, nearest_interpolate


def xy2rc(ds_geo, x, y, method='bilinear'):
    """ Convert X, Y map coordinates to row, column raster coordinates. """
    y_origin = ds_geo[3]
    x_origin = ds_geo[0]

    # Figure out row, col from x, y.
    rows = (y - y_origin) / ds_geo[5]
    cols = (x - x_origin) / ds_geo[1]

    return rows, cols


def get_geolocation_arrays(gdal_raster_ds):
    """ Return geolocation arrays (arrays of the x and y coordinates for each
        pixel) for the given GDAL raster. """
    geo = gdal_raster_ds.GetGeoTransform()

    x_array = np.tile(np.linspace(geo[0], geo[0]+geo[1]*(gdal_raster_ds.RasterXSize-1), num=gdal_raster_ds.RasterXSize), (gdal_raster_ds.RasterYSize, 1))
    y_array = np.tile(np.linspace(geo[3], geo[3]+geo[5]*(gdal_raster_ds.RasterYSize-1), num=gdal_raster_ds.RasterYSize), (gdal_raster_ds.RasterXSize, 1)).T

    return x_array, y_array


def interp_raster_at_xy(ds_array, ds_geo, x, y, method='bilinear'):
    """ Interpolate values of raster at input coordinates. """
    y_origin = ds_geo[3]
    x_origin = ds_geo[0]

    # Figure out row, col from x, y.
    rows = (y - y_origin) / ds_geo[5]
    cols = (x - x_origin) / ds_geo[1]

    if method == 'bilinear':
        ds_interp = bilinear_interpolate(ds_array, cols, rows)
    else:
        ds_interp = nearest_interpolate(ds_array, cols, rows)

    return ds_interp


def get_geoloc_from_raster(ds_array, ds_geo):
    """ Return grids of the X and Y coordinates for each pixel in a given
        raster image. """
    x_origin = ds_geo[0]
    y_origin = ds_geo[3]

    x_grid = np.tile(np.linspace(x_origin, x_origin + (ds_array.shape[1]-1)*ds_geo[1], num=ds_array.shape[1]), (ds_array.shape[0], 1))
    y_grid = np.tile(np.linspace(y_origin, y_origin + (ds_array.shape[0]-1)*ds_geo[5], num=ds_array.shape[0]), (ds_array.shape[1], 1)).T

    return x_grid, y_grid


def interp_raster_to_raster(source_array, source_geo, target_array, target_geo, method='bilinear'):
    """ Interpolate values of source raster at target raster coordinates.  The
        two rasters must have the same projection for this function to work,
        because this function does not do any coordinate transform, it only
        performs interpolation. """
    x_target_grid, y_target_grid = get_geoloc_from_raster(target_array, target_geo)

    if source_array.ndim == 2: # Single-Band Raster
        source_interp = interp_raster_at_xy(source_array, source_geo, x_target_grid, y_target_grid, method=method)
    elif source_array.ndim == 3: # Multi-Band Raster
        source_interp = np.zeros((source_array.shape[0], x_target_grid.shape[0], x_target_grid.shape[1]), dtype=source_array.dtype)
        for band in range(source_interp.shape[0]):
            source_interp[band] = interp_raster_at_xy(source_array[band], source_geo, x_target_grid, y_target_grid, method=method)
    else:
        print('interp_raster_to_raster -- Number of raster dimensions not supported!')
        source_interp = None

    return source_interp


def write_raster(file, data, output_transform, oformat='GTiff', epsg=4326, dtype=gdal.GDT_Float32, nodataval=None, create_overviews=False, creation_options=None):
    """ Write a raster file using GDAL. """
    driver = gdal.GetDriverByName(oformat)

    if data.ndim == 2: # single-band data
        if creation_options is None:
            out = driver.Create(file, data.shape[1], data.shape[0], 1, dtype)
        else:
            out = driver.Create(file, data.shape[1], data.shape[0], 1, dtype, options=creation_options)

        out.GetRasterBand(1).WriteArray(data)
        if nodataval is not None:
            out.GetRasterBand(1).SetNoDataValue(nodataval)
    elif data.ndim == 3: # multi-band data
        num_bands = data.shape[0]
        if creation_options is None:
            out = driver.Create(file, data.shape[2], data.shape[1], num_bands, dtype)
        else:
            out = driver.Create(file, data.shape[2], data.shape[1], num_bands, dtype, options=creation_options)

        for band_num in range(num_bands):
            out.GetRasterBand(band_num + 1).WriteArray(data[band_num])
            if nodataval is not None:
                out.GetRasterBand(band_num + 1).SetNoDataValue(nodataval)
    else:
        print('Given data dimensions not supported by write_raster().')
        return

    out.SetGeoTransform(output_transform)
    out_srs = osr.SpatialReference()
    out_srs.ImportFromEPSG(epsg)
    out.SetProjection(out_srs.ExportToWkt())
    out = None

    # Generate overviews.
    if create_overviews:
        cmd = 'gdaladdo -r average {} 8 16 32 64 128 256'.format(file)
        print(cmd)
        print(subprocess.getoutput(cmd))

    return