# -*- coding: utf-8 -*-
"""Geocoding Module

    Use GDAL to make output products resampled to geographic projection with
    constant lat/lon spacing.
    
    Authors: Brian Hawkins, Michael Denbina
	
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
import os
import subprocess
import sys

import numpy as np
try:
    import pyresample as pr
except ImportError:
    pass

from kapok.lib import bilinear_interpolate



def radar2ll_pr(outpath, datafile, data, lat, lon, outformat='ENVI',
                nodataval=None, tr=2.7777778e-4, **kwargs):
    """Create a geocoded file, in geographic projection, from input data
    in azimuth, slant range radar coordinates.

    Uses latitude and longitude arrays containing the geographic coordinates
    of each pixel (geolocation arrays), in order to perform the resampling
    using the pyresample Python library
    (https://pypi.python.org/pypi/pyresample).
    
    Author: Brian Hawkins

    Arguments:
        outpath (str): The path in which to save the geocoded file, as well as
            temporary latitude/longitude files used during the resampling
            process.
        datafile (str): The output file name for the geocoded file.
        data (array): 2D array containing the data to geocode.  Should be
            in float32 format.  (If it isn't, it will be converted to it.)
            If resampling of complex-valued parameters is needed, geocode
            the real and imaginary parts separately using this function.
        lat (array): 2D array containing the latitudes for each pixel, in
            degrees.
        lon (array): 2D array containing the longitudes for each pixel, in
            degrees.
        outformat (str): The output format.  Should be an identifying string
            recognized by GDAL.  Default is 'ENVI'.  Other options include
            'GTiff' or 'KEA', etc.  For reference, see
            http://www.gdal.org/formats_list.html.
        nodataval:  No data value for the output raster.  This will be the
            value of the raster for all pixels outside the input data
            extent.  Default: None (points will be set to zero).
        tr (float): Set output file resolution (in degrees).  Can be set
            to a tuple to set (longitude, latitude) resolution separately.
            Default: 2.7777778e-4 (1 arc second).

    """
    if nodataval is None:
        nodataval = 0.0
        
    if sys.byteorder == 'little':
        byte = 'LSB'
    else:
        byte = 'MSB'

    if outpath != '':
        outpath = outpath + '/'

    # Figure out output posting.
    if isinstance(tr, tuple):
        dlon, dlat = [float(x) for x in tr[:2]]
    else:
        dlon = dlat = float(tr)
    dlat = abs(dlat)
    dlon = abs(dlon)

    # Figure out bounding box in lat/lon domain.
    lat0 = np.max(lat)
    lat1 = np.min(lat)
    lon0 = np.min(lon)
    lon1 = np.max(lon)

    # Clip to integer dlat/dlon grid.
    pad = 10.5
    ilat0 = int(lat0/dlat + pad)
    ilat1 = int(lat1/dlat - pad)
    ilon0 = int(lon0/dlat - pad)
    ilon1 = int(lon1/dlat + pad)
    nlat = ilat0 - ilat1
    nlon = ilon1 - ilon0
    lat0 = ilat0 * dlat
    lon0 = ilon0 * dlon

    # Define output grid.
    x = lon0 + dlon * np.arange(nlon)
    y = lat0 - dlat * np.arange(nlat)
    X, Y = np.meshgrid(x, y)
    area = pr.geometry.GridDefinition(lons=X, lats=Y)

    # Define input grid.
    swath = pr.geometry.SwathDefinition(lons=lon, lats=lat)

    # Figure out scale for Gaussian smoothing, assuming we're on Earth.
    dx = 6378137. * np.radians(dlat)
    scale = pr.utils.fwhm2sigma(dx)

    # Cast input data to float32.
    data = data.astype('float32')

    # TODO: Number of neighbors to query is related to ratio of input and
    # output grid spacing.
    nn = kwargs.get('nn', 36)
    out = pr.kd_tree.resample_gauss(swath, data, area, radius_of_influence=3*dx,
                                    sigmas=scale, neighbours=nn, segments=1,
                                    fill_value=nodataval)

    # Careful, these routines can promote floats to doubles.
    outdata = out.astype('f4')
    outfile = outpath + 'out.dat'
    with open(outfile, 'w') as f:
        outdata.tofile(f)

    # Create a VRT file.
    lineoffset = 4 * nlon
    # Pyresample doesn't seem to have a clear statement of what the pixel
    # address convention is, so assume centered coordinates.  Move to edge for
    # VRT file.  This gives good agreement with GDAL version (as measured by
    # ampcor) and good self-consistency between different resolutions (as
    # observed in Google Earth).
    ullat = lat0 + 0.5 * dlat
    ullon = lon0 - 0.5 * dlon
    vrt = """\
        <VRTDataset rasterXSize="{nlon}" rasterYSize="{nlat}">
          <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]</SRS>
          <GeoTransform> {ullon:0.17g}, {dlon:0.17g}, 0.0, {ullat:0.17g}, 0.0, -{dlat:0.17g}</GeoTransform>
          <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">
            <SourceFilename relativetoVRT="1">{outfile}</SourceFilename>
            <ImageOffset>0</ImageOffset>
            <PixelOffset>4</PixelOffset>
            <LineOffset>{lineoffset}</LineOffset>
            <ByteOrder>{byte}</ByteOrder>
          </VRTRasterBand>
        </VRTDataset>
    """.format(**locals())
    vrtname = outpath + 'out.vrt'
    with open(vrtname, 'w') as f:
        f.write(vrt)

    # Call GDAL
    command = 'gdal_translate -ot Float32 -of ' + outformat
    if nodataval is not None:
        command = command + ' -a_nodata '+str(nodataval)
    command = ' '.join((command, vrtname, outpath+datafile))
    print(command)
    print(subprocess.getoutput(command))

    # Remove temporary files.
    os.remove(vrtname)
    if 'VRT' not in outformat.upper():
        os.remove(outfile)
    
    return


def radar2ll_gdal(outpath, datafile, data, lat, lon, outformat='ENVI',
                  resampling='bilinear', nodataval=None, tr=2.7777778e-4):
    """Create a geocoded file, in geographic projection, from input data
    in azimuth, slant range radar coordinates.
    
    Uses latitude and longitude arrays containing the geographic coordinates
    of each pixel (geolocation arrays), in order to perform the resampling
    using gdalwarp.  For gdalwarp reference, see
    http://www.gdal.org/gdalwarp.html.
    
    Author: Michael Denbina
    
    Arguments:
        outpath (str): The path in which to save the geocoded file, as well as
            temporary latitude/longitude files used during the resampling
            process.
        datafile (str): The output file name for the geocoded file.
        data (array): 2D array containing the data to geocode.  Should be
            in float32 format.  (If it isn't, it will be converted to it.)
            If resampling of complex-valued parameters is needed, geocode
            the real and imaginary parts separately using this function.
        lat (array): 2D array containing the latitudes for each pixel, in
            degrees.
        lon (array): 2D array containing the longitudes for each pixel, in
            degrees.
        outformat (str): The output format.  Should be an identifying string
            recognized by GDAL.  Default is 'ENVI'.  Other options include
            'GTiff' or 'KEA', etc.  For reference, see
            http://www.gdal.org/formats_list.html.
        resampling (str): String identifying the resampling method.  Options
            include 'near', 'bilinear', 'cubic', 'lanczos', and others.
            Default is 'bilinear'.  For reference and more options, see
            http://www.gdal.org/gdalwarp.html.
        nodataval:  No data value for the output raster.  This will be the
            value of the raster for all pixels outside the input data
            extent.  Default: None.
        tr (float): Set output file resolution (in degrees).  Can be set
            to a tuple to set (longitude, latitude) resolution separately.
            Can be set to None in order for gdalwarp to automatically choose
            an output resolution based on input data spacing.
            Default: 2.7777778e-4 (1 arc second).
    
    """   
    if sys.byteorder == 'little':
        byte = 'LSB'
    else:
        byte = 'MSB'
        
    if outpath != '':
        outpath = outpath + '/'        
        
    # Save the lat/lon to temporary flat binary files.
    lat.tofile(outpath+'templat.dat')
    lon.tofile(outpath+'templon.dat')
    
    # Save the data file.
    data = data.astype('float32')
    data.tofile(outpath+'tempdata.dat')
    
    # Raster properties.
    xsize = int(data.shape[1])
    ysize = int(data.shape[0])
    lineoffset = xsize*4
    
    # Create the lat/lon .vrts.
    outfile = outpath+'templat.vrt'
    with open(outfile, "w") as hdr:
        hdr.write('<VRTDataset rasterXSize="'+str(xsize)+'" rasterYSize="'+str(ysize)+'">\n')
        hdr.write('  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">\n')
        hdr.write('    <SourceFilename relativetoVRT="1">templat.dat</SourceFilename>\n')
        hdr.write('    <ImageOffset>0</ImageOffset>\n')
        hdr.write('    <PixelOffset>4</PixelOffset>\n')
        hdr.write('    <LineOffset>'+str(lineoffset)+'</LineOffset>\n')
        hdr.write('    <ByteOrder>'+byte+'</ByteOrder>\n')
        hdr.write('  </VRTRasterBand>\n')
        hdr.write('</VRTDataset>\n')
        
    outfile = outpath+'templon.vrt'
    with open(outfile, "w") as hdr:
        hdr.write('<VRTDataset rasterXSize="'+str(xsize)+'" rasterYSize="'+str(ysize)+'">\n')
        hdr.write('  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">\n')
        hdr.write('    <SourceFilename relativetoVRT="1">templon.dat</SourceFilename>\n')
        hdr.write('    <ImageOffset>0</ImageOffset>\n')
        hdr.write('    <PixelOffset>4</PixelOffset>\n')
        hdr.write('    <LineOffset>'+str(lineoffset)+'</LineOffset>\n')
        hdr.write('    <ByteOrder>'+byte+'</ByteOrder>\n')
        hdr.write('  </VRTRasterBand>\n')
        hdr.write('</VRTDataset>\n')
        
        
    # Create the data file vrt.
    outfile = outpath+'tempdata.vrt'
    with open(outfile, "w") as hdr:
        hdr.write('<VRTDataset rasterXSize="'+str(xsize)+'" rasterYSize="'+str(ysize)+'">\n')
        hdr.write('  <Metadata domain="GEOLOCATION">\n')
        hdr.write('    <MDI key="LINE_OFFSET">0</MDI>\n')
        hdr.write('    <MDI key="LINE_STEP">1</MDI>\n')
        hdr.write('    <MDI key="PIXEL_OFFSET">0</MDI>\n')
        hdr.write('    <MDI key="PIXEL_STEP">1</MDI>\n')
        hdr.write('    <MDI key="X_BAND">1</MDI>\n')
        hdr.write('    <MDI key="X_DATASET">'+outpath+'templon.vrt</MDI>\n')
        hdr.write('    <MDI key="Y_BAND">1</MDI>\n')
        hdr.write('    <MDI key="Y_DATASET">'+outpath+'templat.vrt</MDI>\n')
        hdr.write('  </Metadata>\n')
        hdr.write('  <VRTRasterBand dataType="Float32" band="1" subClass="VRTRawRasterBand">\n')
        hdr.write('    <SourceFilename relativetoVRT="1">tempdata.dat</SourceFilename>\n')
        hdr.write('    <ImageOffset>0</ImageOffset>\n')
        hdr.write('    <PixelOffset>4</PixelOffset>\n')
        hdr.write('    <LineOffset>'+str(lineoffset)+'</LineOffset>\n')
        hdr.write('    <ByteOrder>'+byte+'</ByteOrder>\n')
        hdr.write('  </VRTRasterBand>\n')
        hdr.write('</VRTDataset>\n')
        
        
    # Call gdalwarp:
    command = 'gdalwarp -overwrite -geoloc -t_srs EPSG:4326 -ot Float32 -r ' + resampling + ' -of ' + outformat

    if nodataval is not None:
        command = command + ' -dstnodata '+str(nodataval)

    if tr is not None:
        if isinstance(tr, tuple):
            command = command + ' -tr '+str(tr[0])+' '+str(tr[1])
        else:
            command = command + ' -tr '+str(tr)+' '+str(tr)
        
    
    command = command + ' ' + outpath + 'tempdata.vrt ' + outpath + datafile
    
    
    print(command)
    print(subprocess.getoutput(command))
    
    
    # Remove temporary files.
    os.remove(outpath+'templat.dat')
    os.remove(outpath+'templat.vrt')
    os.remove(outpath+'templon.dat')
    os.remove(outpath+'templon.vrt')
    os.remove(outpath+'tempdata.dat')
    os.remove(outpath+'tempdata.vrt')
    
    return


def ll2radar(data, origin, spacing, lat, lon):
    """Convert an array in geographic (lat,lon) coordinates into a
    corresponding array in the (azimuth, slant range) coordinates of the
    radar image.
    
    Author: Michael Denbina
    
    Arguments:
        data (array): 2D array containing data in regularly spaced latitude/
            longitude coordinates.  First dimension of array should be
            latitude, second dimension should be longitude.
        origin (tuple): (Latitude, Longitude) of the first pixel in data,
            in degrees.
        spacing (tuple): (Latitude, Longitude) spacing of the data, in
            degrees.
        lat (array): 2D array containing the latitude values, in degrees,
            for each pixel of the radar image.
        lon (array): 2D array containing the longitude values, in degrees,
            for each pixel of the radar image.
            
    Returns:
        resdata (array): Data resampled to radar coordinates, with the same
            size as lat and lon.

    """               
    x = (lon - origin[1])/spacing[1]
    y = (lat - origin[0])/spacing[0]
    
    if np.any(np.imag(data) != 0): # if data is complex, interpolate real and imaginary parts separately
        resdata = 1j*bilinear_interpolate(np.imag(data),x,y)
        resdata += bilinear_interpolate(np.real(data),x,y)
    else: # data is real
        resdata = bilinear_interpolate(data,x,y)
        
    return resdata