# -*- coding: utf-8 -*-
"""Geocoding Module

    Use GDAL to make output products resampled to geographic projection with
    constant lat/lon spacing.
    
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
import os
import subprocess
import sys

import numpy as np

from kapok.lib import bilinear_interpolate


def radar2ll(outpath, datafile, data, lat, lon, outformat='ENVI',
             resampling='bilinear', nodataval=None, tr=None):
    """Create a geocoded file, in geographic projection, from input data
    in azimuth, slant range radar coordinates.
    
    Uses latitude and longitude arrays containing the geographic coordinates
    of each pixel (geolocation arrays), in order to perform the resampling
    using gdalwarp.  For gdalwarp reference, see
    http://www.gdal.org/gdalwarp.html.
    
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
            Default: None (GDAL will decide output file resolution based on
            input).
    
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
        hdr.write('  <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">\n')
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
        hdr.write('  <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">\n')
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
        hdr.write('  <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTRawRasterBand">\n')
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