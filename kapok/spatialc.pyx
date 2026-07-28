# -*- coding: utf-8 -*-
# cython: language_level=3
"""Spatial Filtering Cython Functions.

    This file contains Cython code for spatial filtering of the covariance
    matrix.  Imported by main kapok.spatial module.

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
import time
import os
import os.path

import numpy as np
import numpy.linalg as linalg
import h5py

cimport numpy as np
cimport cython
np.import_array()
np.import_ufunc()

import kapok



def nlmfilter(scene, outfile, nlmnoise, overwrite=False):
    """Perform non-local means filtering on the covariance matrix.
    
        The filtered covariance matrix is saved to a newly created
        Kapok HDF5 file, separate from the input scene.  This function
        returns the new Kapok Scene object for the filtered data.  This
        function is an interface to the non-local means filtering as
        implemented in the nlsartoolbox Python library, available at:
        https://www.math.u-bordeaux.fr/~cdeledal/nlsar.php
        
        The above library must be installed to use this function.
            
        Arguments:
            scene (Scene object): Input Kapok Scene object to be filtered.
            outfile (str): Path and filename for the new filtered Kapok
                HDF5 file.
            nlmnoise: The non-local means filter needs a sample
                of homogeneous data from which to estimate the noise.
                This argument can either be set to a four-element tuple
                containing the (azmin,azmax,rngmin,rngmax) bounds of
                a suitable noise subset, or alternatively, you can give
                an array containing the actual noise covariance matrix
                (which should be in the form of a 4D array, with axes:
                azimuth, range, row, column).
            overwrite (bool): Set to True to overwrite outfile, if it
                already exists.  Default: False.
                
        Returns:
            filtered (Scene object): Newly created Kapok Scene object
                with the filtered covariance matrix.  Note: Coherence
                optimization will need to be rerun for the filtered data.
                        
    """
    from kapok.lib import zerobelowdiag, makehermitian
       
    # Create the new HDF5 file.        
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.spatial.nlmfilter | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return    
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.spatial.nlmfilter | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return

    # Try to import the NL-SAR toolbox.
    try:
        import nlsartoolbox as nlsartb
    except ImportError:
        print('kapok.spatial.nlmfilter | Couldn\'t import nlsartoolbox.  Is it installed and working?  Aborting.')
        return None
        
    print('kapok.spatial.nlmfilter | Performing filtering of covariance matrix using non-local means filtering.')
    
    
    nlmnoise = np.array(nlmnoise)
    if nlmnoise is None:
        print('kapok.spatial.nlmfilter | \'nlmnoise\' keyword not specified.  Aborting.')
        return None
    elif (nlmnoise.ndim == 1): # Noise subset bounds specified.
        covnoise = scene.cov[nlmnoise[0]:nlmnoise[1],nlmnoise[2]:nlmnoise[3],:,:]
        covnoise = makehermitian(covnoise)
    else:
        covnoise = nlmnoise
        
    print('kapok.spatial.nlmfilter | Estimating noise statistics. ('+time.ctime()+')')
    stats = nlsartb.sarnlstats(covnoise, (scene.ml_window[0]*scene.ml_window[1])/1.44)

    print('kapok.spatial.nlmfilter | Starting non-local means filtering. ('+time.ctime()+')')
    cov = makehermitian(scene.cov[:])
    covfilt = nlsartb.sarnlsar(cov, stats)
    del cov
    
    nlsartb.sarnlstatsfree(stats)
    
    print('kapok.spatial.nlmfilter | Filtering complete. ('+time.ctime()+')')
    
    # Put filtered covariance matrix back into Kapok format (zeros below diagonal--will save space in the compressed HDF5 dataset).
    covfilt = zerobelowdiag(covfilt)
    
    # Save filtered covariance matrix to file.
    print('kapok.spatial.nlmfilter | Exporting filtered HDF5 scene file. ('+time.ctime()+')')
    cov_ds = f.create_dataset('cov', data=covfilt, dtype='complex64', compression=scene.compression, shuffle=True, compression_opts=scene.compression_opts)
    del covfilt
    f.flush()
    
    # Copy covariance matrix attributes.
    for key in scene.f['cov'].attrs:
        cov_ds.attrs[key] = scene.f['cov'].attrs[key]
    cov_ds.attrs['spatial_filter'] = True
    cov_ds.attrs['spatial_filter_method'] = 'nonlocalmeans'
    
    # Copy latitudes and attributes.
    lat = f.create_dataset('lat', data=scene.f['lat'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['lat'].attrs:
        lat.attrs[key] = scene.f['lat'].attrs[key]

    # Copy longitudes and attributes.
    lon = f.create_dataset('lon', data=scene.f['lon'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['lon'].attrs:
        lon.attrs[key] = scene.f['lon'].attrs[key]

    # Copy DEM heights and attributes.
    dem = f.create_dataset('dem', data=scene.f['dem'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['dem'].attrs:
        dem.attrs[key] = scene.f['dem'].attrs[key]

    # Copy incidence angles and attributes.
    inc = f.create_dataset('inc', data=scene.f['inc'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['inc'].attrs:
        inc.attrs[key] = scene.f['inc'].attrs[key]

    # Copy kz values and attributes.
    kz = f.create_dataset('kz', data=scene.f['kz'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['kz'].attrs:
        kz.attrs[key] = scene.f['kz'].attrs[key]

    # Copy main attributes.
    for key in scene.f.attrs:
        f.attrs[key] = scene.f.attrs[key]
    
    
    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.spatial.nlmfilter | Complete. ('+time.ctime()+')')
    return kapok.Scene(outfile)

   
def segfilter(scene, outfile, segfeatures=None, overwrite=False):
    """Perform segmentation-based filtering on the covariance matrix.
    
        The filtered covariance matrix is saved to a newly created
        Kapok HDF5 file, separate from the current scene!  The function
        returns the new Kapok Scene object for the filtered data.  This
        function uses the a segmentation algorithm implemented in the
        rsgislib Python library, available at: http://www.rsgislib.org/
        
        The above library must be installed to use this function.
            
        Arguments:
            scene (Scene object): Input Kapok Scene object to be filtered.
            outfile (str): Path and filename for the new filtered Kapok
                HDF5 file.
            segfeatures (array): Array of features to use for the
                segmentation.  Array should have dimensions (P, Az, Rng),
                where P is the number of features, Az is the azimuth size,
                and range is the range size.  This keyword is only required
                if you wish to specify custom features for the
                segmentation.  By default, this function will use a list
                of features derived from the PolInSAR data itself: the HH, HV,
                and VV backscatter of the reference track, as well as for each
                baseline the HH, HV, VV, HH-VV, and HH+VV coherence magnitudes,
                as well as the height separations between the different
                polarizations.
            overwrite (bool): Set to True to overwrite outfile, if it
                already exists.  Default: False.
                
        Returns:
            filtered (Scene object): Newly created Kapok Scene object
                with the filtered covariance matrix.  Note: Coherence
                optimization will need to be rerun for the filtered data.
                        
    """
    from osgeo import gdal

    # Create the new HDF5 file.        
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.spatial.segfilter | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return    
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.spatial.segfilter | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return

    # Try to import rsgislib.
    try:
        import rsgislib.segmentation.segutils
    except ImportError:
        print('kapok.spatial.segfilter | Couldn\'t import rsgislib.segmentation.segutils.  Is it installed and working?  Aborting.')
        return None
        
    tempdir = os.path.dirname(outfile) # Directory for intermediate files.
    if tempdir != '':
        tempdir += '/'
    else:
        tempdir = './'
    tempfile = tempdir + 'temp_segfeatures.kea'
    
    print('kapok.spatial.segfilter | Performing spatial filtering of covariance matrix using segmentation.')
    
    # If not provided, derive features for segmentation from
    # backscatter, coherence magntude, and phase center separations.
    if segfeatures is None:
        print('kapok.spatial.segfilter | Using backscatter and coherences as segmentation features. ('+time.ctime()+')')
        num_features = scene.num_baselines*9 + 3
        segfeatures = np.zeros((num_features,scene.dim[0],scene.dim[1]),dtype='float32')
        
        # Backscatter features:
        temp = scene.power('hh')
        temp[temp < 1e-10] = 1e-10
        segfeatures[0] = 10*np.log10(temp)
        
        temp = scene.power('hv')
        temp[temp < 1e-10] = 1e-10
        segfeatures[1] = 10*np.log10(temp)
        
        temp = scene.power('vv')
        temp[temp < 1e-10] = 1e-10
        segfeatures[2] = 10*np.log10(temp)
        
        
        for bl in range(scene.num_baselines):
            skip = bl*9
            
            kz = scene.kz(bl)
            
            segfeatures[3+skip] = np.abs(scene.coh('hh+vv',bl=bl))
            segfeatures[4+skip] = np.abs(scene.coh('hh-vv',bl=bl))
            segfeatures[5+skip] = np.abs(scene.coh('hh',bl=bl))
            segfeatures[6+skip] = np.abs(scene.coh('hv',bl=bl))
            segfeatures[7+skip] = np.abs(scene.coh('vv',bl=bl))
            
            refcoh = scene.coh('hh',bl=bl)
            segfeatures[8+skip] = np.angle(scene.coh('hh+vv',bl=bl)*np.conj(refcoh))/kz
            segfeatures[9+skip] = np.angle(scene.coh('hh-vv',bl=bl)*np.conj(refcoh))/kz
            segfeatures[10+skip] = np.angle(scene.coh('hv',bl=bl)*np.conj(refcoh))/kz
            segfeatures[11+skip] = np.angle(scene.coh('vv',bl=bl)*np.conj(refcoh))/kz
        
    
    gdalformat = 'Kea'
    driver = gdal.GetDriverByName(gdalformat)
    
    # If temp_segfeatures file already exists (e.g., from a previously aborted run), delete it.        
    if os.path.isfile(tempfile):
        os.remove(tempfile)       

    # Write features for segmentation to temporary file:            
    ds_features = driver.Create(tempfile, segfeatures.shape[2], segfeatures.shape[1], segfeatures.shape[0], gdal.GDT_Float32)
    for band in range(segfeatures.shape[0]):
        ds_features.GetRasterBand(band+1).WriteArray(segfeatures[band], 0, 0)
        
    del segfeatures
    del ds_features
        
    # Perform segmentation using rsgislib:
    print('kapok.spatial.segfilter | Running segmentation in rsgislib. ('+time.ctime()+')')
    clumpsfile = tempdir + 'temp_clumps.kea'
    rsgislib.segmentation.segutils.runShepherdSegmentation(tempfile, clumpsfile, numClusters=60, minPxls=10, distThres=1e6, tmpath=tempdir, gdalformat=gdalformat)
    
    
    # Do the covariance matrix averaging for each segment.
    ds_clumps = gdal.Open(clumpsfile, gdal.GA_ReadOnly)
    cdef np.ndarray[np.uint32_t, ndim=2] clumps = ds_clumps.ReadAsArray()
    del ds_clumps
    cdef unsigned long num_clumps = np.max(clumps)
    print('kapok.spatial.segfilter | Total number of segments: '+str(num_clumps)+'.')
    
    
    cov_ds = f.create_dataset('cov', scene.cov.shape, dtype='complex64', compression=scene.compression, shuffle=True, compression_opts=scene.compression_opts)
    
    cdef unsigned int row, col, n, k, num_rows, num_elements
    num_rows = scene.cov.shape[2]
    num_elements = (num_rows) * (num_rows + 1) // 2 # number of covariance matrix elements above and including main diagonal
    
    cdef np.ndarray[np.complex64_t, ndim=2] covtemp = np.zeros((scene.dim[0],scene.dim[1]),dtype='complex64')
    cdef np.ndarray[np.complex64_t, ndim=2] covfilt = np.zeros((scene.dim[0],scene.dim[1]),dtype='complex64')
    
    k = 0
    n = 0
    print('kapok.spatial.segfilter | Averaging covariance matrix for each segment. ('+time.ctime()+')')
    print('kapok.spatial.segfilter | Progress: '+str(np.round(((n/num_clumps/num_elements)+(k/num_elements))*100,decimals=2))+'%. ('+time.ctime()+')     ', end='\r')
    for row in range(0,num_rows):
        for col in range(row,num_rows):
            k += 1
            covtemp = scene.cov[:,:,row,col]
            
            ind = (clumps == 0)
            if np.any(ind):
                covfilt[ind] = covtemp[ind]
            
            for n in range(1,num_clumps):
                if (n % 1000) == 0:
                    print('kapok.spatial.segfilter | Progress: '+str(np.round(((n/num_clumps/num_elements)+(k/num_elements))*100,decimals=2))+'%. ('+time.ctime()+')     ', end='\r')
                
                ind = (clumps == n)
                if np.any(ind):
                    covfilt[ind] = np.nanmean(covtemp[ind])
                    
            cov_ds[:,:,row,col] = covfilt
    
    
    # Create HDF5 dataset to store clump information (in case we want to look at the segmentation results later).
    clumps_ds = f.create_dataset('clumps', data=clumps, compression=scene.compression, shuffle=True, compression_opts=scene.compression_opts)
    
    # Clean up temporary files.
    print('kapok.spatial.segfilter | Deleting temporary files. ('+time.ctime()+')               ')
    os.remove(tempfile)
    os.remove(clumpsfile)
    
    # Copy covariance matrix attributes.
    for key in scene.f['cov'].attrs:
        cov_ds.attrs[key] = scene.f['cov'].attrs[key]
    cov_ds.attrs['spatial_filter'] = True
    cov_ds.attrs['spatial_filter_method'] = 'segmentation'
    
    # Copy latitudes and attributes.
    lat = f.create_dataset('lat', data=scene.f['lat'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['lat'].attrs:
        lat.attrs[key] = scene.f['lat'].attrs[key]

    # Copy longitudes and attributes.
    lon = f.create_dataset('lon', data=scene.f['lon'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['lon'].attrs:
        lon.attrs[key] = scene.f['lon'].attrs[key]

    # Copy DEM heights and attributes.
    dem = f.create_dataset('dem', data=scene.f['dem'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['dem'].attrs:
        dem.attrs[key] = scene.f['dem'].attrs[key]

    # Copy incidence angles and attributes.
    inc = f.create_dataset('inc', data=scene.f['inc'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['inc'].attrs:
        inc.attrs[key] = scene.f['inc'].attrs[key]

    # Copy kz values and attributes.
    kz = f.create_dataset('kz', data=scene.f['kz'], dtype='float32', compression=scene.compression, compression_opts=scene.compression_opts)
    for key in scene.f['kz'].attrs:
        kz.attrs[key] = scene.f['kz'].attrs[key]

    # Copy main attributes.
    for key in scene.f.attrs:
        f.attrs[key] = scene.f.attrs[key]
    
    
    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.spatial.segfilter | Complete. ('+time.ctime()+')                                                  ')
    return kapok.Scene(outfile)