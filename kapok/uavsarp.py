# -*- coding: utf-8 -*-
"""UAVSAR Data Import Cython Functions

    Module for importing UAVSAR data.  This is Python code which the uavsar.py
    wrapper module defaults to when the Cython import fails.
    
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
import os.path
import time
from glob import glob

import numpy as np
import h5py
from scipy.ndimage.interpolation import zoom

import kapok
from kapok.lib import mlook, smooth, sch2enu


def load(infile, outfile, mlwin=(20,5), smwin=(1,1), azbounds=None,
         rngbounds=None, tracks=None, compression='gzip',
         compression_opts=4, num_blocks=20, kzcalc=False, kzvertical=True,
         overwrite=False):
    """Load a UAVSAR dataset into the Kapok format, and save it to target HDF5
    file.
    
    Arguments:
        infile (str): Path and filename of a UAVSAR .ann file containing the
            metadata for a UAVSAR SLC stack.
        outfile (str): Path and filename for the saved HDF5 dataset.
        mlwin: Tuple containing the multilooking window sizes in azimuth and
            slant range (in that order).  div=(12,3) results in multilooking
            using a window of 12 pixels in azimuth and 3 pixels in slant range.
            Within the window, all SLC pixels are averaged into a single
            covariance matrix, reducing the size of the data.  The size of the
            incidence angle, latitude, longitude, and other rasters are also
            reduced by the same factor.  Default: (20,5).
        smwin: Tuple containing boxcar moving average window sizes in azimuth
            and slant range (in that order).  Each covariance matrix will be
            smoothed using this window, without reducing the number of pixels
            in the data.  Smoothing is performed after multilooking, if both
            are specified.  Default: (1,1).
        azbounds: List containing starting and ending SLC azimuth index for
            desired subset.  Data outside these bounds will not be loaded.
            Default is to import all data.
        rngbounds: List containing starting and ending SLC range index for
            desired subset.  Data outside these bounds will not be loaded.
            Default is to import all data.
        tracks: List containing desired track indices to import.  For example,
            tracks=[0,1] will import only the first two tracks listed in the
            annotation file.  Default: All tracks imported.
        compression (str): Compression keyword used when creating the HDF5
            datasets to store the covariance matrix, incidence angle, kz,
            and DEM heights.  Default: 'gzip'.
        compression_opts (int): Compression option value used when creating
            the HDF5 datasets.  Number from 0 to 9, with 9 being maximum
            compression.  Default: 4.
        num_blocks (int): Number of blocks to use for splitting up the
            covariance matrix calculation.  Should be a positive integer
            >= 1.  Higher numbers use less memory, but will be slower.
            Default: 15.
        kzcalc (bool): Set to True if you wish to calculate the kz from the
            .baseline and .lkv files.  Set to False if you have UAVSAR .kz
            files you wish to import.  Note that even if this argument is
            set to False, if matching .kz files are not found, this function
            will print a warning and calculate the kz from the geometry.
            Note: The UAVSAR provided .kz files are corrected for the ground
            topography.  The kz values calculated by this script are not.
            During the model inversion procedure as called in
            kapok.Scene.inv(), the calculated kz will be corrected for the
            effects of the range-facing terrain slope angle (but not the
            azimuth-facing terrain slope angle, or the squint angle), if
            the rngslope keyword is supplied to that function.
            If the imported .kz files are used, no further correction is
            necessary, so kapok.Scene.inv() will use the kz values as is.
            Default: False.
        kzvertical (bool): If this flag is True, and kzcalc is False, the
            kz values imported from the UAVSAR stack files will be assumed to
            be kz values for measuring height perpendicular to the ground
            surface.  This flag will change the kz values to use the vertical
            direction.  This flag will have no effect if kzcalc is True, or
            if the .kz files cannot be found, as the kz values calculated by
            this function are always for the vertical direction.
            Default: True.
        overwrite (bool): Set to True to overwrite an existing Kapok HDF5
            file if necessary.  Default: False.
            
    Returns:
        scene: A Kapok scene object pointing to the newly created HDF5 file.
    
    """
    # Load the annotation file.
    try:
        ann = Ann(infile)
    except:
        print('kapok.uavsar.load | Cannot load UAVSAR annotation file. Aborting.')
        return
    
    
    # Create the HDF5 file.
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.uavsar.load | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return    
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.uavsar.load | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return
    
       
    
    # Get SLC dimensions and number of segments.
    rngsize_slc = ann.query('slc_1_1x1 Columns')
    num_segments = ann.query('Number of Segments')
    if num_segments > 1:
        azsize_slc = np.zeros(num_segments,dtype='int32')
        for n in range(num_segments):
            azsize_slc[n] = ann.query('slc_'+str(n+1)+'_1x1 Rows')
    else:
        azsize_slc = ann.query('slc_1_1x1 Rows')
    
    
    # Get track filenames and number of tracks.
    temp = ann.query('stackline1')
    num = 1
    if temp is not None:
        tracknames = [temp.split('_L090')[0]]
        num += 1
        temp = ann.query('stackline'+str(num))
        while temp is not None:
            tracknames.append(temp.split('_L090')[0])
            num += 1
            temp = ann.query('stackline'+str(num))
    else:
        print('kapok.uavsar.load | Cannot find track names in UAVSAR annotation file.  Aborting.')
        return
        

    # Subset track names if desired tracks were specified:
    tracknames = np.array(tracknames)
    if tracks is not None:
        tracks = np.array(tracks,dtype='int')
        tracknames = tracknames[tracks]
        
    num_tracks = len(tracknames) # Number of Tracks
    num_bl = int(num_tracks * (num_tracks-1) / 2) # Number of Baselines
    num_pol = int(3) # Number of Polarizations (HH, sqrt(2)*HV, VV)
    num_cov_elements = num_tracks*num_pol # Number of Covariance Matrix Elements in Each Row

    f.attrs['stack_name'] = ann.query('Stack Name')    
    f.attrs['site'] = ann.query('Site Description')
    f.attrs['url'] = ann.query('URL')
    f.attrs['sensor'] = 'UAVSAR'
    f.attrs['processor'] = 'production'
    
    print('kapok.uavsar.load | Stack ID: '+f.attrs['stack_name'])
    print('kapok.uavsar.load | Site Description: '+f.attrs['site'])
    print('kapok.uavsar.load | URL: '+f.attrs['url'])
    
    print('kapok.uavsar.load | Importing metadata. ('+time.ctime()+')')
    
    f.attrs['average_altitude'] = ann.query('Average Altitude')
    f.attrs['image_starting_slant_range'] = ann.query('Image Starting Slant Range')*1000
    f.attrs['slc_azimuth_pixel_spacing'] = ann.query('1x1 SLC Azimuth Pixel Spacing')
    f.attrs['slc_slant_range_pixel_spacing'] = ann.query('1x1 SLC Range Pixel Spacing')
    f.attrs['cov_azimuth_pixel_spacing'] = ann.query('1x1 SLC Azimuth Pixel Spacing') * mlwin[0]
    f.attrs['cov_slant_range_pixel_spacing'] = ann.query('1x1 SLC Range Pixel Spacing') * mlwin[1]
    
    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_baselines'] = num_bl
    f.attrs['tracks'] = np.array(tracknames,dtype='S')
    
    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts
    
    # Get SCH Peg from Annotation File
    peglat = ann.query('Peg Latitude')
    peglon = ann.query('Peg Longitude')
    peghdg = ann.query('Peg Heading')
    f.attrs['peg_latitude'] = peglat
    f.attrs['peg_longitude'] = peglon
    f.attrs['peg_heading'] = peghdg
    
    # Save in attributes as degrees, but convert to radians for coordinate transformation functions.
    peglat = np.radians(peglat)
    peglon = np.radians(peglon)
    peghdg = np.radians(peghdg)
    
    if (azbounds is not None) or (rngbounds is not None):
        f.attrs['subset'] = True
    else:
        f.attrs['subset'] = False
    
    
    # Check azimuth bounds for validity.
    if azbounds is None:
        azbounds = [0,np.sum(azsize_slc)]
        
    if azbounds[1] <= azbounds[0]:
        print('kapok.uavsar.load | Invalid azimuth bounds.  Must be ascending.  Aborting.')
        return
        
    if azbounds[0] < 0:
        print('kapok.uavsar.load | Lower azimuth bound ('+str(azbounds[0])+') is less than zero.  Setting lower azimuth bound to zero.')
        azbounds[0] = 0
        
    if azbounds[1] > np.sum(azsize_slc):
        print('kapok.uavsar.load | Upper azimuth bound ('+str(azbounds[1])+') greater than number of SLC lines ('+str(azsize_slc)+').')
        print('kapok.uavsar.load | Setting upper azimuth bound to '+str(azsize_slc)+'.')
        azbounds[1] = np.sum(azsize_slc)
        
        
    # Check range bounds for validity.
    if rngbounds is None:
        rngbounds = [0,rngsize_slc]
        
    if rngbounds[1] <= rngbounds[0]:
        print('kapok.uavsar.load | Invalid range bounds.  Must be ascending.  Aborting.')
        return
        
    if rngbounds[0] < 0:
        print('kapok.uavsar.load | Lower range bound ('+str(rngbounds[0])+') is less than zero.  Setting lower azimuth bound to zero.')
        rngbounds[0] = 0
        
    if rngbounds[1] > rngsize_slc:
        print('kapok.uavsar.load | Upper range bound ('+str(rngbounds[1])+') greater than number of SLC columns ('+str(rngsize_slc)+').')
        print('kapok.uavsar.load | Setting upper range bound to '+str(rngsize_slc)+'.')
        rngbounds[1] = rngsize_slc
    
    
    # Multi-looked, subsetted, image dimensions:
    azsize = (azbounds[1]-azbounds[0]) // mlwin[0]
    rngsize = (rngbounds[1]-rngbounds[0]) // mlwin[1]
    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['dim_slc'] = (int(np.sum(azsize_slc)),rngsize_slc)
    f.attrs['ml_window'] = mlwin
    f.attrs['sm_window'] = smwin
    f.attrs['azimuth_bounds_slc'] = azbounds
    f.attrs['range_bounds_slc'] = rngbounds

    # Path containing SLCs and other files (assume in same folder as .ann):
    datapath = os.path.dirname(infile)
    if datapath != '':
        datapath = datapath + '/'
    
    # Get filenames of SLCs for each track, in polarization order HH, HV, VV.
    slcfiles = []
    for seg in range(num_segments):
        for tr in range(num_tracks):
            for pol in ['HH','HV','VV']:
                file = glob(datapath+tracknames[tr]+'*'+pol+'_*_s'+str(seg+1)+'_1x1.slc')
                
                if len(file) == 1:            
                    slcfiles.append(file[0])
                elif len(file) > 1:
                    print('kapok.uavsar.load | Too many SLC files matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_1x1.slc'+'".  Aborting.')
                    return
                else:
                    print('kapok.uavsar.load | Cannot find SLC file matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_1x1.slc'+'".  Aborting.')
                    return
    
    
    # Initialize covariance matrix dataset in HDF5 file:
    cov = f.create_dataset('cov', (azsize, rngsize, num_cov_elements, num_cov_elements), dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)
    
    
    # Calculate covariance matrix.
    az_vector = np.round(np.linspace(0, azsize, num=num_blocks+1)).astype('int')
    az_vector[num_blocks] = azsize
    
    for n, azstart in enumerate(az_vector[0:-1]):
        azend = az_vector[n+1]
        azstart_slc = azstart*mlwin[0] + azbounds[0]
        azend_slc = azend*mlwin[0] + azbounds[0]
        seg_start, azoffset_start = findsegment(azstart_slc, azsize_slc)
        seg_end, azoffset_end = findsegment(azend_slc, azsize_slc)
        
        print('kapok.uavsar.load | Calculating covariance matrix for rows '+str(azstart)+'-'+str(azend-1)+' / '+str(azsize-1)+' ('+str(np.round(azstart/azsize*100))+'%). ('+time.ctime()+')')
        
        for slcnum in range(0,num_cov_elements):
            print('kapok.uavsar.load | Loading SLCs: '+str(slcnum+1)+'/'+str(num_cov_elements)+'. ('+time.ctime()+')     ', end='\r')
            file = slcfiles[slcnum+(seg_start*num_cov_elements)]
            
            if seg_start == seg_end:
                slc = getslcblock(file, rngsize_slc, azoffset_start, azoffset_end, rngbounds=rngbounds)
            else:
                file2 = slcfiles[slcnum+(seg_end*num_cov_elements)]
                slc = getslcblock(file, rngsize_slc, azoffset_start, azoffset_end, rngbounds=rngbounds, file2=file2, azsize=azsize_slc[seg_start])
                
            slc[np.abs(slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.
                
            if slcnum == 0:
                slcstack = np.zeros((num_cov_elements,slc.shape[0],slc.shape[1]),dtype='complex64')
                
                if (slcnum % 3) == 1: # HV Polarization
                    slcstack[slcnum] = np.sqrt(2)*slc
                else: # HH or VV Polarization
                    slcstack[slcnum] = slc
            else:
                if (slcnum % 3) == 1: # HV Polarization
                    slcstack[slcnum] = np.sqrt(2)*slc
                else:  # HH or VV Polarization
                    slcstack[slcnum] = slc
            
            
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                print('kapok.uavsar.load | Calculating matrix element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')     ', end='\r')
                cov[azstart:azend,:,row,col] = mlook(slcstack[row]*np.conj(slcstack[col]),mlwin)
    
    
    del slcstack
    f.flush()
    
    # Boxcar Smoothing:
    if smwin != (1,1):
        print('kapok.uavsar.load | Boxcar averaging covariance matrix. ('+time.ctime()+')')
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                cov[:,:,row,col] = smooth(cov[:,:,row,col],smwin)
    
    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*HV', 'VV'],dtype='S')
    print('kapok.uavsar.load | Covariance matrix calculation complete. ('+time.ctime()+')     ')
    
    
    # Load LLH files.
    llh = None
    mlwin_lkv = (int(ann.query('Number of Azimuth Looks in 2x8 SLC')), int(ann.query('Number of Range Looks in 2x8 SLC')))
    
    for seg in range(num_segments):
        llhname = ann.query('llh_'+str(seg+1)+'_2x8')
        file = glob(datapath+llhname)
        
        if len(file) == 1:
            llh_rows = int(ann.query('llh_'+str(seg+1)+'_2x8 Rows'))
            llh_cols = int(ann.query('llh_'+str(seg+1)+'_2x8 Columns'))
            if llh is None:
                llh = np.memmap(file[0], dtype='float32', mode='r', shape=(llh_rows, llh_cols, 3))
            else:
                llh = np.vstack((llh, np.memmap(file[0], dtype='float32', mode='r', shape=(llh_rows, llh_cols, 3))))
        elif len(file) > 1:
            print('kapok.uavsar.load | Too many LLH files matching pattern: "'+datapath+llhname+'".  Aborting.')
            return
        else:
            print('kapok.uavsar.load | Cannot find LLH file matching pattern: "'+datapath+llhname+'".  Aborting.')
            return
    

    # LLH Subset Bounds and Offsets for Trimming Multilooked Arrays
    azllhstart = azbounds[0] // mlwin_lkv[0]
    azllhend = azbounds[1] // mlwin_lkv[0]
    azllhoffset = azbounds[0] % mlwin_lkv[0]
    if (azllhend < llh.shape[0]):
        azllhend += 1
    
    rngllhstart = rngbounds[0] // mlwin_lkv[1]
    rngllhend = rngbounds[1] // mlwin_lkv[1]
    rngllhoffset = rngbounds[0] % mlwin_lkv[1]
    if (rngllhend < llh.shape[1]):
        rngllhend += 1

    # Initialize latitude dataset and import values:   
    print('kapok.uavsar.load | Importing latitude values. ('+time.ctime()+')')    
    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)   
    lat[:] = mlook(zoom(llh[azllhstart:azllhend,rngllhstart:rngllhend,0],mlwin_lkv),mlwin)[azllhoffset:(azllhoffset+azsize),rngllhoffset:(rngllhoffset+rngsize)]
    lat.attrs['units'] = 'degrees'
    lat.attrs['description'] = 'Latitude'
    
    # Initialize longitude dataset and import values:
    print('kapok.uavsar.load | Importing longitude values. ('+time.ctime()+')')
    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lon[:] = mlook(zoom(llh[azllhstart:azllhend,rngllhstart:rngllhend,1],mlwin_lkv),mlwin)[azllhoffset:(azllhoffset+azsize),rngllhoffset:(rngllhoffset+rngsize)]
    lon.attrs['units'] = 'degrees'
    lon.attrs['description'] = 'Longitude'

    # Initialize DEM height dataset and import values:
    print('kapok.uavsar.load | Importing DEM heights. ('+time.ctime()+')')
    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = mlook(zoom(llh[azllhstart:azllhend,rngllhstart:rngllhend,2],mlwin_lkv),mlwin)[azllhoffset:(azllhoffset+azsize),rngllhoffset:(rngllhoffset+rngsize)]
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Processor DEM'
    
    # Convert LLH to ENU coordinates.  Used to calculate platform position later. 
    posmm = llh2enu(np.radians(llh[:,:,1]), np.radians(llh[:,:,0]), llh[:,:,2], peglat, peglon, peghdg)
    
    f.flush()
    del llh
    
    
    # Load LKV files.
    lkvmm = None
    for seg in range(num_segments):
        lkvname = ann.query('lkv_'+str(seg+1)+'_2x8')
        file = glob(datapath+lkvname)
        
        if len(file) == 1:
            lkv_rows = int(ann.query('lkv_'+str(seg+1)+'_2x8 Rows'))
            lkv_cols = int(ann.query('lkv_'+str(seg+1)+'_2x8 Columns'))
            if lkvmm is None:
                lkvmm = np.memmap(file[0], dtype='float32', mode='r', shape=(lkv_rows, lkv_cols, 3))
            else:
                lkvmm = np.vstack((lkvmm, np.memmap(file[0], dtype='float32', mode='r', shape=(lkv_rows, lkv_cols, 3))))
        elif len(file) > 1:
            print('kapok.uavsar.load | Too many LKV files matching pattern: "'+datapath+lkvname+'".  Aborting.')
            return
        else:
            print('kapok.uavsar.load | Cannot find LKV file matching pattern: "'+datapath+lkvname+'".  Aborting.')
            return           
            
    
    # Get sensor wavelength from annotation file.
    wavelength = ann.query('Center Wavelength')
    wavelength /= 100 # convert from cm to m
    f.attrs['wavelength'] = wavelength
       
    # Starting S coordinate of SLCs, and S coordinate spacing (in meters).
    sstart = ann.query('slc_1_1x1_mag.row_addr')
    sspacing = ann.query('slc_1_1x1_mag.row_mult')

    # S Coordinate Bounds for Desired Subset:
    sbounds = azbounds*sspacing + sstart
    
    # Calculate platform position from LLH and LKV:
    posmm = posmm - lkvmm
    
    
    # Incidence Angle Calculation for Reference Track
    print('kapok.uavsar.load | Calculating incidence angle values from reference track look vector. ('+time.ctime()+')')
    refvelocity = np.gradient(posmm, axis=0)
    refvelocity = refvelocity / (np.linalg.norm(refvelocity,axis=2)[:,:,np.newaxis])
    
    proj_reflkv_v = np.sum(lkvmm*refvelocity,axis=2)[:,:,np.newaxis]*refvelocity
    
    incident_lkv = lkvmm - proj_reflkv_v
    inc_ref = np.arccos(np.abs(incident_lkv[:,:,2])/np.linalg.norm(incident_lkv,axis=2))
    
    # Initialize incidence angle dataset (stores reference track incidence angle):
    inc = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Reference Track Incidence Angle'
    inc[:] = mlook(zoom(inc_ref[azllhstart:azllhend,rngllhstart:rngllhend],mlwin_lkv),mlwin)[azllhoffset:(azllhoffset+azsize),rngllhoffset:(rngllhoffset+rngsize)]
    del refvelocity, proj_reflkv_v, incident_lkv, inc_ref    
    
    
    
    # Initialize kz dataset:
    kz = f.create_dataset('kz', (num_tracks, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    kz.attrs['indexing'] = 'track'
    
    
    # Static Typed Variables (for kz calculation)
    lkv = np.array(lkvmm,dtype='float64')
    del lkvmm
    
    pos = np.array(posmm,dtype='float64')
    del posmm
    
    inc_buffer = np.zeros((mlwin[0],rngsize_slc),dtype='float64')
    kz_buffer = np.zeros((mlwin[0],rngsize_slc),dtype='float64')
    
    tracklkv = np.zeros((mlwin[0],rngsize_slc,3),dtype='float64')
    velocity = np.zeros((mlwin[0],rngsize_slc,3),dtype='float64')
    lookcrossv = np.zeros((mlwin[0],rngsize_slc,3),dtype='float64')
    tempdiff = np.zeros((mlwin[0],rngsize_slc,3),dtype='float64')
    proj_lkv_velocity = np.zeros((mlwin[0],rngsize_slc,3),dtype='float64')

    base_enu = np.zeros((azbounds[1]-azbounds[0],3),dtype='float64')

    baseline = np.zeros((mlwin[0],3),dtype='float64')    
    baselinep = np.zeros((mlwin[0],rngsize_slc),dtype='float64')
    baselinepsign = np.zeros((mlwin[0],rngsize_slc),dtype='float64')
    
    
    # Try to import .kz files if option kzcalc == False.
    if not kzcalc:
        kz.attrs['slope_corrected'] = True
        for tr in range(0,num_tracks):
            kz_temp = None
            print('kapok.uavsar.load | Importing kz between track '+str(tr)+' and reference track. ('+time.ctime()+')')
            for seg in range(num_segments):
                if not kzcalc:
                    file = glob(datapath+tracknames[tr]+'*s'+str(seg+1)+'*.kz')
                    file_alt = glob(datapath+tracknames[tr]+'*.kz')
                    
                    if len(file) >= 1:
                        kz_rows = int(ann.query('lkv_'+str(seg+1)+'_2x8 Rows'))
                        kz_cols = int(ann.query('lkv_'+str(seg+1)+'_2x8 Columns'))
                        if kz_temp is None:
                            kz_temp = np.memmap(file[0], dtype='float32', mode='r', shape=(kz_rows, kz_cols))
                        else:
                            kz_temp = np.vstack((kz_temp, np.memmap(file[0], dtype='float32', mode='r', shape=(kz_rows, kz_cols))))
                    elif (num_segments == 1) and (len(file_alt) >= 1):
                        kz_rows = int(ann.query('lkv_'+str(seg+1)+'_2x8 Rows'))
                        kz_cols = int(ann.query('lkv_'+str(seg+1)+'_2x8 Columns'))
                        if kz_temp is None:
                            kz_temp = np.memmap(file_alt[0], dtype='float32', mode='r', shape=(kz_rows, kz_cols))
                        else:
                            kz_temp = np.vstack((kz_temp, np.memmap(file_alt[0], dtype='float32', mode='r', shape=(kz_rows, kz_cols))))
                    else:
                        print('kapok.uavsar.load | Cannot find .kz file matching pattern: "'+datapath+tracknames[tr]+'*s'+str(seg+1)+'*.kz".  Attempting to calculate kz from .baseline files.')
                        kzcalc = True
                        break
            
            if not kzcalc:
                # kz files appear to be kz_i0 rather than kz_0i.  Multiplying by -1 so that in the Scene object, kz_ij = kz[j] - kz[i].
                kz[tr,:,:] = -1*mlook(zoom(kz_temp[azllhstart:azllhend,rngllhstart:rngllhend],mlwin_lkv),mlwin)[azllhoffset:(azllhoffset+azsize),rngllhoffset:(rngllhoffset+rngsize)]
        
        
        if (not kzcalc) and kzvertical:
            print('kapok.uavsar.load | Adjusting kz to use vertical direction. ('+time.ctime()+')')
            # Change kz values to measure vertical heights, rather than distances perpendicular to the ground surface.
            from kapok.lib import calcslope
            
            # Calculate range and azimuth terrain slope angles.
            spacing = (f.attrs['cov_azimuth_pixel_spacing'], f.attrs['cov_slant_range_pixel_spacing'])
            rngslope, azslope = calcslope(dem, spacing, inc)
            
            # Compute dot product of vertical and surface normal.
            height_dot_normal = 1 / np.sqrt(np.square(np.tan(rngslope)) + np.square(np.tan(azslope)) + 1)
            
            # Normalize kz values by dot product.
            kz[:] *= height_dot_normal[np.newaxis,:,:]
            del height_dot_normal, rngslope, azslope
    
    
    # Calculating kz from .baseline and .lkv, if kzcalc == True.
    if kzcalc:
        kz.attrs['slope_corrected'] = False
        
        for tr in range(0,num_tracks):
            print('kapok.uavsar.load | Calculating kz between track '+str(tr)+' and reference track. ('+time.ctime()+')')
            basefile = glob(datapath+tracknames[tr]+'*.baseline')
            
            if len(basefile) >= 1:
                basesch = np.loadtxt(basefile[0])
            else:
                print('kapok.uavsar.load | Cannot find .baseline file matching pattern: "'+datapath+tracknames[tr]+'*.baseline'+'".  Aborting.')
                return
                
            istart = np.where(np.isclose(basesch[:,0],sbounds[0]))[0][0] # index of first azimuth line in subset
            iend = np.where(np.isclose(basesch[:,0],sbounds[1]))[0][0] # index of last azimuth line in subset
            base_enu[:] = sch2enu(basesch[istart:iend,1],basesch[istart:iend,2],basesch[istart:iend,3], peglat, peglon, peghdg)[0:base_enu.shape[0],:]
            
            
            # Main kz calculation loop.                
            for az_ml in range(0,azsize):
                # Calculate indices.
                azstart = az_ml*mlwin[0]
                azend = azstart + mlwin[0]
                lkvrowstart = (azstart + azbounds[0]) // mlwin_lkv[0]
                lkvrowend = (azend + azbounds[0]) // mlwin_lkv[0]
                if lkvrowend < np.shape(lkv)[0]:
                    lkvrowend += 1
                lkvtrimstart = (azstart + azbounds[0]) % mlwin_lkv[0]
                lkvtrimend = lkvtrimstart + mlwin[0]
                
                # Current track look vector.
                tracklkv[:,:,0] = (zoom(lkv[lkvrowstart:lkvrowend,:,0],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,0])[0:mlwin[0],0:rngsize_slc]
                tracklkv[:,:,1] = (zoom(lkv[lkvrowstart:lkvrowend,:,1],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,1])[0:mlwin[0],0:rngsize_slc]
                tracklkv[:,:,2] = (zoom(lkv[lkvrowstart:lkvrowend,:,2],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,2])[0:mlwin[0],0:rngsize_slc]
                
                # Platform velocity vector.
                velocity[:,:,0] = np.gradient(zoom(pos[lkvrowstart:lkvrowend,:,0],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,0], axis=0)[0:mlwin[0],0:rngsize_slc]          
                velocity[:,:,1] = np.gradient(zoom(pos[lkvrowstart:lkvrowend,:,1],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,1], axis=0)[0:mlwin[0],0:rngsize_slc]          
                velocity[:,:,2] = np.gradient(zoom(pos[lkvrowstart:lkvrowend,:,2],mlwin_lkv)[lkvtrimstart:lkvtrimend] + base_enu[azstart:azend,np.newaxis,2], axis=0)[0:mlwin[0],0:rngsize_slc]          
                velocity = velocity / (np.linalg.norm(velocity,axis=2)[:,:,np.newaxis])
                
                # Project look vector onto velocity vector.
                proj_lkv_velocity = np.sum(tracklkv*velocity,axis=2)[:,:,np.newaxis]*velocity
                
                # Get incidence angle with same dimensions as single-look kz buffer.
                inc_buffer[:,0:rngsize*mlwin[1]] = zoom(inc[az_ml,:][np.newaxis,:],mlwin)
                inc_buffer[:,rngsize*mlwin[1]:] = inc_buffer[:,(rngsize*mlwin[1]-1)][:,np.newaxis]
                
                # Component of look vector orthogonal to velocity.
                lookcrossv[:,:,0] = tracklkv[:,:,1]*velocity[:,:,2] - tracklkv[:,:,2]*velocity[:,:,1]
                lookcrossv[:,:,1] = tracklkv[:,:,2]*velocity[:,:,0] - tracklkv[:,:,0]*velocity[:,:,2]
                lookcrossv[:,:,2] = tracklkv[:,:,0]*velocity[:,:,1] - tracklkv[:,:,1]*velocity[:,:,0]
                
                # Baseline and perpendicular baseline.
                baseline = base_enu[azstart:azend]
                baselinepsign = np.sum(baseline[:,np.newaxis,:]*lookcrossv,axis=2)/np.sum(lookcrossv*lookcrossv,axis=2)
                baselinep = np.linalg.norm((np.sum(baseline[:,np.newaxis,:]*lookcrossv,axis=2)/np.sum(lookcrossv*lookcrossv,axis=2))[:,:,np.newaxis]*lookcrossv,axis=2)
                
                # Calculate kz.
                kz_buffer = (4*np.pi/wavelength) * baselinep / (np.linalg.norm(tracklkv,axis=2)*np.sin(inc_buffer)) * np.sign(baselinepsign)
                
                kz[tr,az_ml,:] = mlook(kz_buffer[:,rngbounds[0]:rngbounds[1]], mlwin)
    
    
    
    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.uavsar.load | Complete. ('+time.ctime()+')     ')
    return kapok.Scene(outfile)
    
    
class Ann(object):
    """Class for loading and interacting with a UAVSAR annotation file."""
    
    def __init__(self, file):
        """Load in the specified .ann file as a list of strings and
            initialize.
        
        Arguments:
            file (str): UAVSAR annotation filename to load.
            
        """
        self.file = file
        fd = open(self.file, 'r')
        self.ann = fd.read().split('\n')
        return
        
    
    def query(self, keyword):
        """Query the annotation file for the specified annotation keyword. 
        
        Arguments:
            keyword (str): The keyword to query.
            
        Returns:
            value: The value of the specified keyword.
            
        """
        for n in range(len(self.ann)):
            if self.ann[n].startswith(keyword):
                try:
                    val = self.ann[n].rsplit('=')[-1].split(';')[0].split()[0]
                    val = np.array(val,dtype='float') # if we can convert the string to a number, do so
                    if (val - np.floor(val)) == 0:
                        val = np.array(val,dtype='int')  # if it's an integer, convert it to one (e.g., number of samples)
                    return val
                except ValueError: # if we can't convert the string to a number, leave it as a string
                    val = self.ann[n].split('=',maxsplit=1)[-1].split(';')[0].strip()
                    return val
                    
        return None
        
               
def findsegment(az,azsize):
    """For a given azimuth index, return the segment number and
    the azimuth index within the given segment.
    
    Arguments:
        az: Azimuth index of interest.
        azsize: List containing the azimuth size of each segment.
        
    Returns:
        seg: Segment number.
        azoff: Azimuth index within the segment.
        
    """
    azstart = np.insert(np.cumsum(azsize), 0, 0)[0:-1]
    
    if az <= np.sum(azsize):
        seg = np.max(np.where(azstart <= az))
        azoff = az - azstart[seg]
    else:
        seg = azstart.shape[0] - 1
        azoff = azsize[seg]
        print('kapok.uavsar.fingsegment | SLC row index of '+str(az)+' is larger than the size of the data.  Returning maximum index.')
        
    return seg, azoff
    
    
def getslcblock(file, rngsize, azstart, azend, rngbounds=None, file2=None,
                azsize=None):
    """Load SLC data into a NumPy array buffer.  If the file2 argument is
    specified in the arguments, this function will treat the two files as
    consecutive segments, and will join them.
    
    Arguments:
        file (str): Filename of the first SLC.
        rngsize (int): Number of columns (range bins).  Same for both SLCs.
        azstart (int): Azimuth index at which to start the buffer, in the
            first SLC.
        azend (int): Azimuth index at which to end the buffer.  If file2
            is specified, this is an azimuth index in the second SLC.  If
            file2 is not specified, this is an azimuth index in the first SLC.
            The row specified by azend is not actually included in the buffer,
            as in the Python range() function.  (azend-1) is the last line
            included in the buffer.  To load the entire SLC, azend should be
            equal to the number of rows in SLC.
        rngbounds (tuple, int): Starting and ending range bounds, if range
            subsetting is desired.
        file2 (str): Filename of the second SLC, if combining multiple
            segments is desired.
        azsize (int): Number of rows (azimuth bins) for the first SLC.  Only
            required if file2 is specified.  Otherwise we only load in the
            lines of the SLC before azend.
        
    Returns:
        block: NumPy array of complex64 datatype, containing the loaded SLC
        data between the specified azimuth bounds.
    
    """
    if file2 is None:
        byteoffset = rngsize * 8 * azstart
        slc = np.memmap(file, dtype='complex64', mode='c', offset=byteoffset, shape=(azend-azstart,rngsize))
        if rngbounds is not None:
            slc = slc[:,rngbounds[0]:rngbounds[1]]
        return slc
    elif azsize is None:
        print('kapok.uavsar.getslcblock: "file2" argument specified, but "azsize" argument missing. Aborting.')
        return
    else:
        byteoffset = rngsize * 8 * azstart
        slca = np.memmap(file, dtype='complex64', mode='c', offset=byteoffset, shape=(azsize-azstart,rngsize))
        slcb = np.memmap(file2, dtype='complex64', mode='c', shape=(azend,rngsize))
        if rngbounds is not None:
            slca = slca[:,rngbounds[0]:rngbounds[1]]
            slcb = slcb[:,rngbounds[0]:rngbounds[1]]
        return np.vstack((slca,slcb))


def quicklook(infile, tr=0, pol='hh', mlwin=(40,10), savefile=None):
    """Display a quick look intensity image for a given UAVSAR SLC stack.
    
    Arguments:
        infile (str): Input annotation file of the UAVSAR stack.
        tr (int): Track index of the desired image.  Default: 0
            (first track in .ann file).
        pol (str): Polarization str of the desired image.  Options are 'hh',
            'hv', or 'vv'.  Default: 'hh'
        mlwin (tuple): Multi-looking window size to use for the quick look
            image.  Note:  The original SLC azimuth indices will be displayed
            on the axes of the image, so that the image can be used as a guide
            for suitable values of the azbounds and rngbounds keywords in
            kapok.uavsar.load.  These multi-looking windows are in terms
            of the original 1x1 SLC image size, not the 8x2 or 4x1
            image sizes.
        savefile (str): Output path and filename to save the displayed image.
    
    """
    import matplotlib.pyplot as plt
    
    # Load the annotation file.
    try:
        ann = Ann(infile)
    except:
        print('kapok.uavsar.quicklook | Cannot load UAVSAR annotation file. Aborting.')
        return
    
    
    # Get track filenames and number of tracks.
    temp = ann.query('stackline1')
    num = 1
    if temp is not None:
        tracknames = [temp.split('_L090')[0]]
        num += 1
        temp = ann.query('stackline'+str(num))
        while temp is not None:
            tracknames.append(temp.split('_L090')[0])
            num += 1
            temp = ann.query('stackline'+str(num))
    else:
        print('kapok.uavsar.quicklook | Cannot find track names in UAVSAR annotation file.  Aborting.')
        return
    
    
    # Path containing SLCs and other files (assume in same folder as .ann):
    datapath = os.path.dirname(infile)
    if datapath != '':
        datapath = datapath + '/'
        
    pol = pol.upper()
    
    # Get filenames and dimensions of SLCs for each segment.
    num_segments = ann.query('Number of Segments')
    
    slcfiles = []
    for seg in range(num_segments):
        file = glob(datapath+tracknames[tr]+'*'+pol+'_*_s'+str(seg+1)+'_2x8.slc')
        
        if len(file) == 1:
            slcfiles.append(file[0])
            slcwindow = (8,2)
            rngsize_slc = ann.query('slc_1_2x8 Columns')
            if num_segments > 1:
                azsize_slc = np.zeros(num_segments,dtype='int32')
                for n in range(num_segments):
                    azsize_slc[n] = ann.query('slc_'+str(n+1)+'_2x8 Rows')
            else:
                azsize_slc = ann.query('slc_1_8x2 Rows')
        elif len(file) > 1:
            print('kapok.uavsar.quicklook | Too many SLC files matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_2x8.slc'+'".  Aborting.')
            return
        else:
            file = glob(datapath+tracknames[tr]+'*'+pol+'_*_s'+str(seg+1)+'_4x1.slc')
            
            if len(file) == 1:
                slcfiles.append(file[0])
                slcwindow = (4,1)
                rngsize_slc = ann.query('slc_1_1x4 Columns')
                if num_segments > 1:
                    azsize_slc = np.zeros(num_segments,dtype='int32')
                    for n in range(num_segments):
                        azsize_slc[n] = ann.query('slc_'+str(n+1)+'_1x4 Rows')
                else:
                    azsize_slc = ann.query('slc_1_1x4 Rows')
            elif len(file) > 1:
                print('kapok.uavsar.quicklook | Too many SLC files matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_1x4.slc'+'".  Aborting.')
                return
            else:
                file = glob(datapath+tracknames[tr]+'*'+pol+'_*_s'+str(seg+1)+'_1x1.slc')
                
                if len(file) == 1:
                    slcfiles.append(file[0])
                    slcwindow = (1,1)
                    rngsize_slc = ann.query('slc_1_1x1 Columns')
                    if num_segments > 1:
                        azsize_slc = np.zeros(num_segments,dtype='int32')
                        for n in range(num_segments):
                            azsize_slc[n] = ann.query('slc_'+str(n+1)+'_1x1 Rows')
                    else:
                        azsize_slc = ann.query('slc_1_1x1 Rows')
                elif len(file) > 1:
                    print('kapok.uavsar.quicklook | Too many SLC files matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_1x1.slc'+'".  Aborting.')
                    return
                else:
                    print('kapok.uavsar.quicklook | Cannot find SLC file matching pattern: "'+datapath+tracknames[tr]+'*_'+pol+'_*_1x1.slc'+'".  Aborting.')
                    return
    
    
    mlwin = (mlwin[0]//slcwindow[0],mlwin[1]//slcwindow[1])
    azsize = np.sum(azsize_slc) // mlwin[0]
    rngsize = rngsize_slc // mlwin[1]

    # Get SLC dimensions for the 1x1 SLCs -- these get displayed on the plot axes.
    rngsize_slc1x1 = ann.query('slc_1_1x1 Columns')
    if num_segments > 1:
        azsize_slc1x1 = np.zeros(num_segments,dtype='int32')
        for n in range(num_segments):
            azsize_slc1x1[n] = ann.query('slc_'+str(n+1)+'_1x1 Rows')
    else:
        azsize_slc1x1 = ann.query('slc_1_1x1 Rows')

   
    # Load and multilook quicklook intensity image.
    qlimage = np.zeros((azsize,rngsize),dtype='float32')
    
    num_blocks = int(np.sum(azsize_slc)*rngsize_slc*8/1e9)
    if num_blocks < (num_segments + 1):
        num_blocks = num_segments + 1
    az_vector = np.round(np.linspace(0, azsize, num=num_blocks+1)).astype('int')
    az_vector[num_blocks] = azsize
    
    for n, azstart in enumerate(az_vector[0:-1]):
        azend = az_vector[n+1]
        azstart_slc = azstart*mlwin[0]
        azend_slc = azend*mlwin[0]
        seg_start, azoffset_start = findsegment(azstart_slc, azsize_slc)
        seg_end, azoffset_end = findsegment(azend_slc, azsize_slc)
        
        file = slcfiles[seg_start]
        
        if seg_start == seg_end:
            slc = getslcblock(file, rngsize_slc, azoffset_start, azoffset_end)
        else:
            file2 = slcfiles[seg_end]
            slc = getslcblock(file, rngsize_slc, azoffset_start, azoffset_end, file2=file2, azsize=azsize_slc[seg_start])
        
        qlimage[azstart:azend,:] = mlook(np.real(slc*np.conj(slc)),mlwin)
    
    
    plt.figure()        
    
    qlimage = np.real(qlimage)
    qlimage[qlimage <= 1e-10] = 1e-10
    qlimage = 10*np.log10(qlimage)
    
    plt.imshow(qlimage, vmin=-25, vmax=0, cmap='gray', aspect=0.25, interpolation='nearest', extent=(0,rngsize_slc1x1,np.sum(azsize_slc1x1),0))
    plt.colorbar(label=pol+' Backscatter (dB)')
    plt.xlabel('Range Index')
    plt.ylabel('Azimuth Index')
    plt.tight_layout()

    if savefile is not None:
        plt.savefig(savefile, dpi=125, bbox_inches='tight', pad_inches=0.1)