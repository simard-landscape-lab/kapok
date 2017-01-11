# -*- coding: utf-8 -*-
# cython: language_level=3
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
         compression_opts=4, num_blocks=20, overwrite=False):
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
                print('kapok.uavsar.load | Calculating element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')     ', end='\r')
                cov[azstart:azend,:,row,col] = mlook(slcstack[row]*np.conj(slcstack[col]),mlwin)
                
    del slcstack
    
    
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
    print('kapok.uavsar.load | Covariance matrix calculation completed. ('+time.ctime()+')     ')
    
    
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
    
    
    # Initialize latitude dataset and import values:
    print('kapok.uavsar.load | Importing latitude values. ('+time.ctime()+')')
    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lat[:] = mlook(zoom(llh[:,:,0],mlwin_lkv)[azbounds[0]:azbounds[1],rngbounds[0]:rngbounds[1]],mlwin)
    lat.attrs['units'] = 'degrees'
    lat.attrs['description'] = 'Latitude'
    
    # Initialize longitude dataset and import values:
    print('kapok.uavsar.load | Importing longitude values. ('+time.ctime()+')')
    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lon[:] = mlook(zoom(llh[:,:,1],mlwin_lkv)[azbounds[0]:azbounds[1],rngbounds[0]:rngbounds[1]],mlwin)
    lon.attrs['units'] = 'degrees'
    lon.attrs['description'] = 'Longitude'

    # Initialize DEM height dataset and import values:
    print('kapok.uavsar.load | Importing DEM heights. ('+time.ctime()+')')
    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = mlook(zoom(llh[:,:,2],mlwin_lkv)[azbounds[0]:azbounds[1],rngbounds[0]:rngbounds[1]],mlwin)
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Processor DEM'
    
    del llh
    
    
    # Initialize master track incidence angle dataset:
    inc = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Master Track Incidence Angle'
    
    # Initialize kz dataset:
    if num_bl > 1:
        kz = f.create_dataset('kz', (num_bl, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    else:
        kz = f.create_dataset('kz', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    
    
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
    
    # Get sensor wavelength from annotation file.
    wavelength = ann.query('Center Wavelength')
    wavelength /= 100 # convert from cm to m
    f.attrs['wavelength'] = wavelength
       
    # Starting S coordinate of SLCs, and S coordinate spacing (in meters).
    sstart = ann.query('slc_1_1x1_mag.row_addr')
    sspacing = ann.query('slc_1_1x1_mag.row_mult')

    # S Coordinate Bounds for Desired Subset:
    sbounds = azbounds*sspacing + sstart

    
    # Kz Calculation for Each Baseline
    
    # Statically Defined Variables
    lkv = np.array(lkvmm,dtype='float64')
    del lkvmm
    masterlkv = np.zeros((rngsize_slc,3),dtype='float64')
    slavelkv = np.zeros((rngsize_slc,3),dtype='float64')
    lookcrossv = np.zeros((rngsize_slc,3),dtype='float64')
    
    inc_slc = np.zeros((azbounds[1]-azbounds[0],rngsize_slc),dtype='float64')
    kz_slc = np.zeros((azbounds[1]-azbounds[0],rngsize_slc),dtype='float32')
    
    velocityenu = np.array(sch2enu(1, 0, 0, peglat, peglon, peghdg)).astype('float64')
    proj_lkv_velocity = np.zeros((rngsize_slc,3),dtype='float64')
    
    tempinc = np.zeros((rngsize_slc),dtype='float64')
    tempdiff = np.zeros((rngsize_slc,3),dtype='float64')
    
    mbaseenu = np.zeros((azbounds[1]-azbounds[0],3),dtype='float64')
    sbaseenu = np.zeros((azbounds[1]-azbounds[0],3),dtype='float64')
    baseline = np.zeros((3),dtype='float64')
    baselinep = np.zeros((rngsize_slc),dtype='float64')
    baselinepsign = np.zeros((rngsize_slc),dtype='float64')
       
    
    # Main Kz/Incidence Calculation Loop
    for master in range(0,num_tracks-1):
        mbasefile = glob(datapath+tracknames[master]+'*HH*.baseline')
        
        if len(mbasefile) == 1:
            mbasesch = np.loadtxt(mbasefile[0])
        elif len(mbasefile) > 1:
            print('kapok.uavsar.load | Too many .baseline files matching pattern: "'+datapath+tracknames[master]+'*HH*.baseline'+'".  Aborting.')
            return
        else:
            print('kapok.uavsar.load | Cannot find .baseline file matching pattern: "'+datapath+tracknames[master]+'*HH*.baseline'+'".  Aborting.')
            return
        
        istart = np.where(np.isclose(mbasesch[:,0],sbounds[0]))[0][0] # index of first azimuth line in subset
        iend = np.where(np.isclose(mbasesch[:,0],sbounds[1]))[0][0] # index of last azimuth line in subset
        mbaseenu[:] = sch2enu(mbasesch[istart:iend,1],mbasesch[istart:iend,2],mbasesch[istart:iend,3], peglat, peglon, peghdg)[0:mbaseenu.shape[0],:]
        
        for slave in range(master+1,num_tracks):
            print('kapok.uavsar.load | Calculating kz for tracks '+str(master)+' and '+str(slave)+'. ('+time.ctime()+')')

            sbasefile = glob(datapath+tracknames[slave]+'*HH*.baseline')
            
            if len(sbasefile) == 1:
                sbasesch = np.loadtxt(sbasefile[0])
            elif len(sbasefile) > 1:
                print('kapok.uavsar.load | Too many .baseline files matching pattern: "'+datapath+tracknames[slave]+'*HH*.baseline'+'".  Aborting.')
                return
            else:
                print('kapok.uavsar.load | Cannot find .baseline file matching pattern: "'+datapath+tracknames[slave]+'*HH*.baseline'+'".  Aborting.')
                return

            sbaseenu[:] = sch2enu(sbasesch[istart:iend,1],sbasesch[istart:iend,2],sbasesch[istart:iend,3], peglat, peglon, peghdg)[0:sbaseenu.shape[0],:]
            
            for az in range(0,azbounds[1]-azbounds[0]):
                lkvrowindex = (az+azbounds[0]) // mlwin_lkv[0]

                masterlkv[:,0] = (zoom(lkv[lkvrowindex,:,0],mlwin_lkv[1]) - mbaseenu[az,0])[0:rngsize_slc]
                masterlkv[:,1] = (zoom(lkv[lkvrowindex,:,1],mlwin_lkv[1]) - mbaseenu[az,0])[0:rngsize_slc]
                masterlkv[:,2] = (zoom(lkv[lkvrowindex,:,2],mlwin_lkv[1]) - mbaseenu[az,0])[0:rngsize_slc]
                
                # Project look vector onto velocity vector.
                proj_lkv_velocity = np.sum(masterlkv*velocityenu,axis=1)[:,np.newaxis]*velocityenu/np.linalg.norm(velocityenu)
                
                # Component of look vector orthogonal to velocity.
                tempdiff = (masterlkv-proj_lkv_velocity)
                inc_slc[az,:] = np.arccos(np.abs(tempdiff[:,2])/np.linalg.norm(tempdiff,axis=1))
                
                lookcrossv[:,0] = masterlkv[:,1]*velocityenu[2] - masterlkv[:,2]*velocityenu[1]
                lookcrossv[:,1] = masterlkv[:,2]*velocityenu[0] - masterlkv[:,0]*velocityenu[2]
                lookcrossv[:,2] = masterlkv[:,0]*velocityenu[1] - masterlkv[:,1]*velocityenu[0]
                lookcrossv *= -1
                
                slavelkv[:,0] = masterlkv[:,0] + mbaseenu[az,0] - sbaseenu[az,0]
                slavelkv[:,1] = masterlkv[:,1] + mbaseenu[az,0] - sbaseenu[az,0]
                slavelkv[:,2] = masterlkv[:,2] + mbaseenu[az,0] - sbaseenu[az,0]
                               
                baseline = mbaseenu[az] - sbaseenu[az]
                
                baselinepsign = np.sum(baseline*lookcrossv,axis=1)/np.sum(lookcrossv*lookcrossv,axis=1)
                baselinep = np.linalg.norm((np.sum(baseline*lookcrossv,axis=1)/np.sum(lookcrossv*lookcrossv,axis=1))[:,np.newaxis]*lookcrossv,axis=1)
                
                kz_slc[az] = (4*np.pi/wavelength) * baselinep / (np.linalg.norm(masterlkv,axis=1)*np.sin(inc_slc[az,:])) * np.sign(baselinepsign)
                

            bl = int(slave*(slave-1)/2 + master)
            if num_bl > 1:
                kz[bl] = mlook(kz_slc[:,rngbounds[0]:rngbounds[1]], mlwin)
            else:
                kz[:] = mlook(kz_slc[:,rngbounds[0]:rngbounds[1]], mlwin)
        
        if master == 0: # save incidence angle
            print('kapok.uavsar.load | Saving incidence angle for track 0. ('+time.ctime()+')')
            inc[:] = mlook(inc_slc[:,rngbounds[0]:rngbounds[1]], mlwin)

    del kz_slc, inc_slc
    
     # Close the file, then return it as a Scene object.
    f.close()    
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