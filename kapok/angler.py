# -*- coding: utf-8 -*-
"""Angler Data Import Module

    Module for importing data processed by Angler.
    
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
from glob import glob

import numpy as np
import scipy.constants as constants
import h5py

import kapok
from kapok.lib import mlook, smooth, mb_bl_index


def load(pathlist, outfile, mlwin=(20,5), smwin=(1,1), compression='gzip',
         compression_opts=4, name='', site='', overwrite=False):
    """Load SLC data from the Angler processor, convert into the internal
        Kapok format, and save to target HDF5 file.
    
    Arguments:
        pathlist (str): Paths pointing to Angler processed data.  Should be a
            list of paths if you wish to combine multiple Angler-processed
            datasets.  The data should have some (small) overlap in the SLCs.
            Paths must be given in ascending S coordinate order!  Each path
            should contain folders named 'track1', 'track2', etc., for each
            track.  Each track folder should then contain SLC files for the
            HH, HV, and VV polarizations, as well as inc, kz, lat, lon, and
            DEM height files.  All of these files should have the same
            dimensions as the SLCs.
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
        compression (str): Compression keyword used when creating the HDF5
            datasets to store the covariance matrix, incidence angle, kz,
            and DEM heights.  Default: 'gzip'.
        compression_opts (int): Compression option value used when creating
            the HDF5 datasets.  Number from 0 to 9, with 9 being maximum
            compression.  Default: 4.
        name (str): String describing the stack name (e.g., 'Ecuador').  Will
            be stored in the HDF5 attributes.  Default: None.
        site (str): Longer string describing the data location (e.g., 'Churute
            Mangroves, Ecuador.')  Will be stored in the HDF5 attributes.
            Default: None.
        overwrite (bool): Set to True to overwrite an existing Kapok HDF5
            file if necessary.  Default: False.
            
    Returns:
        scene: A Kapok scene object pointing to the newly created HDF5 file.
    
    """  
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
    
    
    # Number of segments equal to number of input paths.        
    num_segments = len(pathlist)
    
    # Look at config file for first segment to get range index size, basic
    # attributes, etc.
    cfgfile = findfile(pathlist[0]+'/track1/processor_hh*.cfg')
    if cfgfile is None:
        return
    else:
        cfg = Config(cfgfile)
    

    azsize_slc = np.zeros(num_segments,dtype='int32')
    azsize_slc[0] = cfg.query('number of azimuth samples')    
    rngsize_slc = cfg.query('number of range samples')
    
    trackpaths = glob(pathlist[0]+'/track?')
    num_tracks = len(trackpaths)
    num_cov_elements = num_tracks * 3
    
    kzfiles = glob(pathlist[0]+'/track1/kz*.dat')
    num_baselines = len(kzfiles)



    # Loop to load segments.
    for seg in range(num_segments):
        print('kapok.angler.load | Loading segment #'+str(seg+1)+'/'+str(num_segments)+'. ('+time.ctime()+')                ')
        
        cfgfile = findfile(pathlist[seg]+'/track1/processor_hh*.cfg')
        if cfgfile is None:
            return
        else:
            cfg = Config(cfgfile)
            
        azsize_slc[seg] = cfg.query('number of azimuth samples')
        
        # Find starting azimuth to best combine segments.
        sstart = cfg.query('first azimuth sample')
        sspacing = cfg.query('azimuth spacing')
        
        if seg == 0:
            azstart = 0
            slast = sstart + sspacing*((azsize_slc[seg] // mlwin[0] * mlwin[0]) - 1)
        else:
            svector = np.linspace(sstart,sstart + (sspacing*(azsize_slc[seg]-1)),num=azsize_slc[seg])
            sdist = np.square(svector - slast)
            azstart = np.argmin(sdist) + 1
            if np.min(sdist) > 0.01:
                print('kapok.angler.load | Warning: Join between segments '+str(seg)+' and '+str(seg+1)+' off by '+str(np.round(svector[azstart-1] - slast,decimals=2))+' m.  Check that these segments are actually contiguous.')
            
            if azstart > 500:
                print('kapok.angler.load | Warning:  Starting azimuth index for segment '+str(seg+1)+' is '+str(azstart)+'!  Check order of input path list.')
                
            slast = sstart + sspacing*((azsize_slc[seg] // mlwin[0] * mlwin[0]) - 1)
        
        
        # Load SLC Files for Segment
        slcstack = np.empty((num_cov_elements,azsize_slc[seg]-azstart,rngsize_slc),dtype='complex64')
        for tr in range(num_tracks):
            for pol, pol_str in enumerate(['hh', 'hv', 'vv']):
                index = (tr * 3) + pol
                slcfile = findfile(pathlist[seg]+'/track'+str(tr+1)+'/slc_'+pol_str+'_*.dat')
                
                print('kapok.angler.load | Loading SLC file #'+str(index+1)+'/'+str(num_cov_elements)+'. ('+time.ctime()+')               ', end='\r')                
                if slcfile is None:
                    return
                elif pol == 1:
                    slcstack[index] = np.sqrt(2) * np.memmap(slcfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<c8',mode='c')[azstart:]
                else:
                    slcstack[index] = np.memmap(slcfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<c8',mode='c')[azstart:]

                    
        # Calculate covariance matrix buffer.
        covbuffer = np.zeros(((azsize_slc[seg]-azstart)//mlwin[0], rngsize_slc//mlwin[1], num_cov_elements, num_cov_elements), dtype='complex64')
        for row in range(num_cov_elements):
            for col in range(row,num_cov_elements):
                print('kapok.angler.load | Calculating element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')               ', end='\r')
                covbuffer[:,:,row,col] = mlook(slcstack[row] * np.conj(slcstack[col]), mlwin)
        
        
        # Load Other Files (Latitude, Longitude, etc.)
        print('kapok.angler.load | Loading latitudes. ('+time.ctime()+')               ', end='\r')
        latfile = findfile(pathlist[seg]+'/track1/latitude*.dat')
        if latfile is None:
            return
        else:
            latbuffer = np.degrees(mlook(np.memmap(latfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin))
        
        
        print('kapok.angler.load | Loading longitudes. ('+time.ctime()+')               ', end='\r')
        lonfile = findfile(pathlist[seg]+'/track1/longitude*.dat')
        if lonfile is None:
            return
        else:
            lonbuffer = np.degrees(mlook(np.memmap(lonfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin))
        
        
        print('kapok.angler.load | Loading DEM heights. ('+time.ctime()+')               ', end='\r')
        demfile = findfile(pathlist[seg]+'/track1/height*.dat')
        if demfile is None:
            return
        else:
            dembuffer = mlook(np.memmap(demfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin)
        
        
        print('kapok.angler.load | Loading incidence angles. ('+time.ctime()+')               ', end='\r')            
        incfile = findfile(pathlist[seg]+'/track1/inc*.dat')
        if incfile is None:
            return
        else:
            incbuffer = mlook(np.memmap(incfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin)
        
        
        print('kapok.angler.load | Loading kz. ('+time.ctime()+')               ', end='\r')
        kzbuffer = np.zeros((num_tracks, (azsize_slc[seg]-azstart)//mlwin[0], rngsize_slc//mlwin[1]), dtype='float32')
        
        if num_baselines > 1:
            for tr in range(1,num_tracks):
                bl = mb_bl_index(0, tr)
                kzfile = findfile(pathlist[seg]+'/track1/kz*bl'+str(bl+1)+'*.dat')
                if kzfile is None:
                    return
                else:
                    kzbuffer[tr] = mlook(np.memmap(kzfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin)
        else:
            kzfile = findfile(pathlist[seg]+'/track1/kz*.dat')
            if kzfile is None:
                return
            else:
                kzbuffer[1] = mlook(np.memmap(kzfile, shape=(azsize_slc[seg],rngsize_slc),dtype='<f4',mode='c')[azstart:],mlwin)
        
        
        if seg == 0: # first segment, create arrays.
            cov = covbuffer.copy()
            lat = latbuffer.copy()
            lon = lonbuffer.copy()
            dem = dembuffer.copy()
            inc = incbuffer.copy()
            kz = kzbuffer.copy()
        else: # after first segment, stack buffer with previous data
            cov = np.vstack((cov, covbuffer))
            lat = np.vstack((lat, latbuffer))
            lon = np.vstack((lon, lonbuffer))
            dem = np.vstack((dem, dembuffer))
            inc = np.vstack((inc, incbuffer))            
            kz = np.concatenate((kz, kzbuffer), axis=1)  
    
    
    # Boxcar Smoothing:
    if smwin != (1,1):
        print('kapok.angler.load | Boxcar averaging covariance matrix. ('+time.ctime()+')')
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                cov[:,:,row,col] = smooth(cov[:,:,row,col],smwin)
    
    
    # Save arrays to HDF5 datasets.
    print('kapok.angler.load | Creating datasets. ('+time.ctime()+')')
    cov = f.create_dataset('cov', data=cov, dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)
    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*HV', 'VV'],dtype='S')

    lat = f.create_dataset('lat', data=lat, dtype='float32', compression=compression, compression_opts=compression_opts)
    lat.attrs['units'] = 'degrees'
    lat.attrs['description'] = 'Latitude'
        
    lon = f.create_dataset('lon', data=lon, dtype='float32', compression=compression, compression_opts=compression_opts)
    lon.attrs['units'] = 'degrees'
    lon.attrs['description'] = 'Longitude'

    dem = f.create_dataset('dem', data=dem, dtype='float32', compression=compression, compression_opts=compression_opts)
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Processor DEM'

    inc = f.create_dataset('inc', data=inc, dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Reference Track Incidence Angle'
    
    kz = f.create_dataset('kz', data=kz, dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    kz.attrs['indexing'] = 'track'
    kz.attrs['slope_corrected'] = False
    
    
    # Save HDF5 file attributes.
    print('kapok.angler.load | Saving metadata. ('+time.ctime()+')')
    f.attrs['stack_name'] = name
    f.attrs['site'] = site
    f.attrs['sensor'] = 'UAVSAR'
    f.attrs['processor'] = 'angler'
    
    f.attrs['reference_altitude'] = cfg.query('reference track altitude')
    f.attrs['image_starting_slant_range'] = cfg.query('first range sample')
    f.attrs['slc_azimuth_pixel_spacing'] = cfg.query('azimuth spacing')
    f.attrs['slc_slant_range_pixel_spacing'] = cfg.query('range spacing')
    f.attrs['cov_azimuth_pixel_spacing'] = cfg.query('azimuth spacing') * mlwin[0]
    f.attrs['cov_slant_range_pixel_spacing'] = cfg.query('range spacing') * mlwin[1]
    
    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_baselines'] = num_baselines
    f.attrs['tracks'] = np.array(pathlist,dtype='S')
    
    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts

    f.attrs['subset'] = False
    
    # Multi-looked, subsetted, image dimensions:
    azsize = cov.shape[0]
    rngsize = cov.shape[1]
    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['dim_slc'] = (int(np.sum(azsize_slc)),rngsize_slc)
    f.attrs['ml_window'] = mlwin
    f.attrs['sm_window'] = smwin
    
                    
    # Get SCH peg.
    f.attrs['peg_latitude'] = cfg.query('peg latitude')
    f.attrs['peg_longitude'] = cfg.query('peg longitude')
    f.attrs['peg_heading'] = cfg.query('peg heading')
    
    
    # Get sensor wavelength.
    frequency = cfg.query('center frequency')
    f.attrs['wavelength'] = constants.c / frequency
    
    print('kapok.angler.load | Data import complete. ('+time.ctime()+')')
    
    # Close the file, then return it as a Scene object.
    f.close()    
    return kapok.Scene(outfile)
    
    
class Config(object):
    """Class for loading and interacting with an Angler Config (.cfg) file."""
    
    def __init__(self, file):
        """Load in the specified cfg file as a list of strings and
            initialize.
        
        Arguments:
            file (str): Angler .cfg filename to load.
            
        """
        self.file = file
        fd = open(self.file, 'r')
        self.cfg = fd.read().split('\n')
        return
        
    
    def query(self, keyword):
        """Query the cfg file for the specified annotation keyword. 
        
        Arguments:
            keyword (str): The keyword to query.
            
        Returns:
            value: The value of the specified keyword.
            
        """
        for n in range(len(self.cfg)):
            if self.cfg[n].startswith(keyword):
                try:
                    val = self.cfg[n].rsplit('=')[-1].split('#')[0].split()[0]
                    val = np.array(val,dtype='float') # if we can convert the string to a number, do so
                    if (val - np.floor(val)) == 0:
                        val = np.array(val,dtype='int')  # if it's an integer, convert it to one (e.g., number of samples)
                    return val
                except ValueError: # if we can't convert the string to a number, leave it as a string
                    val = self.cfg[n].split('=',maxsplit=1)[-1].split('#')[0].strip()
                    return val
                    
        return None
        
        
def findfile(path):
    """Check if file matching pattern exists and return the filename.
    
    If multiple files match pattern, print error message and abort.  If no
    files match pattern, print error message and abort.  Uses glob.
    
    Arguments:
        path (str): The path/file pattern to check (can include * and ?).
    
    """
    file = glob(path)
    
    if len(file) == 1:
        return file[0]
    elif len(file) > 1:
        print('kapok.angler.load | Too many files matching pattern: "'+path+'".  Aborting.')
        return None
    else:
        print('kapok.angler.load | Cannot find file matching pattern: "'+path+'".  Aborting.')
        return None
