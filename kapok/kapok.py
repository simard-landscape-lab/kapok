# -*- coding: utf-8 -*-
"""Kapok Main Module

    Core Kapok module containing Scene class definition and methods.  A Scene
    object contains a PolInSAR dataset including covariance matrix, incidence
    angle, kz, latitude, longitude, processor DEM, and metadata.
    
    Methods available for data visualization, coherence optimization, and
    forest model inversion, among others.

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
import collections
import os.path

import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.ion()

import kapok.geo
import kapok.vis
from kapok.lib import makehermitian, mb_cov_index, mb_tr_index, mb_bl_index, mb_num_baselines



class Scene(object):
    """Scene object containing a PolInSAR dataset, and methods for viewing
        and processing the data.
    
    """
    
    def __init__(self, file):
        """Scene initialization method.
        
        Arguments:
            file (str): Path and filename to a previously saved Kapok HDF5
                PolInSAR scene.
        
        """
        # Load the HDF5 file.  If it can't be loaded, abort.
        try:
            self.f = h5py.File(file,'r+')
        except:
            print('kapok.Scene | Cannot load specified HDF5 file: "'+file+'".  Ensure file exists.  Aborting.')
            return
            
        # Easy access to datasets:
        self.cov = self.f['cov']
        
        self.lat = self.f['lat']
        self.lon = self.f['lon']
        self.dem = self.f['dem']
                
        self.kza = self.f['kz']
        self.inc = self.f['inc']
        
        if 'pdopt/coh' in self.f:
            self.pdcoh = self.f['pdopt/coh']
        else:
            self.pdcoh = None
        
        if 'pdopt/weights' in self.f:
            self.pdweights = self.f['pdopt/weights']
        else:
            self.pdweights = None
            
        if 'pdopt/cohminor' in self.f:
            self.pdcohminor = self.f['pdopt/cohminor']
        else:
            self.pdcohminor = None
            
        if 'products' in self.f:
            self.products = self.f['products']
            
        if 'ancillary' in self.f:
            self.ancillary = self.f['ancillary']
        
        
        # Easy access to some commonly used metadata attributes:
        self.site = self.f.attrs['site']
        self.name = self.f.attrs['stack_name']
        self.tracks = self.f.attrs['tracks']
        
        self.dim = tuple(self.f.attrs['dim'])
        self.spacing = (self.f.attrs['cov_azimuth_pixel_spacing'], self.f.attrs['cov_slant_range_pixel_spacing'])
        
        self.num_tracks = self.f.attrs['num_tracks']
        self.num_baselines = self.f.attrs['num_baselines']
        self.num_pol = self.cov.attrs['num_pol']
        
        self.ml_window = self.f.attrs['ml_window']
        self.sm_window = self.f.attrs['sm_window']
        
        self.wavelength = self.f.attrs['wavelength']  
        
        self.compression = self.f.attrs['compression']
        self.compression_opts = self.f.attrs['compression_opts']
        
        # Detect old kz format (baseline-indexed) and convert to the new
        # format (track-indexed).
        if not (('indexing' in self.f['kz'].attrs) and (self.f['kz'].attrs['indexing'] == 'track')):
            print('kapok.Scene | Detected old Kapok file version with kz indexed by baseline.  Do you wish to convert this file to the new format? [y/n]')
            answer = input('')
            answer = answer.lower()
            if (answer == 'y') or (answer == 'yes'):
                print('kapok.Scene | Converting to new format.')
                kz_old = self.f['kz'][:]
                
                if ('slope_corrected' in self.f['kz'].attrs):
                    slope_corrected = self.f['kz'].attrs['slope_corrected']
                else:
                    slope_corrected = False
                
                del self.f['kz']
                
                kz = self.f.create_dataset('kz', (self.num_tracks, self.dim[0], self.dim[1]), dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                kz.attrs['units'] = 'radians/meter'
                kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
                kz.attrs['indexing'] = 'track'
                kz.attrs['slope_corrected'] = slope_corrected
                
                if self.num_baselines > 1:
                    for tr in range(1,self.num_tracks):
                        bl = mb_bl_index(0, tr)
                        kz[tr] = kz_old[bl]
                else:
                    kz[1] = kz_old

                self.kza = self.f['kz']
                self.f.flush()
                print('kapok.Scene | Conversion complete.')
            else:
                print('kapok.Scene | File format not converted to new version.  Note: You will experience errors when using Scene methods which access the kz values.')
        
        
    def get(self, path):
        """Returns a specified dataset from the HDF5 file.
        
        Arguments:
            path (str): Path and name to an HDF5 dataset within the Scene
                file.  Note that shortcuts can be used if you wish to access
                a dataset within the 'products/' or 'ancillary/' groups.
                These group names can be omitted and the function will check
                within them for the given dataset name.  Exact paths take
                precedence, followed by datasets in 'products/', followed by
                datasets in 'ancillary/'
                
        Returns:
            data (array): The desired HDF5 dataset, in the form of a NumPy
                array.
                
        """
        if path == 'kz':
            print('kapok.Scene.get | Warning: Don\'t use Scene.get() to retrieve kz values.  Use Scene.kz() instead.  Returning None.')
            return None
        elif path in self.f:
            if isinstance(self.f[path], h5py.Dataset):
                return self.f[path][:]
            else:
                print('kapok.Scene.get | Desired path exists in HDF5 file, but is not a dataset.  Please specify the name of a HDF5 dataset.')
                return None
        elif ('products/'+path) in self.f:
            if isinstance(self.f['products/'+path], h5py.Dataset):
                return self.f['products/'+path][:]
            else:
                print('kapok.Scene.get | Desired path exists in HDF5 file, but is not a dataset.  Please specify the name of a HDF5 dataset.')
                return None
        elif ('products'+path) in self.f:
            if isinstance(self.f['products'+path], h5py.Dataset):
                return self.f['products'+path][:]
            else:
                print('kapok.Scene.get | Desired path exists in HDF5 file, but is not a dataset.  Please specify the name of a HDF5 dataset.')
                return None
        elif ('ancillary/'+path) in self.f:
            if isinstance(self.f['ancillary/'+path], h5py.Dataset):
                return self.f['ancillary/'+path][:]
            else:
                print('kapok.Scene.get | Desired path exists in HDF5 file, but is not a dataset.  Please specify the name of a HDF5 dataset.')
                return None
        elif ('ancillary'+path) in self.f:
            if isinstance(self.f['ancillary'+path], h5py.Dataset):
                return self.f['ancillary'+path][:]
            else:
                print('kapok.Scene.get | Desired path exists in HDF5 file, but is not a dataset.  Please specify the name of a HDF5 dataset.')
                return None
        else:
            print('kapok.Scene.get | Desired path does not exist in HDF5 file.')
            return None
    
    
    def kz(self, bl):
        """Returns an array of kz values for the specified baseline.
        
        Arguments:
            bl (int or tuple): The baseline of interest.  This argument can
                either be the baseline index itself, or in the form of a tuple
                of track indices (first value is the master track, second
                value is the slave track).  Note: All baselines and tracks
                are indexed starting at zero.  bl=(0,1) returns the kz values
                between the first and second tracks.  See lib/mb.py for an
                explanation of how baselines are indexed.
            
        Returns:
            kz (array): Array of kz values.
                
        """
        if not isinstance(bl, (collections.Sequence, np.ndarray)):
            bl = mb_tr_index(bl)
            
        if (bl is None) or (bl[0] >= self.num_tracks) or (bl[1] >= self.num_tracks):
            print('kapok.Scene.kz | Invalid baseline index.  Returning None.')
            return None
        
        return (self.kza[bl[1]] - self.kza[bl[0]])
    
    
    def query(self, path=None):
        """Prints useful lists of datasets and attributes inside the HDF5 file.

        Arguments:
            path (str): Path and name to an HDF5 dataset or group within the
                Scene file.  Its attributes will be printed.  If path is not
                specified, a list of all groups and datasets in the HDF5 file
                will be printed instead.  If path is an empty string, the
                attributes of the main HDF5 file will be printed.
        
        """
        if path is None:
            print('kapok.Scene.query | Printing groups and datasets in HDF5 file...')
            def printindex(name):
                print('kapok.Scene.query | '+name)
            self.f.visit(printindex)
        elif path in self.f:
            print('kapok.Scene.index | Printing attributes of "'+path+'"...')
            for item in self.f[path].attrs.keys():
                try:
                    print('kapok.Scene.query | ' + item + ":", self.f[path].attrs[item].astype('str'))
                except:
                    print('kapok.Scene.query | ' + item + ":", self.f[path].attrs[item])
        elif path == '':
            print('kapok.Scene.index | Printing attributes of main HDF5 file...')
            for item in self.f.attrs.keys():
                try:
                    print('kapok.Scene.query | ' + item + ":", self.f.attrs[item].astype('str'))
                except:
                    print('kapok.Scene.query | ' + item + ":", self.f.attrs[item])
        else:
            print('kapok.Scene.query | Desired path does not exist in HDF5 file.')
    
    
    def inv(self, method='rvog', name=None, desc=None, overwrite=False,
            bl='all', tdf=None, epsilon=0.4, groundmag=None, ext=None, mu=0,
            rngslope=None, mask=None, blcriteria='prod', minkz=0.0314,
            **kwargs):
        """Forest model inversion.
        
        Estimate forest height using one of a number of models relating
        the forest's physical parameters to the PolInSAR observables.
        
        Currently implemented models:
            'sinc': Sinc model for the coherence magnitude.  Calls
                function kapok.sinc.sincinv().
            'sincphase': Sum of weighted sinc coherence model and phase
                height of volume coherence above ground surface.  Estimates
                ground phase using kapok.topo.groundsolver().  Model
                implemented in function kapok.sinc.sincphase().
            'rvog' (default): Random Volume over Ground model.  Estimates
                ground phase using kapok.topo.groundsolver().  Model inversion
                is performed by function kapok.rvog.rvoginv().
                
        After model inversion is performed, HDF5 datasets will be created in
        'products/<name>/' for each estimated model parameter, where the model
        parameters are:
            'hv': Forest/volume height, in meters.  Used by all models.
            'ext': Wave extinction, in Nepers/meter.  Used by rvog model.
            'phi': Complex-valued ground coherence.  The phase of this
                coherence is the topographic phase.  Used by sincphase and
                rvog models.
            'mu': Ground-to-volume amplitude ratio for the highest observed
                coherence.  In the single baseline case, this is generally
                set to a fixed value, often zero.  Used by rvog model.
            'tdf': Temporal decorrelation factor, describing the effect of
                temporal decorrelation on the highest observed coherence.
                Can be used by sinc, sincphase, and rvog models, though the
                default value is generally unity.
                
        Note that datasets will not be created for all of the above parameters
        for every inversion.  Only parameters that vary from pixel to pixel
        will be saved as datasets.  In the event that parameters are fixed
        during a model inversion, those fixed parameter values will be saved
        as attributes to 'products/<name>/'.       
        
        Arguments:
            method (str): Name of the desired model to invert.  Currently
                supported values: 'sinc', 'sincphase', and 'rvog'.
                Default: 'rvog'.
            name (str): Name for the saved group containing the inverted model
                parameters (forest height, extinction, etc.).  Default: Equal
                to method.
            desc (str): String describing the inversion model, parameters,
                etc., which will be stored in the attributes
            overwrite (bool): If a dataset with the requested name already
                exists, overwrite it.  Default: If the dataset already exists,
                function will abort with an error message.
            bl (int or array or str): Baseline index specifying which
                baseline(s) to invert.  To do an inversion using multiple
                baselines, set bl to a list or array of baselines to include.
                To include all baselines, you can use bl='all'.  Note that the
                current implementation only supports multi-baseline inversion
                of the RVoG model in an incoherent manner, through the
                baseline selection function kapok.rvog.rvogblselect()
                Multi-baseline inversion using the 'sinc' or 'sincphase'
                models is not supported.  Default: 'all'.
            tdf (array): Array of temporal decorrelation factors to use in
                the model inversion, if desired.  Default: None.
            epsilon: Value of the epsilon parameter of the sinc and phase
                difference inversion.  Only used for method 'sincphase'.
            groundmag (array): Value for the magnitude of the estimated ground
                coherences.  Used when finding the ground solution using the
                line fit procedure.  Default: No ground decorrelation.
            ext: Fixed value(s) for the extinction parameter of the
                RVoG model, if desired.
            mu: Fixed value(s) for the ground-to-volume scattering ratio
                of the RVoG model.  Defaults to zero.
            rngslope: Terrain slope angles in the ground range direction.
                Not used by the sinc model.  Defaults to zero (flat terrain).
                Note: If you want this function to automatically calculate
                the slope angle from the SRTM DEM used by the UAVSAR
                processor, you can set rngslope to True.
            mask (array): Boolean array of mask values.  Only pixels where
                (mask == True) will be inverted.  Defaults to array of ones
                (all pixels inverted).
            blcriteria (str): Set to 'prod' to use coherence line product
                as the baseline selection criteria, to 'var' to use
                expected height variance, or to 'ecc' to use coherence region
                eccentricity.  See kapok.rvog.rvogblselect() for more details.
                Note that this keyword is only considered if multi-baseline
                inversion is enabled by setting the bl keyword to a list of
                baseline indices, or to the string 'all'.  Default: 'prod'.
            minkz (float): For a baseline to be inverted, the absolute
                value of kz must be at least this amount.  This keyword
                argument allows baselines with zero spatial separation to
                easily be excluded from multi-baseline inversions.  Note that
                for single-baseline inversions, the inversion will still
                proceed, but a warning will be printed.  Default: 0.0314
                (representing a pi height of ~100m).
            **kwargs: Additional keyword arguments passed to the model
                inversion functions, if desired.  See model inversion
                function headers for more details.  Default: None.
                
        Returns:
            hv (hdf5 dataset): Link to the HDF5 dataset containing the
                estimated forest heights.
        
        """
        if name is None:
            name = method
            
        if isinstance(bl, str) and 'all' in bl:
            bl = np.arange(self.num_baselines)
        
        if rngslope is True:
            from kapok.lib import calcslope
            rngslope, azslope = calcslope(self.dem, self.spacing, self.inc)
            del azslope
            
        # Check If Group Exists:
        if ('products/'+name in self.f) and overwrite:
            del self.f['products/'+name]
        elif ('products/'+name in self.f) and (overwrite == False):
            print('kapok.Scene.inv | Model inversion group with name "products/'+name+'" already exists.  If you wish to replace it, set overwrite keyword.  Aborting.')
            return None

        result = self.f.create_group('products/'+name)
        
        if desc is not None:
            result.attrs['desc'] = desc 
        
        # Perform Model Inversion...
        if isinstance(bl, (collections.Sequence, np.ndarray)):
            if 'rvog' not in method:
                print('kapok.Scene.inv | Multiple baselines selected for inversion, but model is not set to "rvog".  Currently, multi-baseline inversion is only supported for the RVoG model.  Aborting.')
                return None
        else:
            print('kapok.Scene.inv | Performing model inversion for baseline #'+str(bl)+'.  Average kz: '+str(np.nanmean(self.kz(bl)))+'. ('+time.ctime()+')')
        
        # Sinc Model
        if method == 'sinc':
            import kapok.sinc
            if 'pdopt/coh' in self.f:
                print('kapok.Scene.inv | Performing sinc model inversion using phase diversity highest coherence. ('+time.ctime()+')')
                if desc is None:
                    result.attrs['desc'] = 'Sinc coherence model.  Used phase diversity high coherence as volume coherence.'

                result.attrs['pol'] = 'high'                    
                gammav = self.coh('high', bl=bl)
            else:
                print('kapok.Scene.inv | Phase diversity coherence optimization not yet run.  Performing sinc model inversion using HV polarization coherence. ('+time.ctime()+')')
                if desc is None:
                    result.attrs['desc'] = 'Sinc coherence model.  Used HV coherence as volume coherence.'

                result.attrs['pol'] = 'HV'
                gammav = self.coh('HV', bl=bl)
            
            
            hv = kapok.sinc.sincinv(gammav, self.kz(bl), tdf=tdf, mask=mask, **kwargs)
            
            result.attrs['model'] = 'sinc'
            result.attrs['baseline'] = bl
            result.create_dataset('hv', data=hv, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
            result['hv'].attrs['fixed'] = False
            result['hv'].attrs['name'] = 'PolInSAR Forest Height'
            result['hv'].attrs['units'] = 'm'
            
            # Create TDF dataset, if TDF varies across image.
            if isinstance(tdf, (collections.Sequence, np.ndarray)):
                result.create_dataset('tdf', data=tdf, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                result['tdf'].attrs['fixed'] = True
                result['tdf'].attrs['name'] = 'Temporal Decorrelation Factor'
                result['tdf'].attrs['units'] = ''
            elif tdf is not None:
                result.attrs['tdf'] = tdf
                
        
        # Sinc and Phase Diff Model
        elif method == 'sincphase':
            import kapok.sinc
            import kapok.topo
            
            kz = self.kz(bl)
        
            ambh = (2*np.pi/np.nanmean(np.abs(kz)))
            if  np.nanmean(np.abs(kz)) < minkz:
                print('kapok.Scene.inv | Warning: Selected baseline has mean ambiguity height of: '+str(ambh)+' m!  Are you sure you want to invert this baseline?')

            # Slope Correct Kz, if Range Slope Specified
            if (rngslope is not None) and (self.kza.attrs['slope_corrected'] == False):
                print('kapok.Scene.inv | Correcting kz values for range-facing terrain slope angle.')
                kz *= np.sin(self.inc) * np.cos(rngslope) / np.sin(self.inc - rngslope)
            elif (rngslope is not None) and (self.kza.attrs['slope_corrected'] == True):
                print('kapok.Scene.inv | Note: kz values have already been corrected for terrain slope.  No further correction applied.')
            
            
            if 'pdopt/coh' not in self.f:
                print('kapok.Scene.inv | Run phase diversity coherence optimization before performing sinc and phase difference inversion.  Aborting.')
                result = None
            else:                
                if self.num_baselines > 1:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[bl], kz=kz, groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[bl,1],self.pdcoh[bl,0])
                    hv = kapok.sinc.sincphaseinv(coh_high, np.angle(ground), kz, epsilon=epsilon, tdf=tdf, mask=mask, **kwargs)
                else:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[:], kz=kz, groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[1],self.pdcoh[0])
                    hv = kapok.sinc.sincphaseinv(coh_high, np.angle(ground), kz, epsilon=epsilon, tdf=tdf, mask=mask, **kwargs)
                
                if desc is None:
                    result.attrs['desc'] = 'Sinc Coherence and Phase Difference model.'

                result.attrs['model'] = 'sincphase'
                result.attrs['baseline'] = bl
                result.create_dataset('hv', data=hv, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                result['hv'].attrs['fixed'] = False
                result['hv'].attrs['name'] = 'Forest Height'
                result['hv'].attrs['units'] = 'm'
                result.create_dataset('ground', data=ground, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                result['ground'].attrs['fixed'] = False
                result['ground'].attrs['name'] = 'Ground Complex Coherence'
                result['ground'].attrs['units'] = ''
                    
                # Create epsilon dataset, if epsilon varies across the image.
                # Otherwise, store it in an attribute.
                if isinstance(epsilon, (collections.Sequence, np.ndarray)):
                    result.create_dataset('epsilon', data=epsilon, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['epsilon'].attrs['fixed'] = True
                    result['epsilon'].attrs['name'] = 'Epsilon'
                    result['epsilon'].attrs['units'] = ''
                elif epsilon is not None:
                    result.attrs['epsilon'] = epsilon
                    
                # Create TDF dataset, if TDF varies across image.
                if isinstance(tdf, (collections.Sequence, np.ndarray)):
                    if np.any(np.iscomplex(tdf)):
                        result.create_dataset('tdf', data=tdf, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                    else:
                        result.create_dataset('tdf', data=tdf, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['tdf'].attrs['fixed'] = True
                    result['tdf'].attrs['name'] = 'Temporal Decorrelation Factor'
                    result['tdf'].attrs['units'] = ''
                elif tdf is not None:
                    result.attrs['tdf'] = tdf
        
        
        # RVoG Model
        elif method == 'rvog':
            import kapok.rvog
            import kapok.topo
            
            if isinstance(bl, (collections.Sequence, np.ndarray)):
                bl = np.sort(bl)
                kz = np.zeros((len(bl),self.dim[0],self.dim[1]),dtype='float32')
                for m, n in enumerate(bl):
                    kz[m] = self.kz(n)
                    
                if 'ecc' in blcriteria:
                    gamma, kz, blsel = kapok.rvog.rvogblselect(self.pdcoh[bl,:,:,:], kz, method=blcriteria, gammaminor=self.pdcohminor[bl,:,:,:], minkz=minkz)
                else:
                    gamma, kz, blsel = kapok.rvog.rvogblselect(self.pdcoh[bl,:,:,:], kz, method=blcriteria, minkz=minkz)
                    
                # Save which baseline was selected for each pixel.
                result.create_dataset('bl', data=bl[blsel], dtype='int16', compression=self.compression, compression_opts=self.compression_opts)
                result['bl'].attrs['fixed'] = False
                result['bl'].attrs['name'] = 'Chosen Baseline Index'
                result['bl'].attrs['units'] = ''
            else:
                kz = self.kz(bl)
            
                ambh = (2*np.pi/np.nanmean(np.abs(kz)))
                if  np.nanmean(np.abs(kz)) < minkz:
                    print('kapok.Scene.inv | Warning: Selected baseline has mean ambiguity height of: '+str(ambh)+' m!  Are you sure you want to invert this baseline?')
                

            # Slope Correct Kz, if Range Slope Specified            
            if (rngslope is not None) and (self.kza.attrs['slope_corrected'] == False):
                print('kapok.Scene.inv | Correcting kz values for range-facing terrain slope angle.')
                kz *= np.sin(self.inc) * np.cos(rngslope) / np.sin(self.inc - rngslope)
            elif (rngslope is not None) and (self.kza.attrs['slope_corrected'] == True):
                print('kapok.Scene.inv | kz values have already been corrected for range-facing terrain slope angle.')
            elif rngslope is None:
                rngslope = 0
            
            if 'pdopt/coh' not in self.f:
                print('kapok.Scene.inv | Run phase diversity coherence optimization before performing RVoG inversion.  Aborting.')
                result = None
            else:
                if isinstance(bl, (collections.Sequence, np.ndarray)):
                    ground, groundalt, volindex = kapok.topo.groundsolver(gamma, kz=kz, groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,gamma[1],gamma[0])
                    hv, exttdf, converged = kapok.rvog.rvoginv(coh_high, ground, self.inc, kz, ext=ext, tdf=tdf, mu=mu, rngslope=rngslope,
                        mask=mask, **kwargs)                
                elif self.num_baselines > 1:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[bl], kz=kz, groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[bl,1],self.pdcoh[bl,0])
                    hv, exttdf, converged = kapok.rvog.rvoginv(coh_high, ground, self.inc, kz, ext=ext, tdf=tdf, mu=mu, rngslope=rngslope,
                        mask=mask, **kwargs)
                else:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[:], kz=kz, groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[1],self.pdcoh[0])
                    hv, exttdf, converged = kapok.rvog.rvoginv(coh_high, ground, self.inc, kz, ext=ext, tdf=tdf, mu=mu, rngslope=rngslope,
                        mask=mask, **kwargs)
                
                if desc is None:
                    result.attrs['desc'] = 'Random Volume over Ground model.'
    
                result.attrs['model'] = 'rvog'
                result.attrs['baseline'] = bl
                if isinstance(bl, (collections.Sequence, np.ndarray)):
                    result.attrs['baseline_selection'] = blcriteria
                else:
                    result.attrs['baseline_selection'] = 'Single Baseline'
                result.create_dataset('hv', data=hv, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                result['hv'].attrs['fixed'] = False
                result['hv'].attrs['name'] = 'Forest Height'
                result['hv'].attrs['units'] = 'm'
                result.create_dataset('ground', data=ground, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                result['ground'].attrs['fixed'] = False
                result['ground'].attrs['name'] = 'Ground Complex Coherence'
                result['ground'].attrs['units'] = ''
                
                # Save Extinction
                if (ext is not None) and isinstance(ext, (collections.Sequence, np.ndarray)):
                    result.create_dataset('ext', data=ext, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['ext'].attrs['fixed'] = True
                    result['ext'].attrs['name'] = 'Extinction'
                    result['ext'].attrs['units'] = 'Np/m'
                elif ext is None:
                    result.create_dataset('ext', data=exttdf, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['ext'].attrs['fixed'] = False
                    result['ext'].attrs['name'] = 'Extinction'
                    result['ext'].attrs['units'] = 'Np/m'
                else:
                    result.attrs['ext'] = ext
                    
                # Save TDF
                if (tdf is not None) and isinstance(tdf, (collections.Sequence, np.ndarray)):
                    if np.any(np.iscomplex(tdf)):
                        result.create_dataset('tdf', data=tdf, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                    else:
                        result.create_dataset('tdf', data=tdf, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['tdf'].attrs['fixed'] = True
                    result['tdf'].attrs['name'] = 'Temporal Decorrelation Factor'
                    result['tdf'].attrs['units'] = ''
                elif tdf is None:
                    result.create_dataset('tdf', data=exttdf, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                    result['tdf'].attrs['fixed'] = False
                    result['tdf'].attrs['name'] = 'Temporal Decorrelation Factor'
                    result['tdf'].attrs['units'] = ''
                else:
                    result.attrs['tdf'] = tdf
                

        
        else:
            print('kapok.Scene.inv | Inversion method "'+method+'" not recognized.  Aborting.')
            return None

        
        # Create groundmag dataset, if groundmag varies across the image.
        # Otherwise, store it in an attribute.
        if (method == 'sincphase') or (method == 'rvog'):
            if isinstance(groundmag, (collections.Sequence, np.ndarray)):
                result.create_dataset('groundmag', data=groundmag, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
                result['groundmag'].attrs['fixed'] = True
                result['groundmag'].attrs['name'] = 'Ground Coherence Magnitude'
                result['groundmag'].attrs['units'] = ''
            elif groundmag is not None:
                result.attrs['groundmag'] = groundmag
        
        
        self.f.flush()
        return result['hv']
    
       
    def opt(self, method='pdopt', reg=0.0, saveall=True, overwrite=False,
            **kwargs):
        """Coherence optimization.
        
        Perform coherence optimization, and save the results to the HDF5 file.
        Currently, phase diversity coherence optimization is the only method
        supported.  The coherences will be saved in 'pdopt/coh'.  If the
        saveall keyword is set to True, the coherence weight vectors will
        be saved in 'pdopt/weights', and the coherences with minimum
        separation in the complex plane (e.g., the coherences along the minor
        axis of an elliptical coherence region) will be saved as
        'pdopt/cohminor'.
            
        Arguments:
            method (str): Desired optimization algorithm.  Currently 'pdopt'
                (phase diversity) is the only method supported.
            reg (float): Covariance matrix regularization factor (see
                kapok.cohopt.pdopt() function for details).  Default: 0.
            saveall (bool): True/False flag, specifies whether to
                save the polarization weight vectors for the optimized
                coherences in 'pdopt/weights', and the minor axis coherences
                in 'pdopt/cohminor'.  Default: True.
            overwrite (bool): True/False flag that determines whether to
                overwrite the current coherences.  If False, will abort if
                the coherences already exist.
            **kwargs: Additional keyword arguments passed to coherence
                optimization function.
        
        """
        if ('pd' in method) or ('pdopt' in method) or ('phase diversity' in method):
            if ('pdopt/coh' in self.f) and (overwrite == False):
                print('kapok.Scene.opt | Phase diversity coherence optimization already performed.  If you want to overwrite, set the overwrite keyword to True.  Aborting.')
            else:
                import kapok.cohopt
                
                # If datasets already exist, remove them.
                if ('pdopt/coh' in self.f):
                    del self.f['pdopt/coh']
                    self.pdcoh = None
                
                if ('pdopt/weights' in self.f):
                    del self.f['pdopt/weights']
                    self.pdweights = None
                    
                if ('pdopt/cohminor' in self.f):
                    del self.f['pdopt/cohminor']
                    self.pdcohminor = None
                    

                if self.num_baselines > 1: # Multiple Baselines
                    self.pdcoh = self.f.create_dataset('pdopt/coh', (self.num_baselines, 2, self.dim[0], self.dim[1]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                
                    if saveall:
                        self.pdweights = self.f.create_dataset('pdopt/weights', (self.num_baselines, 2, self.dim[0], self.dim[1], 3), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                        self.pdcohminor = self.f.create_dataset('pdopt/cohminor', (self.num_baselines, 2, self.dim[0], self.dim[1]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)

                    for bl in range(self.num_baselines):
                        # Get Omega and T matrices:
                        row, col = mb_cov_index(bl, 0, n_pol=self.num_pol)
                        tm = 0.5*(self.cov[:,:,row:row+self.num_pol,row:row+self.num_pol] + self.cov[:,:,col:col+self.num_pol,col:col+self.num_pol])
                        tm = makehermitian(tm)
                        om = self.cov[:,:,row:row+self.num_pol,col:col+self.num_pol]
                        
                        print('kapok.cohopt.pdopt | Beginning phase diversity coherence optimization for baseline index '+str(bl)+'. ('+time.ctime()+')')
                        if saveall:
                            gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin = kapok.cohopt.pdopt(tm, om, reg=reg, returnall=saveall, **kwargs)
                        else:
                            gammamax, gammamin = kapok.cohopt.pdopt(tm, om, reg=reg, returnall=saveall, **kwargs)
                            
                        temp = np.angle(gammamin*np.conj(gammamax))
                        ind = (np.sign(temp) == np.sign(self.kz(bl)))
                        
                        swap = gammamax[ind].copy()
                        gammamax[ind] = gammamin[ind]
                        gammamin[ind] = swap
                        
                        self.pdcoh[bl,0,:,:] = gammamax
                        self.pdcoh[bl,1,:,:] = gammamin
                        del gammamax, gammamin
                        
                        if saveall:
                            self.pdcohminor[bl,0,:,:] = gammaminormax
                            self.pdcohminor[bl,1,:,:] = gammaminormin
                            
                            ind = np.dstack((ind,ind,ind))
                            swap = wmax[ind].copy()
                            wmax[ind] = wmin[ind]
                            wmin[ind] = swap
                            
                            self.pdweights[bl,0,:,:,:] = wmax
                            self.pdweights[bl,1,:,:,:] = wmin
                            
                else: # Single Baseline
                    self.pdcoh = self.f.create_dataset('pdopt/coh', (2, self.dim[0], self.dim[1]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                    
                    # Get Omega and T matrices:
                    row, col = mb_cov_index(0, 0, n_pol=self.num_pol)
                    tm = 0.5*(self.cov[:,:,row:row+self.num_pol,row:row+self.num_pol] + self.cov[:,:,col:col+self.num_pol,col:col+self.num_pol])
                    tm = makehermitian(tm)
                    om = self.cov[:,:,row:row+self.num_pol,col:col+self.num_pol]
                    
                    print('kapok.cohopt.pdopt | Beginning phase diversity coherence optimization for single baseline. ('+time.ctime()+')')
                    if saveall:
                        self.pdweights = self.f.create_dataset('pdopt/weights', (2, self.dim[0], self.dim[1], 3), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                        self.pdcohminor = self.f.create_dataset('pdopt/cohminor', (2, self.dim[0], self.dim[1]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                        gammamax, gammamin, gammaminormax, gammaminormin, wmax, wmin = kapok.cohopt.pdopt(tm, om, reg=reg, returnall=saveall, **kwargs)
                    else:
                        gammamax, gammamin = kapok.cohopt.pdopt(tm, om, reg=reg, returnall=saveall, **kwargs)
                        
                    temp = np.angle(gammamin*np.conj(gammamax))
                    ind = (np.sign(temp) == np.sign(self.kz(0)))
                    
                    swap = gammamax[ind].copy()
                    gammamax[ind] = gammamin[ind]
                    gammamin[ind] = swap
                    
                    self.pdcoh[0,:,:] = gammamax
                    self.pdcoh[1,:,:] = gammamin
                    
                    if saveall:
                        self.pdcohminor[0,:,:] = gammaminormax
                        self.pdcohminor[1,:,:] = gammaminormin
                        
                        ind = np.dstack((ind,ind,ind))
                        swap = wmax.copy()
                        wmax[ind] = wmin[ind]
                        wmin[ind] = swap[ind]
                        
                        self.pdweights[0,:,:,:] = wmax
                        self.pdweights[1,:,:,:] = wmin
                        
                self.f['pdopt'].attrs['name'] = 'Phase Diversity Coherence Optimization'
                self.f['pdopt'].attrs['regularization_factor'] = reg
                self.f.flush()

        else:
            print('kapok.Scene.opt | Requested coherence optimization method "'+method+'" not recognized.  Aborting.')
            
        return
    
    
    def show(self, imagetype='pauli', pol=0, bl=0, tr=0, vmin=None, vmax=None,
             bounds=None, cmap=None, figsize=None, dpi=125, savefile=None,
             **kwargs):
        """Display images of backscatter, coherence, estimated forest
            heights, etc.
        
        Arguments:
            imagetype (str): String describing what to display, or an array
                containing data to plot.  Possible string options: 'pow' or
                'power' for backscatter image, 'coh' or 'coherence' for
                complex coherence image, 'coh mag' for a coherence magnitude
                image, 'coh ph' for coherence phase image, 'pauli' or 'rgb'
                for Pauli basis RGB composite image, 'inc' or 'incidence'
                for incidence angle image, 'kz' for vertical wavenumber image,
                'dem' for processor DEM heights.  Derived products can be
                displayed by entering their path within the HDF5 file, in the
                form 'products/<name>/<param>', where <name> is the name
                keyword argument originally given to Scene.inv(), and <param>
                is the parameter of interest.  For example, 'products/rvog/hv'
                would display the estimated forest height from a RVoG model
                inversion with the default name.  If '/hv', '/ext', or '/tdf'
                are in the imagetype, the 'products/' group can be omitted.
                The function will assume that is the location of parameters
                with these names.  See kapok.Scene.inv() for more details on
                how the forest model inversion results are stored.  Default:
                'pauli'.
            pol: Polarization identifier, used only for backscatter and
                coherence images.  pol can be an integer from 0 to 2
                (0: HH, 1: HV, 2: VV), a three element list containing the
                polarimetric weights, or a string ('HH', 'HH+VV', 'HH-VV',
                'HV', or 'VV').  Can also be 'high' or 'low' to display the
                phase diversity optimized coherences, if a coherence is
                being displayed.  Default: HH.
            bl (int): Baseline number.  Only used for coherence and kz images.
                Default: 0.
            tr (int): Track number.  Only used for backscatter and Pauli RGB
                images.  Default: 0.
            vmin (float): Min value for colormap.  Only used for some
                image types.  Default value depends on image type and data.
            vmax (float): Max value for colormap.
            bounds (tuple): Bounds containing (azimuth start, azimuth end,
                range start, range end), in that order.  Will plot a subset of
                the image rather than the entire image.  For a full swath
                subset, two element bounds can be given: (azimuth start,
                azimuth end).  Default: Full image.
            cmap: Matplotlib colormap, if you wish to override the default
                colormaps for each image type.  cmap has no effect for some
                image types (e.g., 'pauli' or 'coh').
            figsize (tuple): Figure size argument passed to plotting functions.
                Tuple of (x,y) sizes in inches.  Default is based on data
                shape.
            dpi (int): Dots per inch argument passed to plotting functions.
                Default: 125.
            savefile (str): If specified, the plotted figure is saved under
                this filename.
            **kwargs: Additional keyword arguments provided to plotting
                functions.  Only works if imagetype is input data array.  The
                string image types generally have preset options.
            
        """           
        if bounds is not None:
            if len(bounds) == 2:
                extent = (0,self.dim[1],bounds[1],bounds[0])
            else:
                extent = (bounds[2],bounds[3],bounds[1],bounds[0])
        else:
            extent = None
        
        
        if isinstance(imagetype,str):
            # Which type of image to display?
            if ('/hv' in imagetype) and ((imagetype in self.f) or ('products/'+imagetype in self.f)):
                if imagetype in self.f:
                    data = self.f[imagetype]
                else:
                    data = self.f['products/'+imagetype]
                
                vmin = 0 if vmin is None else vmin
                vmax = 50 if vmax is None else vmax
                cmap = 'Greens' if cmap is None else cmap
                if ('name' in data.attrs) and ('units' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' ('+data.attrs['units']+')', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                elif ('name' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name'], bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=imagetype, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
    
            elif ('/ext' in imagetype) and ((imagetype in self.f) or ('products/'+imagetype in self.f)):
                if imagetype in self.f:
                    data = self.f[imagetype]
                else:
                    data = self.f['products/'+imagetype]
                    
                vmin = 0 if vmin is None else vmin
                vmax = 0.6 if vmax is None else vmax
                cmap = 'viridis' if cmap is None else cmap
                if ('name' in data.attrs) and ('units' in data.attrs):
                    if 'Np/m' in data.attrs['units']:
                        nptodb = 20/np.log(10) # convert to dB/m for display
                        kapok.vis.show_linear(data[:]*nptodb, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' (dB/m)', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                    elif 'dB/m' in data.attrs['units']:
                        kapok.vis.show_linear(data[:], vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' (dB/m)', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                    else:
                        kapok.vis.show_linear(data[:], vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' ('+data.attrs['units']+')', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                elif ('name' in data.attrs):
                    kapok.vis.show_linear(data[:], vmin=vmin, vmax=vmax, cbar_label=data.attrs['name'], bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(data[:], vmin=vmin, vmax=vmax, cbar_label=imagetype, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
    
            elif ('/tdf' in imagetype) and ((imagetype in self.f) or ('products/'+imagetype in self.f)):
                if imagetype in self.f:
                    data = self.f[imagetype]
                else:
                    data = self.f['products/'+imagetype]
                    
                vmin = 0 if vmin is None else vmin
                vmax = 1 if vmax is None else vmax
                cmap = 'afmhot' if cmap is None else cmap
                
                if np.any(np.iscomplex(data)):
                    kapok.vis.show_complex(data, bounds=bounds, cbar=True, cbar_label=data.attrs['name'], figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name'], bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
    
            elif ('products/' in imagetype) and (imagetype in self.f):
                data = self.f[imagetype]
                cmap = 'viridis' if cmap is None else cmap
                kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' ('+data.attrs['units']+')', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('ancillary/' in imagetype) and (imagetype in self.f):
                data = self.f[imagetype]
                cmap = 'viridis' if cmap is None else cmap
                
                if ('name' in data.attrs) and ('units' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' ('+data.attrs['units']+')', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                elif ('name' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name'], bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=imagetype, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
            
            elif ('pow' in imagetype) or ('power' in imagetype):
                cmap = 'gray' if cmap is None else cmap
                kapok.vis.show_power(self.power(pol=pol, tr=tr, bounds=bounds), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('mag' in imagetype) and (('coh' in imagetype) or ('gamma' in imagetype)):
                cmap = 'afmhot' if cmap is None else cmap
                vmin = 0 if vmin is None else vmin
                vmax = 1 if vmax is None else vmax
                
                try:
                    kapok.vis.show_linear(np.abs(self.coh(pol=pol, bl=bl, bounds=bounds)), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label='Coherence Magnitude', figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif (('ph' in imagetype) or ('arg' in imagetype)) and (('coh' in imagetype) or ('gamma' in imagetype)):
                vmin = -np.pi if vmin is None else vmin
                vmax = np.pi if vmax is None else vmax
                
                try:
                    kapok.vis.show_linear(np.angle(self.coh(pol=pol, bl=bl, bounds=bounds)), extent=extent, vmin=vmin, vmax=vmax, cmap='hsv', cbar_label='Phase (radians)', figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif ('coh' in imagetype) or ('gamma' in imagetype):
                try:
                    kapok.vis.show_complex(self.coh(pol=pol, bl=bl, bounds=bounds), extent=extent, cbar=True, figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif ('pauli' in imagetype) or ('rgb' in imagetype):
                i = tr*self.num_pol
                tm = makehermitian(self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol])
                kapok.vis.show_paulirgb(tm, bounds=bounds, vmin=vmin, vmax=vmax, figsize=figsize, dpi=dpi, savefile=savefile, **kwargs)
                
            elif ('inc' in imagetype):
                vmin = 25 if vmin is None else vmin
                vmax = 65 if vmax is None else vmax 
                cmap = 'viridis' if cmap is None else cmap
                kapok.vis.show_linear(np.degrees(self.inc), cmap=cmap, vmin=vmin, vmax=vmax, cbar_label='Incidence Angle (degrees)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('kz' in imagetype) or ('vertical wavenumber' in imagetype):
                cmap = 'viridis' if cmap is None else cmap
                
                kapok.vis.show_linear(self.kz(bl), cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=r'$k_{z}$ (rad/m)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('dem' in imagetype):
                cmap = 'gist_earth' if cmap is None else cmap
                kapok.vis.show_linear(self.dem, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=r'Elevation (m)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('close' in imagetype):
                plt.close('all')
                
            elif (imagetype in self.f):
                data = self.f[imagetype]
                cmap = 'viridis' if cmap is None else cmap
                
                if ('name' in data.attrs) and ('units' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name']+' ('+data.attrs['units']+')', bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                elif ('name' in data.attrs):
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=data.attrs['name'], bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(data, vmin=vmin, vmax=vmax, cbar_label=imagetype, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile)
                
            else:
                print('kapok.Scene.show | Unrecognized image type: "'+str(imagetype)+'".  Aborting.')
                
        elif isinstance(imagetype, (collections.Sequence, np.ndarray)):
            cmap = 'viridis' if cmap is None else cmap
            kapok.vis.show_linear(imagetype, vmin=vmin, vmax=vmax, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile, **kwargs)
            
        elif isinstance(imagetype, h5py.Dataset):
            kapok.vis.show_linear(imagetype[:], vmin=vmin, vmax=vmax, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile, **kwargs)
            
        else:
            print('kapok.Scene.show | Unrecognized image type: "'+str(imagetype)+'".  Aborting.')
            
        return
    
    
    def power(self, pol=0, tr=0, bounds=None):
        """Return the backscattered power, in linear units, for the specified
            polarization and track number.
            
            Arguments:
                pol: The polarization of the desired power.  Can be an
                    integer polarization index (0: HH, 1: HV, 2: VV), or a
                    list containing a polarization weight vector with three
                    complex elements, or a string.  Allowed strings are 'HH',
                    'HV', 'VV', 'HH+VV', and 'HH-VV'.  Default: 0.
                tr (int): Desired track index.  Default: 0.
                bounds (int): If you only wish to calculate the coherence for
                    a subset of the data, specify the subset boundaries in the
                    form: (azmin,azmax,rngmin,rngmax).
                
            Returns:
                pwr (array): An image of backscattered power, in linear units.
            
        """
        if isinstance(pol,str):
            pol = pol.lower()
            
        if isinstance(pol,str) and (pol == 'hh'):
            pol = 0
        elif isinstance(pol,str) and (pol == 'hv'):
            pol = 1
        elif isinstance(pol,str) and (pol == 'vv'):
            pol = 2
        elif isinstance(pol,str) and (pol == 'hh+vv'):
            pol = [np.sqrt(0.5), 0, np.sqrt(0.5)]
        elif isinstance(pol,str) and (pol == 'hh-vv'):
            pol = [np.sqrt(0.5), 0, -np.sqrt(0.5)]
        
        if pol in range(self.num_pol):
            i = tr*self.num_pol + pol
            if bounds is None:
                pwr = self.cov[:,:,i,i]
            else:
                pwr = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i,i]
        elif len(pol) == self.num_pol:
            i = tr*self.num_pol
            
            if bounds is None:
                tm = self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol]
                tm = makehermitian(tm)
                
                wimage = np.ones((self.dim[0],self.dim[1],3,3),dtype='complex')
                wimage[:,:] = np.array([[pol[0]*pol[0],pol[0]*pol[1],pol[0]*pol[2]],
                                             [pol[1]*pol[0],pol[1]*pol[1],pol[1]*pol[2]],
                                             [pol[2]*pol[0],pol[2]*pol[1],pol[2]*pol[2]]])                                       
                pwr = np.sum(tm*wimage, axis=(2,3))
            else:
                tm = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i:i+self.num_pol,i:i+self.num_pol]
                tm = makehermitian(tm)
                
                wimage = np.ones(tm.shape,dtype='complex')
                wimage[:,:] = np.array([[pol[0]*pol[0],pol[0]*pol[1],pol[0]*pol[2]],
                                             [pol[1]*pol[0],pol[1]*pol[1],pol[1]*pol[2]],
                                             [pol[2]*pol[0],pol[2]*pol[1],pol[2]*pol[2]]])                                       
                pwr = np.sum(tm*wimage, axis=(2,3))                
        else:
            print('kapok.Scene.power | Unrecognized polarization identifier.  Returning None.')
            pwr = None
            
        if np.any(np.iscomplex(pwr)):
            return pwr
        else:
            return np.real(pwr)
    
    
    def coh(self, pol=0, polb=None, bl=0, pix=None, bounds=None, **kwargs):
        """Return the complex coherence for a specified polarization and
            baseline.
            
            Arguments:
                pol: The polarization of the desired coherence.  Can be an
                    integer polarization index (0: HH, 1: HV, 2: VV), or a
                    list containing a polarization weight vector with three
                    complex elements, or a string.  Allowed strings are 'HH',
                    'HV', 'VV', 'HH+VV', and 'HH-VV'.  Can also set to 'high'
                    or 'pdhigh' or 'low' or 'pdlow' if you want the optimized
                    coherences.  In this case, the polb argument will be
                    ignored.  Note that the high/low ordering is an
                    assumption, and is not necessarily true unless the
                    ground phase has been determined, and the
                    coherences reordered as a result.  Another option
                    for this argument is 'ground', in which case, the
                    estimated complex ground coherence from the line fit of
                    the optimized coherences will be returned.  If pol is
                    'ground', you can set the desired ground coherence
                    magnitude with the 'groundmag' keyword argument, which
                    will be passed along to kapok.topo.groundsolver through
                    **kwargs.  Default: 0.
                polb: If different master and slave track polarizations are
                    desired, specify the slave polarization here.  Use same
                    form (int, str, or list) as pol.  Default: None (polb=pol).
                bl (int): Desired baseline index.  Default: 0.
                pix (int): If you only wish to calculate the coherence for a
                single pixel, specify a tuple with the (azimuth,range) indices
                of the pixel here.
                bounds (tuple): If you only wish to calculate the coherence for
                    a subset of the data, specify the subset boundaries in the
                    form: (azmin,azmax,rngmin,rngmax).  If bounds has only
                    two elements, it will be assumed to be (azmin,azmax), with
                    the returned data spanning the full width of the swath.
                    Note: This keyword overrides the pix keyword, if both
                    are given.
                **kwargs: Extra keyword arguments.
                
            Returns:
                coh (array): A complex coherence image.
            
        """
        if bounds is not None:
            if len(bounds) == 2:
                bounds = (bounds[0], bounds[1], 0, self.dim[1])
        
        if polb is None:
            polb = pol
            
        if isinstance(pol,str):
            pol = pol.lower()
        
        if isinstance(pol,str) and (pol == 'hh'):
            pol = 0
        elif isinstance(pol,str) and (pol == 'hv'):
            pol = 1
        elif isinstance(pol,str) and (pol == 'vv'):
            pol = 2
        elif isinstance(pol,str) and (pol == 'hh+vv'):
            pol = [np.sqrt(0.5), 0, np.sqrt(0.5)]
        elif isinstance(pol,str) and (pol == 'hh-vv'):
            pol = [np.sqrt(0.5), 0, -np.sqrt(0.5)]
        elif isinstance(pol,str) and (pol == 'ground'):
            import kapok.topo
            if bounds is None:
                if self.num_baselines > 1:                
                    ground = kapok.topo.groundsolver(self.pdcoh[bl], kz=self.kz(bl), silent=True, **kwargs)
                else:
                    ground = kapok.topo.groundsolver(self.pdcoh[:], kz=self.kz(bl), silent=True, **kwargs)
            else:
                if self.num_baselines > 1:
                    pdcoh = self.pdcoh[bl,:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
                else:
                    pdcoh = self.pdcoh[:,bounds[0]:bounds[1],bounds[2]:bounds[3]]

                kz = self.kz(bl)[bounds[0]:bounds[1],bounds[2]:bounds[3]]                    
                ground = kapok.topo.groundsolver(pdcoh, kz=kz, silent=True, **kwargs)
                return ground
            
            if pix is not None:
                ground = ground[pix[0],pix[1]]
            
            return ground
        
        
        if isinstance(polb,str):
            polb = polb.lower()
            
        if isinstance(polb,str) and (polb == 'hh'):
            polb = 0
        elif isinstance(polb,str) and (polb == 'hv'):
            polb = 1
        elif isinstance(polb,str) and (polb == 'vv'):
            polb = 2
        elif isinstance(polb,str) and (polb == 'hh+vv'):
            polb = [np.sqrt(0.5), 0, np.sqrt(0.5)]
        elif isinstance(polb,str) and (polb == 'hh-vv'):
            polb = [np.sqrt(0.5), 0, -np.sqrt(0.5)]
        
        
        if isinstance(pol,str) and ('high' in pol):
            if self.pdcoh is None:
                coh = None
            elif (pix is None) and (bounds is None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl, 0]
            elif (pix is None) and (bounds is None):
                coh = self.pdcoh[0]
            elif (bounds is not None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl,0,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            elif (bounds is not None):
                coh = self.pdcoh[0,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            elif self.num_baselines > 1:
                coh = self.pdcoh[bl,0,pix[0],pix[1]]
            else:
                coh = self.pdcoh[0,pix[0],pix[1]]
        elif isinstance(pol,str) and ('low' in pol):
            if self.pdcoh is None:
                coh = None
            elif (pix is None) and (bounds is None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl, 1]
            elif (pix is None) and (bounds is None):
                coh = self.pdcoh[1]
            elif (bounds is not None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl,1,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            elif (bounds is not None):
                coh = self.pdcoh[1,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            elif self.num_baselines > 1:
                coh = self.pdcoh[bl,1,pix[0],pix[1]]
            else:
                coh = self.pdcoh[1,pix[0],pix[1]]
        elif not isinstance(pol, (collections.Sequence, np.ndarray)) and (pol in range(self.num_pol)):
            i,j = mb_cov_index(bl, pol=pol, pol2=polb, n_pol=self.num_pol)
            if (pix is None) and (bounds is None):
                coh = self.cov[:,:,i,j] / np.sqrt(np.abs(self.cov[:,:,i,i]*self.cov[:,:,j,j]))
            elif (bounds is not None):
                coh = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i,j] / np.sqrt(np.abs(self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i,i]*self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],j,j]))                
            else:
                coh = self.cov[pix[0],pix[1],i,j] / np.sqrt(np.abs(self.cov[pix[0],pix[1],i,i]*self.cov[pix[0],pix[1],j,j]))
        elif isinstance(pol, (collections.Sequence, np.ndarray)) and (len(pol) == self.num_pol):
            i,j = mb_cov_index(bl, pol=0, n_pol=self.num_pol)
            
            if (pix is None) and (bounds is None):
                t11 = self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol]
                t22 = self.cov[:,:,j:j+self.num_pol,j:j+self.num_pol]
                om = self.cov[:,:,i:i+self.num_pol,j:j+self.num_pol]
                
                t11 = makehermitian(t11)
                t22 = makehermitian(t22)
                
                wimage = np.ones(t11.shape,dtype='complex')
                
                wimage[:,:] = np.array([[pol[0]*pol[0],pol[0]*pol[1],pol[0]*pol[2]],
                                             [pol[1]*pol[0],pol[1]*pol[1],pol[1]*pol[2]],
                                             [pol[2]*pol[0],pol[2]*pol[1],pol[2]*pol[2]]])                                       
                t11 = np.sum(t11*wimage, axis=(2,3))
    
                wimage[:,:] = np.array([[polb[0]*polb[0],polb[0]*polb[1],polb[0]*polb[2]],
                                             [polb[1]*polb[0],polb[1]*polb[1],polb[1]*polb[2]],
                                             [polb[2]*polb[0],polb[2]*polb[1],polb[2]*polb[2]]])
                t22 = np.sum(t22*wimage, axis=(2,3))
    
                
                wimage[:,:] = np.array([[pol[0]*polb[0],pol[0]*polb[1],pol[0]*polb[2]],
                                             [pol[1]*polb[0],pol[1]*polb[1],pol[1]*polb[2]],
                                             [pol[2]*polb[0],pol[2]*polb[1],pol[2]*polb[2]]])
                om = np.sum(om*wimage, axis=(2,3))
                
                coh = om / np.sqrt(np.abs(t11*t22))
            elif (bounds is not None):
                t11 = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i:i+self.num_pol,i:i+self.num_pol]
                t22 = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],j:j+self.num_pol,j:j+self.num_pol]
                om = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],i:i+self.num_pol,j:j+self.num_pol]
                
                t11 = makehermitian(t11)
                t22 = makehermitian(t22)
                
                wimage = np.ones(t11.shape,dtype='complex')
                
                wimage[:,:] = np.array([[pol[0]*pol[0],pol[0]*pol[1],pol[0]*pol[2]],
                                             [pol[1]*pol[0],pol[1]*pol[1],pol[1]*pol[2]],
                                             [pol[2]*pol[0],pol[2]*pol[1],pol[2]*pol[2]]])                                       
                t11 = np.sum(t11*wimage, axis=(2,3))
    
                wimage[:,:] = np.array([[polb[0]*polb[0],polb[0]*polb[1],polb[0]*polb[2]],
                                             [polb[1]*polb[0],polb[1]*polb[1],polb[1]*polb[2]],
                                             [polb[2]*polb[0],polb[2]*polb[1],polb[2]*polb[2]]])
                t22 = np.sum(t22*wimage, axis=(2,3))
    
                
                wimage[:,:] = np.array([[pol[0]*polb[0],pol[0]*polb[1],pol[0]*polb[2]],
                                             [pol[1]*polb[0],pol[1]*polb[1],pol[1]*polb[2]],
                                             [pol[2]*polb[0],pol[2]*polb[1],pol[2]*polb[2]]])
                om = np.sum(om*wimage, axis=(2,3))
                
                coh = om / np.sqrt(np.abs(t11*t22))
            else:
                t11 = self.cov[pix[0],pix[1],i:i+self.num_pol,i:i+self.num_pol]
                t22 = self.cov[pix[0],pix[1],j:j+self.num_pol,j:j+self.num_pol]
                om = self.cov[pix[0],pix[1],i:i+self.num_pol,j:j+self.num_pol]
                
                t11 = makehermitian(t11)
                t22 = makehermitian(t22)
                                
                wimage = np.array([[pol[0]*pol[0],pol[0]*pol[1],pol[0]*pol[2]],
                                             [pol[1]*pol[0],pol[1]*pol[1],pol[1]*pol[2]],
                                             [pol[2]*pol[0],pol[2]*pol[1],pol[2]*pol[2]]])                                       
                t11 = np.sum(t11*wimage)
    
                wimage = np.array([[polb[0]*polb[0],polb[0]*polb[1],polb[0]*polb[2]],
                                             [polb[1]*polb[0],polb[1]*polb[1],polb[1]*polb[2]],
                                             [polb[2]*polb[0],polb[2]*polb[1],polb[2]*polb[2]]])
                t22 = np.sum(t22*wimage)
    
                
                wimage = np.array([[pol[0]*polb[0],pol[0]*polb[1],pol[0]*polb[2]],
                                             [pol[1]*polb[0],pol[1]*polb[1],pol[1]*polb[2]],
                                             [pol[2]*polb[0],pol[2]*polb[1],pol[2]*polb[2]]])
                om = np.sum(om*wimage)
                
                coh = om / np.sqrt(np.abs(t11*t22))
        else:
            print('kapok.Scene.coh | Unrecognized polarization identifier.  Returning None.')
            coh = None
            
        return coh
        
        
    def region(self, az=None, rng=None, mode='basic', bl=0, savefile=None,
               **kwargs):
        """Coherence region plotting.
        
            Creates a plot showing the coherence region in the complex plane.
            There are two modes.  'basic', the default, simply plots the
            observed coherence region, the standard Lexicographic and Pauli
            coherences, the optimized coherences, the line fit, and
            the estimated ground coherence.  'interactive' creates a 
            coherence region with a UI which allows the user to specify
            values for the RVoG model parameters and observe the effect
            they have on the modelled coherences.
            
            Arguments:
                az (int): Azimuth index of the pixel to plot.
                rng (int): Range index of the pixel to plot.
                mode (str): Mode setting, either 'basic' or 'interactive'.
                    Default: 'basic' if az and rng are specified.
                    'interactive' otherwise.
                bl (int): Desired baseline index.
                savefile (str): Path and filename to save the plot.  Only
                    valid for mode == 'basic'.
                **kwargs: Additional keyword arguments to pass to the
                    kapok.region.cohregion and kapok.region.rvogregion
                    functions (see those function headers for details).
        
        """
        import kapok.region
        
        if (mode == 'basic') and (az is not None) and (rng is not None):
            kapok.region.cohregion(self, az, rng, bl=bl, savefile=savefile, **kwargs)
        else:
            kapok.region.rvogregion(self, az=az, rng=rng, bl=bl, **kwargs)
            
        return


    def geo(self, data, outfile, outformat='ENVI', resampling='pyresample',
            nodataval=None, tr=2.7777778e-4, **kwargs):
        """Output a geocoded raster.
        
            Resampling from radar coordinates to latitude/longitude using
            either the pyresample library, or gdalwarp (see description of
            'resampling' keyword argument, below).
            
            Arguments:
                data: Either a 2D array containing the data to geocode, or a
                    string identifying an HDF5 dataset in the file.
                    If data is an array, it should have type float32.
                    If not, it will be converted to it.  If resampling of
                    complex-valued parameters is needed, geocode the real and
                    imaginary parts separately using this function.
                outfile (str): The destination filename for the geocoded file.
                outformat (str): String identifying an output format
                    recognized by GDAL.  Default is 'ENVI'.  Other options
                    include 'GTiff' or 'KEA', etc.  For reference, see
                    http://www.gdal.org/formats_list.html.
                resampling (str): String identifying the resampling method.
                    The default option is 'pyresample', which uses the
                    pyresample Python library as implemented in the 
                    kapok.geo.radar2ll_pr() function.  Other options use
                    GDAL with the gdalwarp command line tool, as implemented
                    in the kapok.geo.radar2ll_gdal() function.  Possible GDAL
                    resampling options include 'near', 'bilinear', 'cubic',
                    'average', and others.  For reference and more options,
                    see http://www.gdal.org/gdalwarp.html.
                    Default: 'pyresample'.
                nodataval:  No data value for the output raster.  This will be the
                    value of the raster for all pixels outside the input data
                    extent.  Default: None.
                tr (float): Set output file resolution (in degrees).  Can be set
                    to a tuple to set (longitude, latitude) resolution separately.
                    If you are using a gdalwarp-based resampling method, tr
                    can be set to None in order for gdalwarp to choose the
                    output resolution automatically based on the input data.
                    Default: 2.7777778e-4 (1 arc second).
            
        """        
        if isinstance(data, str):
            data = self.get(data)
        
        outpath, outfile = os.path.split(outfile)
        
        if 'pyresample' in resampling:
            try:
                import pyresample as pr
            except ImportError:
                print('kapok.Scene.geo | "pyresample" chosen as resampling method, but pyresample Python library cannot be imported!  Make sure it is installed, or use the gdalwarp-based geocoding implementation by changing the resampling keyword.  Aborting.')
                return
            print('kapok.Scene.geo | Performing geocoding using pyresample library. ('+time.ctime()+')')
            kapok.geo.radar2ll_pr(outpath, outfile, data, self.lat[:], self.lon[:],
                               outformat=outformat, nodataval=nodataval, tr=tr)
        else:
            print('kapok.Scene.geo | Performing geocoding using gdalwarp. ('+time.ctime()+')')
            kapok.geo.radar2ll_gdal(outpath, outfile, data, self.lat[:], self.lon[:],
                               outformat=outformat, resampling=resampling,
                               nodataval=nodataval, tr=tr)
        
        return


    def ingest(self, file, name, attrname=None, attrunits='', overwrite=False):
        """Ingest ancillary raster data in the WGS84 Geographic coordinate
            system, reproject into radar coordinates, and save in the Kapok
            HDF5 file.
        
            Allows the user to import ancillary raster data in ENVI format
            such as lidar, external DEMs, etc.  This external raster data will
            be resampled to the radar coordinates using bilinear interpolation,
            then saved to the HDF5 file as datasets with the same dimensions as
            the radar data. The ingested data can then be compared to the
            radar-derived products or used in guided inversion functions, etc.
            
            Data will be stored in the HDF5 file under 'ancillary/<name>', where
            <name> is the string given in the name argument to this function.
            
            N.B. The data to import must be in WGS84 Geographic
            (latitude, longitude) coordinates.
            
            Arguments:
                file (str): Path and filename to the external raster data (in
                    ENVI format).  Will be loaded in GDAL.
                name (str): Name of the HDF5 dataset which will be created to
                    store the ingested data.
                attrname (str): Name which will be put into a 'name' attribute
                    of the dataset.  Will be shown when displaying the data
                    using Scene.show(), etc.  Default: Same as name.
                attrunits (str): Units of the data.  Will be shown on plots of
                    the data using Scene.show(), etc.
                overwrite (bool): Set to True to overwrite an already existing
                    HDF5 dataset, if one already exists under the same name as
                    the name input argument.  Default: False.
                
            Returns:
                data: A link to the newly created HDF5 dataset containing the
                    ingested data.
        
        """
        from osgeo import gdal
        
        # Check If Group Exists:
        if ('ancillary/'+name in self.f) and overwrite:
            del self.f['ancillary/'+name]
        elif ('ancillary/'+name in self.f) and (overwrite == False):
            print('kapok.Scene.ingest | Ancillary data in "ancillary/'+name+'" already exists.  If you wish to replace it, set overwrite keyword.  Aborting.')
            return None

        # Load the data using GDAL and resample to radar coordinates:
        data = gdal.Open(file, gdal.GA_ReadOnly)
        geodata = data.GetGeoTransform()
        
        origin = (geodata[3], geodata[0])
        spacing = (geodata[5], geodata[1])       
        
        data = kapok.geo.ll2radar(data.ReadAsArray(), origin, spacing, self.lat[:], self.lon[:])
        
        # Create the HDF5 dataset:
        if np.any(np.iscomplex(data)):
            data = self.f.create_dataset('ancillary/'+name, data=data, dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
        else:
            data = self.f.create_dataset('ancillary/'+name, data=data, dtype='float32', compression=self.compression, compression_opts=self.compression_opts)            
            
        if attrname is None:
            attrname = name
        data.attrs['name'] = attrname
        data.attrs['units'] = attrunits
        
        self.f.flush()
        
        return data


    def subset(self, outfile, bounds=None, tracks=None, overwrite=False):
        """Subset the Kapok Scene (either spatially or by track), and
            create a new HDF5 file containing the subsetted data.
            
            Note that HDF5 datasets in the 'products/' and
            'ancillary/' groups will not be transferred.  They will
            need to be recreated/reimported.
            
            Arguments:
                outfile (str): Path and filename for the new subsetted
                    Kapok HDF5 file.
                bounds (tuple): Bounds of the subset, in the order:
                    (az min, az max, rng min, rng max).  Can also be
                    in the form of a two element tuple: (az min,
                    az max).  In this case, full range extent will be
                    preserved.  Default: None (keep full data extents).
                tracks (tuple): List of track indices to keep.  Tracks
                    not specified will be discarded.  Default: None
                    (keep all tracks).
                overwrite (bool): Set to True to overwrite outfile if
                    it already exists.  Default: False.
                    
            Returns:
                subsetted (Scene object): Newly created Kapok Scene object
                    containing the subsetted data.
            
        """
        # Create the new HDF5 file.        
        if overwrite:
            try:
                f = h5py.File(outfile,'w')
            except:
                print('kapok.Scene.subset | Cannot create new HDF5 file. Check if path is valid. Aborting.')
                return None 
        else:
            try:
                f = h5py.File(outfile,'x')
            except:
                print('kapok.Scene.subset | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
                return None
        
        if (bounds is None) and (tracks is None):
            print('kapok.Scene.subset | No subset bounds or subset tracks specified. No subsetting performed. Aborting.')
            return None
            
        bounds_set = True
        if bounds is None:
            bounds_set = False
            bounds = (0, self.dim[0], 0, self.dim[1])
        elif len(bounds) == 2:
            bounds = (bounds[0], bounds[1], 0, self.dim[1])
            
        tracks_set = True
        if tracks is None:
            tracks_set = False
            tracks = np.linspace(0, self.num_tracks-1, num=self.num_tracks)
        else:
            tracks = np.array(tracks)
            
        print('kapok.Scene.subset | Creating subset. ('+time.ctime()+')')
        
        
        # Create subsetted covariance matrix.
        num_cov_elements = len(tracks) * self.num_pol
        cov_subset = f.create_dataset('cov', (bounds[1]-bounds[0], bounds[3]-bounds[2], num_cov_elements, num_cov_elements), dtype='complex64', compression=self.compression, shuffle=True, compression_opts=self.compression_opts)
        
        
        # Create subsetted kz dataset.
        kz = f.create_dataset('kz', (len(tracks), bounds[1]-bounds[0], bounds[3]-bounds[2]), dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
        
        
        # Copy polarimetric T covariance matrices and kz values.
        for m, tr in enumerate(tracks):
            row = tr * self.num_pol
            row_new = m * self.num_pol
            cov_subset[:,:,row_new:row_new+self.num_pol,row_new:row_new+self.num_pol] = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],row:row+self.num_pol,row:row+self.num_pol]
            kz[m] = self.kza[tr,bounds[0]:bounds[1],bounds[2]:bounds[3]]
            
        # Copy interferometric Omega covariance matrices.
        for m, tr1 in enumerate(tracks[0:-1]):
            for n, tr2 in enumerate(tracks[m+1:]):
                row = tr1 * self.num_pol
                col = tr2 * self.num_pol
                row_new = m * self.num_pol
                col_new = (m+n+1) * self.num_pol
                cov_subset[:,:,row_new:row_new+self.num_pol,col_new:col_new+self.num_pol] = self.cov[bounds[0]:bounds[1],bounds[2]:bounds[3],row:row+self.num_pol,col:col+self.num_pol]
                
        f.flush()
        
        # Copy covariance matrix attributes.
        for key in self.f['cov'].attrs:
            cov_subset.attrs[key] = self.f['cov'].attrs[key]
            
        # Copy kz attributes.
        for key in self.f['kz'].attrs:
            kz.attrs[key] = self.f['kz'].attrs[key]            
        
        # Copy latitudes and attributes.
        lat = f.create_dataset('lat', data=self.f['lat'][bounds[0]:bounds[1],bounds[2]:bounds[3]], dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
        for key in self.f['lat'].attrs:
            lat.attrs[key] = self.f['lat'].attrs[key]

        # Copy longitudes and attributes.
        lon = f.create_dataset('lon', data=self.f['lon'][bounds[0]:bounds[1],bounds[2]:bounds[3]], dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
        for key in self.f['lon'].attrs:
            lon.attrs[key] = self.f['lon'].attrs[key]

        # Copy DEM heights and attributes.
        dem = f.create_dataset('dem', data=self.f['dem'][bounds[0]:bounds[1],bounds[2]:bounds[3]], dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
        for key in self.f['dem'].attrs:
            dem.attrs[key] = self.f['dem'].attrs[key]

        # Copy incidence angles and attributes.
        inc = f.create_dataset('inc', data=self.f['inc'][bounds[0]:bounds[1],bounds[2]:bounds[3]], dtype='float32', compression=self.compression, compression_opts=self.compression_opts)
        for key in self.f['inc'].attrs:
            inc.attrs[key] = self.f['inc'].attrs[key]

        # Copy main attributes.
        for key in self.f.attrs:
            f.attrs[key] = self.f.attrs[key]
        
        # Update track attributes.
        if tracks_set:
            f.attrs['tracks'] = self.f.attrs['tracks'][tracks]
            f.attrs['num_tracks'] = len(tracks)
            f.attrs['num_baselines'] = mb_num_baselines(len(tracks))
        
        # Update subset attributes, if necessary.
        if bounds_set:
            f.attrs['subset'] = True
            azstart_slc = (bounds[0]*self.ml_window[0]) + self.f.attrs['azimuth_bounds_slc'][0]
            azend_slc = (bounds[1]*self.ml_window[0]) + self.f.attrs['azimuth_bounds_slc'][0]
            rngstart_slc = (bounds[2]*self.ml_window[1]) + self.f.attrs['range_bounds_slc'][0]
            rngend_slc = (bounds[3]*self.ml_window[1]) + self.f.attrs['range_bounds_slc'][0]
            f.attrs['azimuth_bounds_slc'] = [azstart_slc, azend_slc]
            f.attrs['range_bounds_slc'] = [rngstart_slc, rngend_slc]
            
        f.flush()
        
        # Copy phase diversity coherence optimization datasets, if they exist.
        if 'pdopt/coh' in self.f:        
            pdcoh = f.create_dataset('pdopt/coh', (f.attrs['num_baselines'], 2, bounds[1]-bounds[0], bounds[3]-bounds[2]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
            
            f['pdopt'].attrs['name'] = 'Phase Diversity Coherence Optimization'
            f['pdopt'].attrs['regularization_factor'] = self.f['pdopt'].attrs['regularization_factor']
            
            for bl in range(f.attrs['num_baselines']):
                tr_new = mb_tr_index(bl)
                tr_old = (tracks[tr_new[0]], tracks[tr_new[1]])
                bl_old = mb_bl_index(tr_old[0], tr_old[1])
                pdcoh[bl] = self.pdcoh[bl_old,:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
        
        if 'pdopt/weights' in self.f:           
            pdweights = f.create_dataset('pdopt/weights', (f.attrs['num_baselines'], 2, bounds[1]-bounds[0], bounds[3]-bounds[2], 3), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
            
            for bl in range(f.attrs['num_baselines']):
                tr_new = mb_tr_index(bl)
                tr_old = (tracks[tr_new[0]], tracks[tr_new[1]])
                bl_old = mb_bl_index(tr_old[0], tr_old[1])
                pdweights[bl] = self.pdweights[bl_old,:,bounds[0]:bounds[1],bounds[2]:bounds[3],:]
            
        if 'pdopt/cohminor' in self.f:
            pdcohminor = f.create_dataset('pdopt/cohminor', (f.attrs['num_baselines'], 2, bounds[1]-bounds[0], bounds[3]-bounds[2]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
            
            for bl in range(f.attrs['num_baselines']):
                tr_new = mb_tr_index(bl)
                tr_old = (tracks[tr_new[0]], tracks[tr_new[1]])
                bl_old = mb_bl_index(tr_old[0], tr_old[1])
                pdcohminor[bl] = self.pdcohminor[bl_old,:,bounds[0]:bounds[1],bounds[2]:bounds[3]]
        
        
        # Close the file, then return it as a Scene object.
        f.close()
        print('kapok.Scene.subset | Complete. ('+time.ctime()+')')
        return kapok.Scene(outfile)