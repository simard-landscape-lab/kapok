# -*- coding: utf-8 -*-
"""Kapok Main Module

    Core Kapok module containing Scene class definition and methods.  A Scene
    object contains a PolInSAR dataset including covariance matrix, incidence
    angle, kz, latitude, longitude, processor DEM, and metadata.
    
    Methods available for data visualization, coherence optimization, and
    forest model inversion.

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
from kapok.lib import mb_cov_index, makehermitian


screen_height = 1000 # Set to screen resolution height.  Determines figure sizes.


class Scene(object):
    """Scene object containing a PolInSAR dataset, and methods for viewing
        and processing the data.
        
    """
    
    def __init__(self, file):
        """Scene initialization method.
        
        Arguments:
            file (str): Path and filename to a previously saved kapok HDF5
                PolInSAR scene.
        
        """
        # Load the HDF5 file.  If it can't be loaded, abort.
        try:
            self.f = h5py.File(file,'r+')
        except:
            print('kapok.Scene | Cannot load specified HDF5 file: "'+file+'".  Ensure file exists.  Aborting.')
            self.f = None
            return
            
        # Easy access to datasets:
        self.cov = self.f['cov']
        
        self.lat = self.f['lat']
        self.lon = self.f['lon']
        self.dem = self.f['dem']
        
        self.kz = self.f['kz']
        self.inc = self.f['inc']
        
        if 'pdopt/coh' in self.f:
            self.pdcoh = self.f['pdopt/coh']
        else:
            self.pdcoh = None
        
        if 'pdopt/weights' in self.f:
            self.pdweights = self.f['pdopt/weights']
        else:
            self.pdweights = None
            
        if 'products' in self.f:
            self.products = self.f['products']
        
        
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
        
        
    def __del__(self):
        """Scene destructor.  Closes the HDF5 file."""
        if self.f is not None:
            self.f.close()
        
        
    def get(self, path):
        """Returns a specified dataset from the HDF5 file.
        
        Arguments:
            path (str): Path and name to an HDF5 dataset within the Scene
                file.  Note that shorts can be used if you wish to access
                a dataset within the 'products/' or 'ancillary/' groups.
                These group names can be omitted and the function will check
                within them for the given dataset name.
                
        Returns:
            data (array): The desired HDF5 dataset, in the form of a NumPy
                array.
                
        """
        if path in self.f:
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
    
    
    def inv(self, method='rvog', name=None, desc=None, overwrite=False, bl=0,
            tdf=None, epsilon=0.4, groundmag=None, ext=None, mu=0, mask=None,
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
            bl (int): Baseline index specifying which baseline to invert, if
                scene contains multiple baselines.  Default: 0.
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
            mask (array): Boolean array of mask values.  Only pixels where
                (mask == True) will be inverted.  Defaults to array of ones
                (all pixels inverted).
            **kwargs: Additional keyword arguments passed to the model
                inversion functions, if desired.  See model inversion
                function headers for more details.  Default: None.
                
        Returns:
            result (hdf5 group): Link to the HDF5 group containing the
                optimized model parameters.
        
        """
        if name is None:
            name = method
            
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

            if self.num_baselines > 1:
                hv = kapok.sinc.sincinv(gammav, self.kz[bl], tdf=tdf, mask=mask, **kwargs)
            else:
                hv = kapok.sinc.sincinv(gammav, self.kz[:], tdf=tdf, mask=mask, **kwargs)
            
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
            
            if 'pdopt/coh' not in self.f:
                print('kapok.Scene.inv | Run phase diversity coherence optimization before performing sinc and phase difference inversion.  Aborting.')
                result = None
            else:                
                if self.num_baselines > 1:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[bl], kz=self.kz[bl], groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[bl,1],self.pdcoh[bl,0])
                    hv = kapok.sinc.sincphaseinv(coh_high, np.angle(ground), self.kz[bl], epsilon=epsilon, tdf=tdf, mask=mask, **kwargs)
                else:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[:], kz=self.kz[:], groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[1],self.pdcoh[0])
                    hv = kapok.sinc.sincphaseinv(coh_high, np.angle(ground), self.kz, epsilon=epsilon, tdf=tdf, mask=mask, **kwargs)
                
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
            
            if 'pdopt/coh' not in self.f:
                print('kapok.Scene.inv | Run phase diversity coherence optimization before performing RVoG inversion.  Aborting.')
                result = None
            else:                
                if self.num_baselines > 1:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[bl], kz=self.kz[bl], groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[bl,1],self.pdcoh[bl,0])
                    hv, exttdf, converged = kapok.rvog.rvoginv(coh_high, ground, self.inc, self.kz[bl], ext=ext, tdf=tdf, mu=mu,
                        mask=mask, **kwargs)
                else:
                    ground, groundalt, volindex = kapok.topo.groundsolver(self.pdcoh[:], kz=self.kz[:], groundmag=groundmag, returnall=True)
                    coh_high = np.where(volindex,self.pdcoh[1],self.pdcoh[0])
                    hv, exttdf, converged = kapok.rvog.rvoginv(coh_high, ground, self.inc, self.kz, ext=ext, tdf=tdf, mu=mu,
                        mask=mask, **kwargs)
                
                if desc is None:
                    result.attrs['desc'] = 'Random Volume over Ground model.'
    
                result.attrs['model'] = 'rvog'
                result.attrs['baseline'] = bl
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
        return result
    
       
    def opt(self, method='pdopt', saveweights=False, overwrite=False):
        """Coherence optimization.
        
        Perform coherence optimization, and save the results to the HDF5 file.
        Currently, phase diversity coherence optimization is the only method
        supported.  The coherences will be saved in 'pdopt/coh'.  The
        coherence weight vectors, if requested, will be saved in
        'pdopt/weights'.
            
        Arguments:
            method (str): Desired optimization algorithm.  Currently 'pdopt'
                (phase diversity) is the only method supported.
            saveweights (bool): True/False flag, specifies whether to
                save the polarization weight vectors for the optimized
                coherences.
            overwrite (bool): True/False flag that determines whether to
                overwrite the current coherences.  If False, will abort if
                the coherences already exist.
        
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
                    

                if self.num_baselines > 1: # Multiple Baselines
                    self.pdcoh = self.f.create_dataset('pdopt/coh', (self.num_baselines, 2, self.dim[0], self.dim[1]), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                
                    if saveweights:
                        self.pdweights = self.f.create_dataset('pdopt/weights', (self.num_baselines, 2, self.dim[0], self.dim[1], 3), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)

                    for bl in range(self.num_baselines):
                        # Get Omega and T matrices:
                        row, col = mb_cov_index(bl, 0, n_pol=self.num_pol)
                        tm = 0.5*(self.cov[:,:,row:row+self.num_pol,row:row+self.num_pol] + self.cov[:,:,col:col+self.num_pol,col:col+self.num_pol])
                        tm = makehermitian(tm)
                        om = self.cov[:,:,row:row+self.num_pol,col:col+self.num_pol]
                        
                        print('kapok.cohopt.pdopt | Beginning phase diversity coherence optimization for baseline index '+str(bl)+'. ('+time.ctime()+')')
                        if saveweights:                     
                            gammamax, gammamin, wmax, wmin = kapok.cohopt.pdopt(tm, om, returnweights=saveweights)
                        else:
                            gammamax, gammamin = kapok.cohopt.pdopt(tm, om, returnweights=saveweights)
                            
                        temp = np.angle(gammamin*np.conj(gammamax))
                        ind = (np.sign(temp) == np.sign(self.kz[bl]))
                        
                        swap = gammamax[ind].copy()
                        gammamax[ind] = gammamin[ind]
                        gammamin[ind] = swap
                        
                        self.pdcoh[bl,0,:,:] = gammamax
                        self.pdcoh[bl,1,:,:] = gammamin
                        
                        if saveweights:
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
                    if saveweights:
                        self.pdweights = self.f.create_dataset('pdopt/weights', (2, self.dim[0], self.dim[1], 3), dtype='complex64', compression=self.compression, compression_opts=self.compression_opts)
                        gammamax, gammamin, wmax, wmin = kapok.cohopt.pdopt(tm, om, returnweights=saveweights)
                    else:
                        gammamax, gammamin = kapok.cohopt.pdopt(tm, om, returnweights=saveweights)
                        
                    temp = np.angle(gammamin*np.conj(gammamax))
                    ind = (np.sign(temp) == np.sign(self.kz))
                    
                    swap = gammamax[ind].copy()
                    gammamax[ind] = gammamin[ind]
                    gammamin[ind] = swap
                    
                    self.pdcoh[0,:,:] = gammamax
                    self.pdcoh[1,:,:] = gammamin
                    
                    if saveweights:
                        ind = np.dstack((ind,ind,ind))
                        swap = wmax.copy()
                        wmax[ind] = wmin[ind]
                        wmin[ind] = swap[ind]
                        
                        self.pdweights[0,:,:,:] = wmax
                        self.pdweights[1,:,:,:] = wmin
                        
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
        # Set up figure size.        
        ysize = screen_height*0.8/dpi
        
        if (bounds is None) and (figsize is None):
            xsize = ysize*self.dim[1]/self.dim[0] + 2
            figsize=(xsize, ysize)
        elif figsize is None:
            if len(bounds) == 2:
                bounds = (bounds[0], bounds[1], 0, self.dim[1])
            xsize = ysize*(bounds[3]-bounds[2])/(bounds[1]-bounds[0]) + 2
            figsize=(xsize, ysize)
        
        
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
                           
            elif ('pow' in imagetype) or ('power' in imagetype):
                vmin = -25 if vmin is None else vmin
                vmax = -3 if vmax is None else vmax
                cmap = 'gray' if cmap is None else cmap
                kapok.vis.show_power(self.power(pol=pol, tr=tr), bounds=bounds, cmap=cmap, vmin=vmin, vmax=vmax, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('mag' in imagetype) and (('coh' in imagetype) or ('gamma' in imagetype)):
                cmap = 'afmhot' if cmap is None else cmap
                try:
                    kapok.vis.show_linear(np.abs(self.coh(pol=pol, bl=bl)), bounds=bounds, cmap=cmap, vmin=0, vmax=1, cbar_label='Coherence Magnitude', figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif (('ph' in imagetype) or ('arg' in imagetype)) and (('coh' in imagetype) or ('gamma' in imagetype)):
                try:
                    kapok.vis.show_linear(np.angle(self.coh(pol=pol, bl=bl)), bounds=bounds, vmin=-np.pi, vmax=np.pi, cmap='hsv', cbar_label='Phase (radians)', figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif ('coh' in imagetype) or ('gamma' in imagetype):
                try:
                    kapok.vis.show_complex(self.coh(pol=pol, bl=bl), bounds=bounds, cbar=True, figsize=figsize, dpi=dpi, savefile=savefile)
                except:
                    print('kapok.Scene.show | Error: Requested coherence cannot be displayed.  Is the polarization identifier valid?  Have you run .opt()?')
                
            elif ('pauli' in imagetype) or ('rgb' in imagetype):
                i = tr*self.num_pol
                vmin = -25 if vmin is None else vmin
                vmax = -3 if vmax is None else vmax
                tm = makehermitian(self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol])
                kapok.vis.show_paulirgb(tm, bounds=bounds, vmin=vmin, vmax=vmax, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('inc' in imagetype):
                vmin = 25 if vmin is None else vmin
                vmax = 65 if vmax is None else vmax 
                cmap = 'viridis' if cmap is None else cmap
                kapok.vis.show_linear(np.degrees(self.inc), cmap=cmap, vmin=vmin, vmax=vmax, cbar_label='Incidence Angle (degrees)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                
            elif ('kz' in imagetype) or ('vertical wavenumber' in imagetype):
                cmap = 'viridis' if cmap is None else cmap
                
                if self.num_baselines > 1:
                    kapok.vis.show_linear(self.kz[bl], cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=r'$k_{z}$ (rad/m)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                else:
                    kapok.vis.show_linear(self.kz, cmap=cmap, vmin=vmin, vmax=vmax, cbar_label=r'$k_{z}$ (rad/m)', bounds=bounds, figsize=figsize, dpi=dpi, savefile=savefile)
                
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
                print('kapok.Scene.show | Unrecognized image type: "'+imagetype+'".  Aborting.')
                
        elif isinstance(imagetype, (collections.Sequence, np.ndarray)):
            cmap = 'viridis' if cmap is None else cmap
            kapok.vis.show_linear(imagetype, vmin=vmin, vmax=vmax, bounds=bounds, cmap=cmap, figsize=figsize, dpi=dpi, savefile=savefile, **kwargs)
            
        else:
            print('kapok.Scene.show | Unrecognized image type: "'+imagetype+'".  Aborting.')
            
        return


    def power(self, pol=0, tr=0):
        """Return the backscattered power, in linear units, for the specified
            polarization and track number.
            
            Arguments:
                pol: The polarization of the desired power.  Can be an
                    integer polarization index (0: HH, 1: HV, 2: VV), or a
                    list containing a polarization weight vector with three
                    complex elements, or a string.  Allowed strings are 'HH',
                    'HV', 'VV', 'HH+VV', and 'HH-VV'.  Default: 0.
                tr (int): Desired track index.  Default: 0.
                
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
            pwr = self.cov[:,:,i,i]
        elif len(pol) == self.num_pol:
            i = tr*self.num_pol
            tm = self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol]
            tm = makehermitian(tm)
            
            wimage = np.ones((self.dim[0],self.dim[1],3,3),dtype='complex')
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
    
    
    def coh(self, pol=0, polb=None, bl=0, pix=None):
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
                    coherences reordered as a result. Default: 0.
                polb: If different master and slave track polarizations are
                    desired, specify the slave polarization here.  Use same
                    form (int, str, or list) as pol.  Default: None (polb=pol).
                bl (int): Desired baseline index.  Default: 0.
                pix (int): If you only wish to calculate the coherence for a
                single pixel, specify a tuple with the (azimuth,range) indices
                of the pixel here.
                
            Returns:
                coh (array): A complex coherence image.
            
        """       
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
            elif (pix is None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl, 0]
            elif (pix is None):
                coh = self.pdcoh[0]
            elif self.num_baselines > 1:
                coh = self.pdcoh[bl,0,pix[0],pix[1]]
            else:
                coh = self.pdcoh[0,pix[0],pix[1]]
        elif isinstance(pol,str) and ('low' in pol):
            if self.pdcoh is None:
                coh = None
            elif (pix is None) and (self.num_baselines > 1):
                coh = self.pdcoh[bl, 1]
            elif (pix is None):
                coh = self.pdcoh[1]
            elif self.num_baselines > 1:
                coh = self.pdcoh[bl,1,pix[0],pix[1]]
            else:
                coh = self.pdcoh[1,pix[0],pix[1]]
        elif not isinstance(pol, (collections.Sequence, np.ndarray)) and (pol in range(self.num_pol)):
            i,j = mb_cov_index(bl, pol=pol, pol2=polb, n_pol=self.num_pol)
            if pix is None:
                coh = self.cov[:,:,i,j] / np.sqrt(np.abs(self.cov[:,:,i,i]*self.cov[:,:,j,j]))
            else:
                coh = self.cov[pix[0],pix[1],i,j] / np.sqrt(np.abs(self.cov[pix[0],pix[1],i,i]*self.cov[pix[0],pix[1],j,j]))
        elif isinstance(pol, (collections.Sequence, np.ndarray)) and (len(pol) == self.num_pol):
            i,j = mb_cov_index(bl, pol=0, n_pol=self.num_pol)
            
            if pix is None:
                t11 = self.cov[:,:,i:i+self.num_pol,i:i+self.num_pol]
                t22 = self.cov[:,:,j:j+self.num_pol,j:j+self.num_pol]
                om = self.cov[:,:,i:i+self.num_pol,j:j+self.num_pol]
                
                t11 = makehermitian(t11)
                t22 = makehermitian(t22)
                
                wimage = np.ones((self.dim[0],self.dim[1],3,3),dtype='complex')
                
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
        
        
    def region(self, az=None, rng=None, mode='basic', bl=0, savefile=None):
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
        
        """
        import kapok.region
        
        if (mode == 'basic') and (az is not None) and (rng is not None):
            kapok.region.cohregion(self, az, rng, bl=bl, mlwin=self.ml_window, savefile=savefile)
        else:
            kapok.region.rvogregion(self, az=az, rng=rng, bl=bl)
            
        return


    def geo(self, data, outfile, outformat='ENVI', resampling='bilinear'):
        """Output a geocoded raster.
        
            Resampling from radar coordinates to latitude/longitude using
            gdalwarp.
            
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
                    Options include 'near', 'bilinear', 'cubic', 'lanczos',
                    and others.  Default is 'bilinear'.  For reference and
                    more options, see http://www.gdal.org/gdalwarp.html.
            
        """        
        if isinstance(data, str):
            data = self.get(data)
        
        outpath, outfile = os.path.split(outfile)        
        
        kapok.geo.radar2ll(outpath, outfile, data, self.lat[:], self.lon[:], outformat=outformat, resampling=resampling)
        
        return
        
        
    def ingest(self, file, name, attrname=None, attrunits='', overwrite=False):
        """Ingest ancillary data into Kapok HDF5 file.
        
            Allows the user to import ancillary raster data in ENVI format such as
            lidar, external DEMs, etc.  This external raster data will be
            resampled to the radar coordinates using bilinear interpolation,
            then saved to the HDF5 file as datasets with the same dimensions as
            the radar data. The ingested data can then be compared to the
            radar-derived products or used in guided inversion functions, etc.
            
            Data will be stored in the HDF5 file under 'ancillary/<name>', where
            <name> is the string given in the name argument to this function.
            
            N.B. The data to import should be in ENVI format, in WGS84 Geographic
            (latitude, longitude) coordinates.
            
            Arguments:
                file (str): Path and filename to the external raster data (in
                    ENVI format).  Will be loaded in GDAL.
                name (str): Name of the HDF5 dataset which will be created to
                    store the ingested data.
                attrname (str): Name which will be put into a 'name' attribute
                    of the dataset.  Will be shown when displaying the data using
                    Scene.show(), etc.  Default: Same as name.
                attrunits (str): Units of the data.  Will be shown on plots of the
                    data using Scene.show(), etc.
                overwrite (bool): Set to True to overwrite an already existing
                    HDF5 dataset, if one already exists under the same name as the
                    name input argument.  Default: Do not overwrite.
                
            Returns:
                data: A link to the newly created HDF5 dataset containing the
                    ingested data.
        
        """
        import osgeo.gdal as gdal
        
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