# -*- coding: utf-8 -*-
"""Data Simulation.

    Simulates a Kapok HDF5 file.  Simulation is done using a forward
    coherence model, with random distributions of the model parameters.
    Note that the underlying physics is not simulated.  Therefore, the
    resulting data will always perfectly match the forward model.  The model
    used is as described in the paper:
    
    Neumann, M., Ferro-Famil, L., & Reigber, A. (2010). Estimation of forest
    structure, ground, and canopy layer characteristics from multibaseline
    polarimetric interferometric SAR data. IEEE Transactions on Geoscience
    and Remote Sensing. http://doi.org/10.1109/TGRS.2009.2031101
    
    There are also a number of simplifications in the simulated geometry.
    For example, a flat earth is used in order to calculate the change in
    incidence angle across the dataset.  The generated interferometric phases
    are already flattened.  That is, an interferometric phase of 0 radians
    in the simulated data is equivalent to a height of 0 m.
    
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
import collections
import time

import numpy as np
import h5py

import kapok
import kapok.smodel
from kapok.rvog import rvogfwdvol
from kapok.lib import mb_num_tracks, mb_tr_index, convertT3toC3, rotateT3


def simulate(outfile, hv=20, hvdist='normal', ext=0.1, azsize=250, rngsize=250,
             altitude=12500, wv=0.2384, inc=(25,65), baselines=(40),
             compression='gzip', compression_opts=4, overwrite=False):
    """Create a Kapok HDF5 Scene file for a simulated dataset with the
        specified parameters.
    
        Arguments:
            outfile (str): Filename for the newly created file containing
                the simulated data.
            hv: Argument specifying the forest height values to simulate.  If
                a single value is given, this will be the mean value of the
                simulated forest heights.  If a two element tuple is given,
                these represent the two parameters of the probability
                distribution used to generate the forest heights.  Note that
                the next argument, hvdist, decides which probability
                distribution to use to distribute the forest height values.
                The two options are 'normal' and 'uniform'.  If hvdist is
                'normal', a normal distribution is used.  If hv is a single
                value, that will be the mean of the normal distribution, with
                a standard deviation of 5m.  If hv is a two element tuple,
                the first element will be the mean value of the distribution,
                and the second value will be the standard deviation.  If
                hvdist is 'uniform', a uniform distribution is used.  If hv is
                a single value, the minimum value of the uniform distribution
                will be hv-5, and the maximum value will be hv+5.  If hv is
                a two element tuple, the first element will be the minimum of
                the normal distribution, and the second element will be the
                maximum of the normal distribution.  If a single value or two
                element tuple is given, a circular forest patch at the center
                of the simulated dataset will be created.  At the edges of the
                simulated dataset will be a short vegetation / mostly bare
                area.  Another option is to pass a 2D array into this
                argument.  In this case, it will be taken as a map
                specifying the hv values for every pixel.  Default: 20.
            hvdist (str): String identifying the distribution of the hv values.
                The default is 'normal'.  This can also be set to 'uniform'.
                See description of the hv argument for more details.
            ext: Argument specifying the extinction values to simulate in the
                forested region.  As with the hv keyword, a single value
                will be used as the mean of the normally distributed simulated
                extinction values in dB/m, with standard deviation of 0.1 dB/m.
                If a two element tuple is given, they will be taken as the
                mean and standard deviation of the values.  If a 2D array is
                given, these will be taken as a map specifying the extinction
                for each pixel.
            azsize (int): Azimuth size of the simulated dataset (number of
                rows).
            rngsize (int): Range size of the simulated dataset (number of
                columns).
            altitude (float): Platform altitude of the master track, in
                meters.  Default: 12500.
            wv (float): Sensor wavelength, in meters.  Default: 0.2384.
            inc (tuple): Incidence angle of the master track at the near and
                far edges of the data, in degrees.  The necessary slant range
                and uniform slant range spacing is calculated in order to
                produce the desired incidence angle variation across the
                simulated data.  Default: (25, 65).
            baselines (tuple): Vertical baselines for the desired secondary
                tracks, in meters.  Note that this function only supports
                vertical baselines.  If multiple baselines are desired,
                specify each baseline as a separate element in the tuple.
                Default: (50).
            compression (str): Compression keyword used when creating the HDF5
                datasets.  Default: 'gzip'.
            compression_opts (int): Compression option value used when creating
                the HDF5 datasets.  Number from 0 to 9, with 9 being maximum
                compression.  Default: 4.
            overwrite (bool): Set to True to overwrite an existing Kapok HDF5
                file if necessary.  Default: False.
                
        Returns:
            scene (object): A Kapok Scene object containing the simulated
                data.
        
    """    
    # Create the HDF5 file.
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.sim.simulate | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return    
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.sim.simulate | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return

    
    
    # Calculate slant range of near and far edges of swath.
    sredges = altitude / np.cos(np.radians(inc))
    sr = np.tile(np.linspace(sredges[0],sredges[1],num=rngsize),(azsize,1))

        
    # Metadata
    f.attrs['site'] = 'Simulated Dataset'
    f.attrs['stack_name'] = 'Simulated data from kapok.sim.simulate().'
    f.attrs['tracks'] = []
    f.attrs['sensor'] = 'Simulated Data'
    f.attrs['average_altitude'] = altitude
    f.attrs['image_starting_slant_range'] = sr[0,0]
    f.attrs['cov_slant_range_pixel_spacing'] = sr[1,0] - sr[0,0]
    f.attrs['cov_azimuth_pixel_spacing'] = 1.0
    f.attrs['ml_window'] = (1,1)
    f.attrs['sm_window'] = (1,1)

    if isinstance(baselines, (collections.Sequence, np.ndarray)):
        num_bl = len(baselines)
    else:
        num_bl = 1
        
    num_tracks = mb_num_tracks(num_bl)
    num_pol = int(3) # Number of Polarizations (HH, sqrt(2)*HV, VV)
    num_cov_elements = num_tracks*num_pol # Number of Covariance Matrix Elements in Each Row
    
    f.attrs['num_baselines'] = num_bl
    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_pol'] = 3
    
    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts

    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['wavelength'] = wv
    
    
    
    # Create covariance matrix dataset.
    cov = f.create_dataset('cov', (azsize, rngsize, num_cov_elements, num_cov_elements), dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)
    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*HV', 'VV'],dtype='S')
    
    
    # Initialize master track incidence angle dataset:
    incdata = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    incdata.attrs['units'] = 'radians'
    incdata.attrs['description'] = 'Reference Track Incidence Angle'
    incdata[:] = np.arccos(altitude / sr)
    
    
    # Initialize kz dataset:
    kz = f.create_dataset('kz', (num_tracks, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz[0] = 0
    
    if num_bl > 1:
        for n in range(1,num_tracks):
            kz[n] = 4*np.pi*np.cos(incdata[:] - (np.pi/2))*baselines[n]/sr/np.sin(incdata[:])/wv
    else:
        kz[1] = 4*np.pi*np.cos(incdata[:] - (np.pi/2))*baselines/sr/np.sin(incdata[:])/wv
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    
    
    # Initialize latitude dataset and import values:
    llspacing = 5.5555555555e-5 # fake longitude/latitude spacing for simulated dataset, just so we have something to put in the arrays
    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    latsim = np.linspace(0,-1*llspacing*(azsize-1),num=azsize)
    lat[:] = np.transpose(np.tile(latsim,(rngsize,1)))
    lat.attrs['units'] = 'degrees'
    lat.attrs['description'] = 'Latitude'
    
    # Initialize longitude dataset and import values:
    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lonsim = np.linspace(0,llspacing*(rngsize-1),num=rngsize)
    lon[:] = np.tile(lonsim,(azsize,1))
    lon.attrs['units'] = 'degrees'
    lon.attrs['description'] = 'Longitude'

    # Initialize DEM height dataset and import values:
    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = 0
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Processor DEM'
    
    
    # Designate forest and non-forest area.   
    r = azsize*0.45   
    y,x = np.ogrid[-azsize/2:azsize/2, -rngsize/2:rngsize/2]
    forestmask = x*x + y*y <= r*r

    
    # Generate forest heights.
    if isinstance(hv, (collections.Sequence, np.ndarray)) and (len(hv) > 2):
        print('kapok.sim.simulate | Using user specified forest height values.')
    elif 'uniform' in hvdist:
        print('kapok.sim.simulate | Generating uniformly distributed forest height values.')
    else:
        print('kapok.sim.simulate | Generating normally distributed forest height values.')
    
    if isinstance(hv, (collections.Sequence, np.ndarray)):
        if len(hv) == 2:
            if 'uniform' in hvdist:
                hvmap = np.random.uniform(hv[0], hv[1], (azsize, rngsize))
            else:
                hvmap = np.random.normal(hv[0], hv[1], (azsize, rngsize))
            hvshort = np.random.normal(1, 1.5, (azsize, rngsize))
            hvmap[~forestmask] = hvshort[~forestmask]
        elif len(hv) == 1:
            if 'uniform' in hvdist:
                hvmap = np.random.uniform(hv-5, hv+5, (azsize, rngsize))
            else:
                hvmap = np.random.normal(hv, 5, (azsize, rngsize))
            hvshort = np.random.normal(1, 1.5, (azsize, rngsize))
            hvmap[~forestmask] = hvshort[~forestmask]
        else:
            hvmap = hv.copy()
    else:
        if 'uniform' in hvdist:
            hvmap = np.random.uniform(hv-5, hv+5, (azsize, rngsize))
        else:
            hvmap = np.random.normal(hv, 5, (azsize, rngsize))
        hvshort = np.random.normal(1, 1.5, (azsize, rngsize))
        hvmap[~forestmask] = hvshort[~forestmask]
    
    hvmap = np.clip(hvmap, 0.01, np.inf)
    
    
    
    # Generate extinction values.
    if isinstance(ext, (collections.Sequence, np.ndarray)):
        if len(hv) == 2:
            extmap = np.random.normal(ext[0], ext[1], (azsize, rngsize))
        elif len(hv) == 1:
            extmap = np.random.normal(ext, 0.1, (azsize, rngsize))
        else:
            extmap = ext.copy()
    else:
        extmap = np.random.normal(ext, 0.1, (azsize, rngsize))

    extmap = np.clip(extmap, 1e-3, 1.0)
    extmap /= (20/np.log(10)) # convert from dB/m to Np/m
    
    
    # Degree of orientation randomness.
    tau = np.random.normal(0.9, 0.1, (azsize, rngsize))
    taushort = np.random.normal(0.3, 0.2, (azsize, rngsize))
    tau[~forestmask] = taushort[~forestmask]
    tau = np.clip(tau, 0, 1)
    g, gc = kapok.smodel.converttau(tau)
    
    
    # Particle scattering anisotropy.
    delta = np.random.normal(-0.75, 0.1, (azsize, rngsize))
    delta = delta.astype('complex64')
    delta = np.clip(delta, -1, 1)
    
    
    # Volume coherency intensity normalization factor.
    tv = kapok.smodel.volumecoherency(delta, g, gc)
    fv = np.random.normal(-10, 3, (azsize, rngsize))
    fv = np.power(10,fv/10)
    fv = np.clip(fv, 0.01, 1.0)
    fv[~forestmask] /= 10

    
    
    # Rotate volume coherency matrix by random orientation angles in
    # polarization plane.
    psi = np.random.normal(np.pi/8, np.pi/4, (azsize, rngsize))
    tv = np.rollaxis(np.rollaxis(tv,-1),-1)
    tvr = rotateT3(tv, psi)
    tvr = np.rollaxis(np.rollaxis(tvr,-1),-1)
    
    
    # Ground coherency matrix parameters.
    betaint = np.random.normal(0.1, 0.1, (azsize, rngsize))
    betaarg = np.random.uniform(-np.pi, np.pi, (azsize, rngsize))
    beta = betaint*np.exp(1j*betaarg)
    beta22 = np.random.normal(1, 0.1, (azsize, rngsize))
    beta33 = np.random.normal(0.0125, 0.00125, (azsize, rngsize))
       
    
    # Ground coherency intensity normalization factor.
    tg = kapok.smodel.groundcoherency(beta, beta22, beta33)    
    fg = np.random.normal(-20, 3, (azsize, rngsize))
    fg = np.power(10,fg/10)
    fg = np.clip(fg, 0.001, 0.1)

    
    # Fill in polarimetric covariance matrices.
    tmat = (fg*tg + fv*tvr)
    tmat = np.rollaxis(np.rollaxis(tmat,-1),-1)
    cmat = convertT3toC3(tmat)
    for tr in range(num_tracks):
        row = (tr*3)
        cov[:,:,row:row+3,row:row+3] = cmat
    
    
    # Fill in interferometric covariance matrices.
    for bl in range(num_bl):
        (tr1, tr2) = mb_tr_index(bl)
        row = (tr1*3)
        col = (tr2*3)
        
        gammav = rvogfwdvol(hvmap, extmap, incdata[:], kz[tr2]-kz[tr1])
            
        tmat = (fg*tg*1 + fv*tvr*gammav)
        tmat = np.rollaxis(np.rollaxis(tmat,-1),-1)
        cmat = convertT3toC3(tmat)
        
        cov[:,:,row:row+3,col:col+3] = cmat
        
        
    # Create some datasets which store the simulation parameters (e.g., for
    # comparison with model inversion results).
    f.create_dataset('sim/hv', data=hvmap, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/ext', data=extmap, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/tau', data=tau, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/delta', data=delta, dtype='complex64', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/beta', data=beta, dtype='complex64', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/beta22', data=beta22, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/beta33', data=beta33, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/psi', data=psi, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/fv', data=fv, dtype='float32', compression=compression, compression_opts=compression_opts)
    f.create_dataset('sim/fg', data=fg, dtype='float32', compression=compression, compression_opts=compression_opts)
    
    
    # Close file and return Scene object.
    print('kapok.sim.simulate | Complete. ('+time.ctime()+')')
    f.close()
    return kapok.Scene(outfile)