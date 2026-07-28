# -*- coding: utf-8 -*-
"""PolSARProSim Data Import Functions

    Module for importing PolSARProSim simulated data.

    Authors: Richard Chen and Michael Denbina

    Copyright 2023-2024 California Institute of Technology.  All rights reserved.
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
from io import StringIO
import re

import numpy as np
import h5py
import pyresample as pr
import netCDF4 as nc
import h5py

import kapok
from kapok.lib import mlook, smooth, nanmlook


def load(inpath, outfile, mlwin=(2,2), smwin=(1,1),
         reference_alt=12500.0, compression='gzip', compression_opts=4,
         overwrite=False):
    """Load a PolSARProSim dataset into the Kapok format, and save it to target
        HDF5 file.

    Arguments:
        inpath (str): Path to the simulated data directory containing the
            individual track directories.
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
        reference_alt (float): Reference altitude in meters.  Default: 12500.0.
        compression (str): Compression keyword used when creating the HDF5
            datasets to store the covariance matrix, incidence angle, kz,
            and DEM heights.  Default: 'gzip'.
        compression_opts (int): Compression option value used when creating
            the HDF5 datasets.  Number from 0 to 9, with 9 being maximum
            compression.  Default: 4.
        overwrite (bool): Set to True to overwrite an existing Kapok HDF5
            file if necessary.  Default: False.

    Returns:
        scene: A Kapok scene object pointing to the newly created HDF5 file.

    """
    # Check if the input directory exists.
    if not os.path.isdir(inpath):
        print('kapok.pspsim.load | Cannot load PolSARProSim data from input path. Aborting.')
        return

    # Create the HDF5 file.
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.pspsim.load | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.pspsim.load | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return
    

    # Figure out path for each track and list available baselines:
    # print('Available baselines:')
    baselines = []
    baseline_names = []
    path_glob = glob(os.path.join(inpath,'BL*'))
    for p in sorted(path_glob):
        if os.path.isdir(p) and 'BL_' in p:
            print(os.path.basename(p))
            baseline_names.append(os.path.basename(p))
            baselines.append(float(re.match(r'BL_v(\d+(\.\d*)?)m', os.path.basename(p)).group(1)))
    baselines = np.array(baselines)
    h_ref = reference_alt + np.mean(baselines)
    # print(f'h_ref: {h_ref}')

    # print(baselines)
    # print(baseline_names)

    log_file = os.path.join(inpath, baseline_names[0], 'track_master/pspsim_config.log')
    params, trees, is_forest = read_log(log_file, is_forest=False)

    n_pol = 3

    slc_stack = np.zeros((len(baselines)*n_pol, params['nx'], params['ny']), dtype='complex64')
    kz_stack = np.zeros((len(baselines), params['nx'], params['ny']), dtype='float32')
    inc_ref = None

    for i, idx in enumerate(np.argsort(baselines)):
        baseline_name = baseline_names[idx]
        # print(i+1, baseline_name)
        baseline_dir = os.path.join(inpath, baseline_name)
        master_dir = os.path.join(baseline_dir, 'track_master')
        slave_dir = os.path.join(baseline_dir, 'track_slave')

        # Read in log file
        log_file = os.path.join(master_dir, 'pspsim_config.log')

        # Retrieve parameters & tree info from log file
        params, _, _ = read_log(log_file)
        Lx = params['Lx'] # image dimensions in x coordinate (azimuth)
        Ly = params['Ly'] # image dimensions in y coordinate (ground range)
        nx = params['nx'] # number of pixels in x
        ny = params['ny'] # number of pixels in y
        # print(f'nx: {nx}, ny: {ny}')

        # Set up axes - x: azimuth, y: ground range
        dx = Lx/nx
        dy = Ly/ny
        x = np.arange(int(-nx/2), int(-nx/2)+nx)*dx
        y = np.arange(int(-ny/2), int(-ny/2)+ny)*dy
        x_corners = np.arange(-Lx/2, Lx/2+dx, dx)
        y_corners = np.arange(-Ly/2, Ly/2+dy, dy)
        # print(baseline_name, y[0], y[-1])

        # Heights
        h_m = reference_alt
        h_s = h_m + float(re.match(r'BL_v(\d+(\.\d*)?)m', baseline_name).group(1))
        # Slant range
        # r_m = np.sqrt(h_m**2 + (y + h_m)**2)
        r_s = np.sqrt(h_s**2 + (y + h_m)**2)
        r_ref = np.sqrt(h_ref**2 + (y + h_m)**2)
        idx_ctr = int(len(r_ref)/2)

        r0 = r_ref[0]
        dr = (r_ref[idx_ctr]-r_ref[0])/idx_ctr
        # print(f'r0: {r0}, dr: {dr}')

        # Incidence angles
        θ_s = np.arctan2(y+h_m, h_s)
        θ_ref = np.arctan2(y+h_m, h_ref)

        if inc_ref is None:
            inc_ref = np.tile(np.reshape(θ_ref, (len(θ_ref), 1)), (1, nx)).transpose()

        # kz
        λ = params['λ']
        kz = 4*np.pi/λ*(θ_s-θ_ref)/np.sin(θ_ref)
        kz = np.tile(np.reshape(kz, (len(kz), 1)), (1, nx)).transpose()

        kz_stack[i] = kz.copy()

        # # Plot tree locations
        # if is_forest:
        #     forest_type = params["forest_type"].lower()
        #     fig, ax = plt.subplots()
        #     img = ax.scatter(trees['x'], trees['y'], s=trees['area'], c=trees['height'], vmin=14, vmax=16, alpha=0.5, cmap='YlGn')
        #     cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="5%")
        #     fig.colorbar(img, cax=cax, label='Tree Height')
        #     ax.set_xlim(-Lx/2, Lx/2)
        #     ax.set_ylim(-Ly/2, Ly/2)
        #     ax.set_title(f'{forest_type} - Tree Locations/Areas/Heights')
        #     ax.set_xlabel('Azimuth w.r.t. Scene Center [m]')
        #     ax.set_ylabel('Ground Range w.r.t. Scene Center [m]')
        #     fig.savefig(fig_dir / f'{forest_type}_trees.jpg', dpi=150)

        # Compute topographic phase
        Δr = r_s - r_ref
        Δr = np.tile(np.reshape(Δr, (len(Δr), 1)), (1, nx)).transpose()
        λ = params['λ']

        # for pq in ['11', '12', '21', '22']:
        for pnum, pq in enumerate(['11', '12', '22']):
            # pq = '11'
            # Read in Spq
            # Spq_bin_m = master_dir / f's{pq}.bin'
            Spq_bin_s = os.path.join(slave_dir, 's{}.bin'.format(pq))
            # Spq_m = np.reshape(np.fromfile(Spq_bin_m, dtype=np.complex64), (nx, ny)).transpose()
            Spq_s = np.reshape(np.fromfile(Spq_bin_s, dtype=np.complex64), (nx, ny))

            # Flatten SLC with topographic phase
            Spq_s = Spq_s * np.exp(1j * 4*np.pi/λ * Δr)

            slc_stack[i*n_pol + pnum] = Spq_s.copy()


    num_tracks = len(baselines) # Number of Tracks
    num_bl = int(num_tracks * (num_tracks-1) / 2) # Number of Baselines
    num_pol = int(3) # Number of Polarizations (HH, sqrt(2)*HV, VV)
    num_cov_elements = num_tracks*num_pol # Number of Covariance Matrix Elements in Each Row

    if np.isclose(λ, 0.238404):
        band = 'L'
    elif np.isclose(λ, 0.689178):
        band = 'P'

    f.attrs['stack_name'] = os.path.basename(os.path.normpath(inpath))
    f.attrs['band'] = band
    f.attrs['wavelength'] = λ
    f.attrs['site'] = 'N/A'
    f.attrs['url'] = 'N/A'
    f.attrs['sensor'] = 'PolSARProSim'
    f.attrs['processor'] = 'N/A'

    print('kapok.pspsim.load | Stack ID: '+f.attrs['stack_name'])
    print('kapok.pspsim.load | Band: '+f.attrs['band'])

    print('kapok.pspsim.load | Importing metadata. ('+time.ctime()+')')

    f.attrs['average_altitude'] = reference_alt
    f.attrs['image_starting_slant_range'] = 'N/A'
    f.attrs['slc_azimuth_pixel_spacing'] = dx
    f.attrs['slc_slant_range_pixel_spacing'] = dr
    f.attrs['cov_azimuth_pixel_spacing'] = dx * mlwin[0]
    f.attrs['cov_slant_range_pixel_spacing'] = dr * mlwin[1]

    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_baselines'] = num_bl
    f.attrs['tracks'] = np.array(baseline_names, dtype='S')

    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts

    f.attrs['peg_latitude'] = 'N/A'
    f.attrs['peg_longitude'] = 'N/A'
    f.attrs['peg_heading'] = 'N/A'

    f.attrs['subset'] = False

    # Multi-looked, subsetted, image dimensions:
    print('***')
    print(nx)
    print(ny)
    print('***')
    azsize = nx // mlwin[0]
    rngsize = ny // mlwin[1]
    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['dim_slc'] = (nx, ny)
    f.attrs['ml_window'] = mlwin
    f.attrs['sm_window'] = smwin

    # Initialize covariance matrix dataset in HDF5 file:
    cov = f.create_dataset('cov', (azsize, rngsize, num_cov_elements, num_cov_elements), dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)

    # Calculate covariance matrix.
    for row in range(0,num_cov_elements):
        for col in range(row,num_cov_elements):
            print('kapok.pspsim.load | Calculating covariance matrix element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')     ', end='\r')
            reference_slc = slc_stack[row]
            secondary_slc = slc_stack[col]

            if (row % 3) == 1: # Cross Polarization
                reference_slc = np.sqrt(2) * reference_slc
            if np.any(np.abs(reference_slc) < 1e-5):
                reference_slc[np.abs(reference_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.
            
            if (col % 3) == 1: # Cross Polarization
                secondary_slc = np.sqrt(2) * secondary_slc
            if np.any(np.abs(secondary_slc) < 1e-5):
                secondary_slc[np.abs(secondary_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.
            
            # print(azsize)
            # print(rngsize)
            # print(reference_slc.shape)
            # print(secondary_slc.shape)
            # print(mlwin)
            cov[:,:,row,col] = mlook(reference_slc*np.conj(secondary_slc),mlwin)
    
    f.flush()

    # Boxcar Smoothing:
    if smwin != (1,1):
        print('kapok.pspsim.load | Boxcar averaging covariance matrix. ('+time.ctime()+')')
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                cov[:,:,row,col] = smooth(cov[:,:,row,col],smwin)

    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*VH', 'VV'],dtype='S')
    print('kapok.pspsim.load | Covariance matrix calculation complete. ('+time.ctime()+')     ')

    # Save tree location data.
    if trees is not None:
        trees_azimuth_pixel = (trees['x'] - x[0]) / (dx*mlwin[0])
        trees_range_pixel = (trees['y'] - y[0]) / (dy*mlwin[1])

        treex = f.create_dataset('ancillary/trees/row', data=trees_azimuth_pixel, compression=compression, compression_opts=compression_opts)
        treey = f.create_dataset('ancillary/trees/col', data=trees_range_pixel, compression=compression, compression_opts=compression_opts)
        treeh = f.create_dataset('ancillary/trees/height', data=trees['height'], compression=compression, compression_opts=compression_opts)
        treer = f.create_dataset('ancillary/trees/radius', data=trees['radius'], compression=compression, compression_opts=compression_opts)

    # Load geolocation data.
    print('kapok.pspsim.load | Loading geolocation data and setting up coordinate grids. ('+time.ctime()+')')

    # Initialize latitude dataset and import values:
    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lat[:] = 0.0
    lat.attrs['units'] = 'N/A'
    lat.attrs['description'] = 'N/A'

    # Initialize longitude dataset and import values:
    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lon[:] = 0.0
    lon.attrs['units'] = 'N/A'
    lon.attrs['description'] = 'N/A'

    # Initialize DEM height dataset and import values:
    print('kapok.pspsim.load | Populating DEM heights with zeros. ('+time.ctime()+')')
    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = 0.0
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Zero meter simulated DEM.'

    # Create dummy gfc treecover, datamask, and lossyear rasters with fake data.
    gfc_treecover2000 = f.create_dataset('ancillary/gfc_treecover2000', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_treecover2000[:] = 100
    gfc_datamask = f.create_dataset('ancillary/gfc_datamask', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_datamask[:] = 1
    gfc_lossyear = f.create_dataset('ancillary/gfc_lossyear', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_lossyear[:] = 0

    f.flush()

    # Incidence Angle of Reference Track
    print('kapok.pspsim.load | Importing incidence angles from reference track RGI product. ('+time.ctime()+')')
    # Initialize incidence angle dataset (stores reference track incidence angle):
    inc = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Reference Track Local Incidence Angle'
    inc.attrs['slope_corrected'] = False # PolSARProSim incidence angle is reference to ellipsoid, no terrain
    inc[:] = mlook(inc_ref, mlwin)

    # Initialize kz dataset:
    kz = f.create_dataset('kz', (num_tracks, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    kz.attrs['indexing'] = 'track'
    kz.attrs['slope_corrected'] = False

    # Load kz files.
    for tr in range(1,num_tracks):
        kz[tr,:,:] = mlook(kz_stack[tr], mlwin)

    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.pspsim.load | Complete. ('+time.ctime()+')     ')
    return kapok.Scene(outfile)


def read_log(log_file, is_forest=False):
    with open(log_file) as f:
        log = f.readlines()
    params = {}
    p_λ = re.compile(r'Wavelength[ \t]*=[ \t]*(\d+(\.\d*)?) metres')
    p_Lx = re.compile(r'Lx[ \t]*=[ \t]*(\d+(\.\d*)?) metres')
    p_Ly = re.compile(r'Ly[ \t]*=[ \t]*(\d+(\.\d*)?) metres')
    p_nx_dx = re.compile(r'There are (\d+) pixels in azimuth of width (\d+(\.\d*)?) metres')
    p_ny_dy = re.compile(r'There are (\d+) pixels in ground range of width (\d+(\.\d*)?) metres')
    p_num_trees = re.compile(r'There are (\d+) trees in a stand of radius')
    p_forest_type = re.compile(r'(\S+) forest simulation$')
    for i, line in enumerate(log):
        if p_λ.match(line):
            params['λ'] = float(p_λ.match(line).group(1))
        if p_Lx.match(line):
            params['Lx'] = float(p_Lx.match(line).group(1))
        if p_Ly.match(line):
            params['Ly'] = float(p_Ly.match(line).group(1))
        if p_nx_dx.match(line):
            params['nx'] = int(p_nx_dx.match(line).group(1))
            # dx = float(p_nx_dx.match(line).group(2))
        if p_ny_dy.match(line):
            params['ny'] = int(p_ny_dy.match(line).group(1))
            # dy = float(p_ny_dy.match(line).group(2))
        if p_num_trees.match(line):
            params['num_trees'] = int(p_num_trees.match(line).group(1))
            params['idx_tree_start'] = i+3
            params['idx_tree_end'] = i+3+params['num_trees']
        if p_forest_type.match(line):
            params['forest_type'] = p_forest_type.match(line).group(1)
            is_forest = True
    # Get tree info
    if is_forest:
        tree_data = np.loadtxt(StringIO(''.join(log[params['idx_tree_start']:params['idx_tree_end']])))[:, 1:]
        trees = {}
        trees['x'] = tree_data[:, 0]
        trees['y'] = tree_data[:, 1]
        trees['height'] = tree_data[:, 2]
        trees['radius'] = tree_data[:, 3]
        trees['area'] = np.pi*trees['radius']**2
    else:
        trees = None
    return params, trees, is_forest


def load_combined_file(fileglob, gridrow, gridcol, gridx, gridy, gridlat, gridlon, gridchm, gridcenterx, gridcentery, gridshape, return_ref=False, dtype='complex64'):
    combined_data = np.ones(gridshape, dtype=dtype) * np.nan
    combined_dist = np.ones(gridshape, dtype=dtype) * 1e9
    if return_ref:
        combined_inc = np.ones(gridshape, dtype=dtype) * np.nan
        combined_lat = np.ones(gridshape, dtype=dtype) * np.nan
        combined_lon = np.ones(gridshape, dtype=dtype) * np.nan
        combined_chm = np.ones(gridshape, dtype=dtype) * np.nan
        combined_tile = np.ones(gridshape, dtype=dtype) * np.nan
    
    for file_num, file in enumerate(fileglob):
        data = np.fromfile(file, dtype=dtype)
        data = np.reshape(data, gridx[file_num].shape)

        dist_from_center = np.sqrt((gridx[file_num] - gridcenterx[file_num])**2 + (gridy[file_num] - gridcentery[file_num])**2)

        col = gridcol[file_num].astype('int32')
        row = gridrow[file_num].astype('int32')
        extracted_dist = combined_dist[col, row]

        ind_replace = (dist_from_center < extracted_dist)

        if np.any(ind_replace):
            col = col[ind_replace]
            row = row[ind_replace]
            combined_data[col, row] = data[ind_replace]
            combined_dist[col, row] = dist_from_center[ind_replace]

            if return_ref:
                # scrape incidence from file path
                inc = np.nan
                for chunk in file.split('_'):
                    if 'INC' in chunk:
                        inc = float(chunk.replace('INC',''))
                combined_inc[col, row] = inc
                combined_lat[col, row] = gridlat[file_num][ind_replace]
                combined_lon[col, row] = gridlon[file_num][ind_replace]
                combined_chm[col, row] = gridchm[file_num][ind_replace]
                combined_tile[col, row] = file_num
    
    if return_ref:
        return combined_data, combined_inc, combined_lat, combined_lon, combined_chm, combined_tile
    else:
        return combined_data


def combine_batch(inpath, outpath, band='L', noise='nonoise', gridpath=None):
    """ Merge SLCs from different simulation tiles into one large merged dataset. """
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    
    if gridpath is None:
        gridpath = inpath
    
    gridglob = np.sort(glob(gridpath + '*/*/slc_grid_wutm.nc'))

    print('Number of Simulated Tiles Found: {}'.format(len(gridglob)))
    print('Tile Grids:')
    print(gridglob)

    gridx = {}
    gridy = {}
    gridlat = {}
    gridlon = {}
    gridchm = {}
    gridorigin = {}
    gridspacing = {}
    gridcenterx = {}
    gridcentery = {}

    for subgrid_num, gridfile in enumerate(gridglob):
        print('Parsing grid from "{}"...'.format(gridfile))
        grid = nc.Dataset(gridfile)

        gridx[subgrid_num] = grid['utm_x'][:]
        gridy[subgrid_num] = grid['utm_y'][:]
        gridlat[subgrid_num] = grid['lat'][:]
        gridlon[subgrid_num] = grid['lon'][:]
        gridchm[subgrid_num] = grid['canopy_height'][:]

        gridorigin[subgrid_num] = (np.nanmin(gridx[subgrid_num]), np.nanmin(gridy[subgrid_num]))
        gridspacing[subgrid_num] = (gridx[subgrid_num][1,0] - gridx[subgrid_num][0,0],
                                    gridy[subgrid_num][0,1] - gridy[subgrid_num][0,0])
        
        gridcenterx[subgrid_num] = grid.center_utm_x
        gridcentery[subgrid_num] = grid.center_utm_y

    # make overall utm grid and calculate pixel locations for each UTM X and Y coordinate
    # weight pixels based on distance from center X and center Y
    neworiginx = gridorigin[0][0]
    neworiginy = gridorigin[0][1]
    for key in gridorigin:
        neworiginx = np.nanmin([neworiginx, gridorigin[key][0]])
        neworiginy = np.nanmin([neworiginy, gridorigin[key][1]])
    
    newspacingx = None
    newspacingy = None
    for key in gridspacing:
        if newspacingx is None:
            newspacingx = np.array([gridspacing[key][0]])
            newspacingy = np.array([gridspacing[key][1]])
        else:
            newspacingx = np.append(newspacingx, gridspacing[key][0])
            newspacingy = np.append(newspacingy, gridspacing[key][1])
        
    newspacingx = np.nanmedian(newspacingx)
    newspacingy = np.nanmedian(newspacingy)
    
    gridrow = {}
    gridcol = {}
    allrow = np.array([])
    allcol = np.array([])
    for key in gridx:
        gridrow[key] = (gridy[key] - neworiginy) / newspacingy
        gridcol[key] = (gridx[key] - neworiginx) / newspacingx
        allrow = np.append(allrow, gridrow[key])
        allcol = np.append(allcol, gridcol[key])
    
    gridshape = np.array([np.nanmax(np.ceil(allcol)), np.nanmax(np.ceil(allrow))], dtype='int32')

    max_baseline = 42

    for pol in np.array(['11', '12', '22']):
        for baseline in np.arange(1, max_baseline):
            slcglob = np.sort(glob(os.path.join(inpath+'*', '*{}_results/{}/S{}_{}.slc'.format(band, noise, pol, baseline))))
            combined_slc = load_combined_file(slcglob, gridrow, gridcol, gridx, gridy, gridlat, gridlon, gridchm, gridcenterx, gridcentery, gridshape)
            np.save(os.path.join(outpath, 'S{}_{}.slc'.format(pol, baseline)), combined_slc)
    
    for baseline in np.arange(1, max_baseline):
        kzglob = np.sort(glob(os.path.join(inpath+'*', '*{}_results/{}/kz_{}.kz'.format(band, noise, baseline))))

        if baseline == 1:
            combined_kz, combined_inc, combined_lat, combined_lon, combined_chm, combined_tile = load_combined_file(kzglob, gridrow, gridcol, gridx, gridy, gridlat, gridlon, gridchm, gridcenterx, gridcentery, gridshape, return_ref=True, dtype='float32')
            np.save(os.path.join(outpath, 'kz_{}.kz'.format(baseline)), combined_kz)
            np.save(os.path.join(outpath, 'ref.inc'), combined_inc)
            np.save(os.path.join(outpath, 'ref.lat'), combined_lat)
            np.save(os.path.join(outpath, 'ref.lon'), combined_lon)
            np.save(os.path.join(outpath, 'ref.chm'), combined_chm)
            np.save(os.path.join(outpath, 'ref.tile'), combined_tile)
        else:
            combined_kz = load_combined_file(kzglob, gridrow, gridcol, gridx, gridy, gridlat, gridlon, gridchm, gridcenterx, gridcentery, gridshape, return_ref=False, dtype='float32')
            np.save(os.path.join(outpath, 'kz_{}.kz'.format(baseline)), combined_kz)
    
    log_file = np.sort(glob(os.path.join(inpath+'*', '*_{}/BL_v0.0m/track_master/pspsim_config.log'.format(band))))[0]
    params, trees, is_forest = read_log(log_file, is_forest=False)
    np.save(os.path.join(outpath, 'ref.wavelength'), params['λ'])


def load_batch(inpath, outfile, mlwin=(8,2), smwin=(1,1),
               reference_alt=12500.0, compression='gzip', compression_opts=4,
               overwrite=False):
    """Load a PolSARProSim dataset into the Kapok format, and save it to target
        HDF5 file.

    Arguments:
        inpath (str): Path to the simulated data directory containing the
            individual track directories.
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
        reference_alt (float): Reference altitude in meters.  Default: 12500.0.
        compression (str): Compression keyword used when creating the HDF5
            datasets to store the covariance matrix, incidence angle, kz,
            and DEM heights.  Default: 'gzip'.
        compression_opts (int): Compression option value used when creating
            the HDF5 datasets.  Number from 0 to 9, with 9 being maximum
            compression.  Default: 4.
        overwrite (bool): Set to True to overwrite an existing Kapok HDF5
            file if necessary.  Default: False.

    Returns:
        scene: A Kapok scene object pointing to the newly created HDF5 file.

    """
    # Check if the input directory exists.
    if not os.path.isdir(inpath):
        print('kapok.pspsim.load_batch | Cannot load PolSARProSim data from input path. Aborting.')
        return

    # Create the HDF5 file.
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.pspsim.load_batch | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.pspsim.load_batch | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return
    
    kzglob = glob(os.path.join(inpath, '*.kz.npy'))
    num_tracks = len(kzglob)
    wavelength = float(np.load(os.path.join(inpath, 'ref.wavelength.npy')))

    num_bl = int(num_tracks * (num_tracks-1) / 2) # Number of Baselines
    num_pol = int(3) # Number of Polarizations (HH, sqrt(2)*HV, VV)
    num_cov_elements = num_tracks*num_pol # Number of Covariance Matrix Elements in Each Row

    temp = np.load(kzglob[0])
    slc_shape = temp.shape
    slc_stack = np.zeros((num_tracks*num_pol, slc_shape[0], slc_shape[1]), dtype='complex64')
    kz_stack = np.zeros((num_tracks, slc_shape[0], slc_shape[1]), dtype='float32')
    del temp

    inc_ref = np.load(os.path.join(inpath, 'ref.inc.npy'))

    if np.isclose(wavelength, 0.238404):
        band = 'L'
    elif np.isclose(wavelength, 0.689178):
        band = 'P'

    f.attrs['stack_name'] = os.path.basename(os.path.normpath(inpath))
    f.attrs['band'] = band
    f.attrs['wavelength'] = wavelength
    f.attrs['site'] = 'N/A'
    f.attrs['url'] = 'N/A'
    f.attrs['sensor'] = 'PolSARProSim'
    f.attrs['processor'] = 'N/A'

    print('kapok.pspsim.load | Stack ID: '+f.attrs['stack_name'])
    print('kapok.pspsim.load | Band: '+f.attrs['band'])

    print('kapok.pspsim.load | Importing metadata. ('+time.ctime()+')')

    f.attrs['average_altitude'] = reference_alt
    f.attrs['image_starting_slant_range'] = 'N/A'
    f.attrs['slc_azimuth_pixel_spacing'] = 'N/A'
    f.attrs['slc_slant_range_pixel_spacing'] = 'N/A'
    f.attrs['cov_azimuth_pixel_spacing'] = 'N/A'
    f.attrs['cov_slant_range_pixel_spacing'] = 'N/A'

    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_baselines'] = num_bl
    f.attrs['tracks'] = 'N/A'

    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts

    f.attrs['peg_latitude'] = 'N/A'
    f.attrs['peg_longitude'] = 'N/A'
    f.attrs['peg_heading'] = 'N/A'

    f.attrs['subset'] = False

    # Multi-looked, subsetted, image dimensions:
    # print('***')
    # print(nx)
    # print(ny)
    # print('***')
    azsize = slc_shape[0] // mlwin[0]
    rngsize = slc_shape[1] // mlwin[1]
    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['dim_slc'] = (slc_shape[0], slc_shape[1])
    f.attrs['ml_window'] = mlwin
    f.attrs['sm_window'] = smwin

    # Load SLC stack.
    for track_num in np.arange(num_tracks):
        for pol_num, pol in enumerate(np.array(['11', '12', '22'])):
            slcfile = os.path.join(inpath, 'S{}_{}.slc.npy'.format(pol, track_num+1))
            slc_stack[track_num*num_pol + pol_num] = np.load(slcfile)
    
    # Load kz stack.
    for track_num in np.arange(num_tracks):
        kzfile = os.path.join(inpath, 'kz_{}.kz.npy'.format(track_num+1))
        kz_stack[track_num] = np.load(kzfile)

    # Initialize covariance matrix dataset in HDF5 file:
    cov = f.create_dataset('cov', (azsize, rngsize, num_cov_elements, num_cov_elements), dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)

    # Calculate covariance matrix.
    for row in range(0,num_cov_elements):
        for col in range(row,num_cov_elements):
            print('kapok.pspsim.load | Calculating covariance matrix element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')     ', end='\r')
            reference_slc = slc_stack[row]
            secondary_slc = slc_stack[col]

            if (row % 3) == 1: # Cross Polarization
                reference_slc = np.sqrt(2) * reference_slc
            if np.any(np.abs(reference_slc) < 1e-5):
                reference_slc[np.abs(reference_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.
            
            if (col % 3) == 1: # Cross Polarization
                secondary_slc = np.sqrt(2) * secondary_slc
            if np.any(np.abs(secondary_slc) < 1e-5):
                secondary_slc[np.abs(secondary_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.
            
            # print(azsize)
            # print(rngsize)
            # print(reference_slc.shape)
            # print(secondary_slc.shape)
            # print(mlwin)
            cov[:,:,row,col] = nanmlook(reference_slc*np.conj(secondary_slc),mlwin)
    
    f.flush()

    # Boxcar Smoothing:
    if smwin != (1,1):
        print('kapok.pspsim.load | Boxcar averaging covariance matrix. ('+time.ctime()+')')
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                cov[:,:,row,col] = smooth(cov[:,:,row,col],smwin)

    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*VH', 'VV'],dtype='S')
    print('kapok.pspsim.load | Covariance matrix calculation complete. ('+time.ctime()+')     ')

    # # Save tree location data.
    # if trees is not None:
    #     trees_azimuth_pixel = (trees['x'] - x[0]) / (dx*mlwin[0])
    #     trees_range_pixel = (trees['y'] - y[0]) / (dy*mlwin[1])

    #     treex = f.create_dataset('ancillary/trees/row', data=trees_azimuth_pixel, compression=compression, compression_opts=compression_opts)
    #     treey = f.create_dataset('ancillary/trees/col', data=trees_range_pixel, compression=compression, compression_opts=compression_opts)
    #     treeh = f.create_dataset('ancillary/trees/height', data=trees['height'], compression=compression, compression_opts=compression_opts)
    #     treer = f.create_dataset('ancillary/trees/radius', data=trees['radius'], compression=compression, compression_opts=compression_opts)

    # Load geolocation data.
    print('kapok.pspsim.load | Loading geolocation data and setting up coordinate grids. ('+time.ctime()+')')

    # Initialize latitude dataset and import values:
    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lat[:] = nanmlook(np.load(os.path.join(inpath, 'ref.lat.npy')), mlwin)
    lat.attrs['units'] = 'Degrees'
    lat.attrs['description'] = 'N/A'

    # Initialize longitude dataset and import values:
    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lon[:] = nanmlook(np.load(os.path.join(inpath, 'ref.lon.npy')), mlwin)
    lon.attrs['units'] = 'Degrees'
    lon.attrs['description'] = 'N/A'

    # Initialize DEM height dataset and import values:
    print('kapok.pspsim.load | Populating DEM heights with zeros. ('+time.ctime()+')')
    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = 0.0
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Zero meter simulated DEM.'

    # Create dataset tracking which tile number was used for each pixel.
    print('kapok.pspsim.load | Loading tile indices for batch simulation. ('+time.ctime()+')')
    tile = f.create_dataset('tile', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    tile[:] = nanmlook(np.load(os.path.join(inpath,'ref.tile.npy')), mlwin)
    tile.attrs['description'] = 'Tile index from before merging batch simulations.'

    # Create canopy height dataset.
    chm = f.create_dataset('ancillary/chm', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    chm_sl = np.load(os.path.join(inpath, 'ref.chm.npy'))
    chm_sl[chm_sl <= 0.1] = np.nan # don't want to average in 0 values (non-forested) into CHM
    chm[:] = nanmlook(chm_sl, mlwin)
    chm.attrs['units'] = 'meters'
    chm.attrs['description'] = 'Simulated canopy height.'

    # Create dummy gfc treecover, datamask, and lossyear rasters with fake data.
    gfc_treecover2000 = f.create_dataset('ancillary/gfc_treecover2000', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_treecover2000[:] = 100
    gfc_datamask = f.create_dataset('ancillary/gfc_datamask', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_datamask[:] = 1
    gfc_lossyear = f.create_dataset('ancillary/gfc_lossyear', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    gfc_lossyear[:] = 0

    f.flush()

    # Incidence Angle of Reference Track
    print('kapok.pspsim.load | Importing incidence angles from reference track RGI product. ('+time.ctime()+')')
    # Initialize incidence angle dataset (stores reference track incidence angle):
    inc = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Reference Track Local Incidence Angle'
    inc.attrs['slope_corrected'] = False # PolSARProSim incidence angle is reference to ellipsoid, no terrain
    inc[:] = mlook(np.radians(inc_ref), mlwin)

    # Initialize kz dataset:
    kz = f.create_dataset('kz', (num_tracks, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    kz.attrs['indexing'] = 'track'
    kz.attrs['slope_corrected'] = False

    # Load kz files.
    for tr in range(0,num_tracks):
        kz[tr,:,:] = nanmlook(kz_stack[tr], mlwin)

    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.pspsim.load | Complete. ('+time.ctime()+')     ')
    return kapok.Scene(outfile)
