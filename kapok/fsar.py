# -*- coding: utf-8 -*-
"""F-SAR Data Import Functions

    Module for importing F-SAR data.

    Author: Michael Denbina

    Copyright 2022-2023 California Institute of Technology.  All rights reserved.
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
import pyresample as pr
import utm

import kapok
from kapok.lib import mlook, smooth


def load(inpath, outfile, band='L', mlwin=(8,2), smwin=(1,1), tracks=None,
         use_VH_pol=False, compression='gzip', compression_opts=4,
         overwrite=False):
    """Load an F-SAR dataset into the Kapok format, and save it to target HDF5
    file.

    Arguments:
        inpath (str): Path to the F-SAR data directory containing the PS02,
            PS03, etc. directories.
        outfile (str): Path and filename for the saved HDF5 dataset.
        band (str): Set to 'L' to load L-band data, or 'P' to load P-band data.
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
        tracks: List containing desired track indices to import.  For example,
            tracks=[0,1] will import only the first two tracks listed in the
            annotation file.  Default: All tracks imported.
        use_VH_pol (bool): Set to True to use VH polarization instead of HV
            polarization.  Helpful for some F-SAR flight lines which are
            missing HV polarization data but which do contain VH.
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
        print('kapok.fsar.load | Cannot load F-SAR data from input path. Aborting.')
        return

    # Create the HDF5 file.
    if overwrite:
        try:
            f = h5py.File(outfile,'w')
        except:
            print('kapok.fsar.load | Cannot create new HDF5 file. Check if path is valid. Aborting.')
            return
    else:
        try:
            f = h5py.File(outfile,'x')
        except:
            print('kapok.fsar.load | Cannot create new HDF5 file. Check if path is valid and ensure file does not already exist. Aborting.')
            return

    # Figure out path for each track, and get SLC dimensions.
    track_path_glob = glob(os.path.join(inpath, '*/T{}*'.format(band)))
    track_path_glob = np.sort(track_path_glob)
    track_names = np.array([])

    # For each path, check detected products of RGI, INF, GTC and make sure
    # we have everything needed.
    print('kapok.fsar.load | Checking Tracks')
    for track_num, track_path in enumerate(track_path_glob):
        print('kapok.fsar.load | Checking Track ID {} ("{}")...'.format(track_num, track_path))

        if not os.path.isdir(os.path.join(track_path, 'RGI')):
            print('kapok.fsar.load | Warning: No RGI product found for track: "{}" (track ID {}).'.format(track_path, track_num))
            # return
        if not os.path.isdir(os.path.join(track_path, 'INF')):
            if track_num > 0: # don't need INF for first (reference track)
                print('kapok.fsar.load | Warning: No INF product found for track: "{}" (track ID {}).'.format(track_path, track_num))
                # return
        if not os.path.isdir(os.path.join(track_path, 'GTC')):
            if track_num == 0:
                print('kapok.fsar.load | Warning: No GTC product found for track: "{}" (track ID {}).'.format(track_path, track_num))
                # return

        track_names = np.append(track_names, track_path.split('/')[-2])

    print('kapok.fsar.load | Number of Tracks: {}'.format(len(track_path_glob)))

    # Subset track names if desired tracks were specified:
    first_track = track_names[0]
    if tracks is not None:
        tracks = np.array(tracks, dtype='int')
        track_names = track_names[tracks]
        track_path_glob = track_path_glob[tracks]
        print('kapok.fsar.load | Number of Tracks After Subsetting: {}'.format(len(track_path_glob)))

    # Get SLC dimensions from reference track SLC header data.
    ref_path = os.path.join(track_path_glob[0], 'INF/INF-SR')
    if not os.path.isdir(ref_path):
        ref_path = os.path.join(track_path_glob[0], 'RGI/RGI-SR')

    ref_header_file = glob(os.path.join(ref_path, 'slc*hh*.rat.hdr'))[0]
    ref_header = load_rat_header(ref_header_file)
    azsize_slc = ref_header['lines']
    rngsize_slc = ref_header['samples']

    num_tracks = len(track_path_glob) # Number of Tracks
    num_bl = int(num_tracks * (num_tracks-1) / 2) # Number of Baselines
    num_pol = int(3) # Number of Polarizations (HH, sqrt(2)*cross-pol, VV)
    num_cov_elements = num_tracks*num_pol # Number of Covariance Matrix Elements in Each Row

    f.attrs['stack_name'] = os.path.basename(os.path.normpath(inpath))
    f.attrs['band'] = band
    f.attrs['site'] = 'N/A'
    f.attrs['url'] = 'N/A'
    f.attrs['sensor'] = 'F-SAR'
    f.attrs['processor'] = 'N/A'

    print('kapok.fsar.load | Stack ID: '+f.attrs['stack_name'])
    print('kapok.fsar.load | Band: '+f.attrs['band'])

    print('kapok.fsar.load | Importing metadata. ('+time.ctime()+')')

    f.attrs['average_altitude'] = 'N/A'
    f.attrs['image_starting_slant_range'] = 'N/A'
    f.attrs['slc_azimuth_pixel_spacing'] = 'N/A'
    f.attrs['slc_slant_range_pixel_spacing'] = 'N/A'
    f.attrs['cov_azimuth_pixel_spacing'] = 'N/A'
    f.attrs['cov_slant_range_pixel_spacing'] = 'N/A'

    f.attrs['num_tracks'] = num_tracks
    f.attrs['num_baselines'] = num_bl
    f.attrs['tracks'] = np.array(track_names, dtype='S')

    f.attrs['compression'] = compression
    f.attrs['compression_opts'] = compression_opts

    f.attrs['peg_latitude'] = 'N/A'
    f.attrs['peg_longitude'] = 'N/A'
    f.attrs['peg_heading'] = 'N/A'

    f.attrs['subset'] = False

    # Multi-looked, subsetted, image dimensions:
    azsize = azsize_slc // mlwin[0]
    rngsize = rngsize_slc // mlwin[1]
    f.attrs['dim'] = (azsize, rngsize)
    f.attrs['dim_slc'] = (azsize_slc, rngsize_slc)
    f.attrs['ml_window'] = mlwin
    f.attrs['sm_window'] = smwin

    # Get filenames of SLCs for each track, in polarization order HH, cross-pol (HV or VH), VV.
    slcfiles = []
    phafiles = []

    if use_VH_pol:
        pol_list = ['HH','VH','VV']
    else:
        pol_list = ['HH','HV','VV']

    for tr in range(num_tracks):
        for pol in pol_list:
            file_pattern = os.path.join(track_path_glob[tr], 'INF/INF-SR/slc*{}*.rat'.format(pol.lower()))
            file = glob(file_pattern)
            if len(file) == 0:
                file_pattern = os.path.join(track_path_glob[tr], 'RGI/RGI-SR/slc*{}*.rat'.format(pol.lower()))
                file = glob(file_pattern)

            if len(file) == 1:
                slcfiles.append(file[0])
                phafiles.append(file[0].replace('slc_coreg', 'pha'))
                if phafiles[-1] == slcfiles[-1]:
                    phafiles[-1] = ''
            elif len(file) > 1:
                print('kapok.fsar.load | Too many SLC files matching pattern: "'+file_pattern+'".  Aborting.')
                return
            else:
                print('kapok.fsar.load | Cannot find SLC file matching pattern: "'+file_pattern+'".  Aborting.')
                return

    # Initialize covariance matrix dataset in HDF5 file:
    cov = f.create_dataset('cov', (azsize, rngsize, num_cov_elements, num_cov_elements), dtype='complex64', compression=compression, shuffle=True, compression_opts=compression_opts)

    # Calculate covariance matrix.
    for row in range(0,num_cov_elements):
        for col in range(row,num_cov_elements):
            print('kapok.fsar.load | Calculating covariance matrix element: ('+str(row)+','+str(col)+'). ('+time.ctime()+')     ', end='\r')
            reference_slc_file = slcfiles[row]
            secondary_slc_file = slcfiles[col]

            # If pha files exist, use those phases as they seem to be cleaner
            # than calculating the interferogram and applying phadem.
            if (row >= num_pol):
                reference_phase_file = phafiles[row]
            else:
                reference_phase_file = None

            secondary_phase_file = phafiles[col]

            if os.path.isfile(secondary_phase_file):
                intf_phase = load_rat(secondary_phase_file, min_lines=azsize_slc)

                # If reference SLC is not F-SAR reference track, then need
                # to adjust the phase with the reference track's pha file
                # as well.
                if reference_phase_file is not None:
                    reference_phase = load_rat(reference_phase_file, min_lines=azsize_slc)
                    intf_phase -= reference_phase
                    del reference_phase

                # Compensate for different polarizations.
                ref_pol = (row % num_pol)
                sec_pol = (col % num_pol)
                if ref_pol != sec_pol:
                    ref_pol_slc = load_rat(slcfiles[ref_pol], min_lines=azsize_slc)
                    sec_pol_slc = load_rat(slcfiles[sec_pol], min_lines=azsize_slc)

                    intf_phase += np.angle(ref_pol_slc)
                    intf_phase -= np.angle(sec_pol_slc)
                    del ref_pol_slc, sec_pol_slc
            else:
                intf_phase = None

            if reference_phase_file is not None:
                reference_phase = load_rat(reference_phase_file, min_lines=azsize_slc)

            reference_slc = load_rat(reference_slc_file, min_lines=azsize_slc)
            if (row % 3) == 1: # Cross Polarization
                reference_slc = np.sqrt(2) * reference_slc
            if np.any(np.abs(reference_slc) < 1e-5):
                reference_slc[np.abs(reference_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.

            secondary_slc = load_rat(secondary_slc_file, min_lines=azsize_slc)
            if (col % 3) == 1: # Cross Polarization
                secondary_slc = np.sqrt(2) * secondary_slc
            if np.any(np.abs(secondary_slc) < 1e-5):
                secondary_slc[np.abs(secondary_slc) < 1e-5] = 1e-5 # Don't want zero-valued pixels.  Set floor to -100 dB.

            # print('Reference File: {}'.format(reference_slc_file))
            # print('Reference Shape: {}'.format(reference_slc.shape))
            # print('Secondary File: {}'.format(secondary_slc_file))
            # print('Secondary Shape: {}'.format(secondary_slc.shape))
            # if intf_phase is not None:
            #     print('Intf File: {}'.format(secondary_phase_file))
            #     print('Intf Shape: {}'.format(intf_phase.shape))

            if intf_phase is not None:
                cov[:,:,row,col] = mlook(np.abs(reference_slc*np.conj(secondary_slc))*np.exp(1j*intf_phase),mlwin)
            else:
                cov[:,:,row,col] = mlook(reference_slc*np.conj(secondary_slc),mlwin)

    del reference_slc, secondary_slc, intf_phase
    f.flush()

    # Boxcar Smoothing:
    if smwin != (1,1):
        print('kapok.fsar.load | Boxcar averaging covariance matrix. ('+time.ctime()+')')
        for row in range(0,num_cov_elements):
            for col in range(row,num_cov_elements):
                cov[:,:,row,col] = smooth(cov[:,:,row,col],smwin)

    cov.attrs['description'] = 'Covariance Matrix'
    cov.attrs['basis'] = 'lexicographic'
    cov.attrs['num_pol'] = 3
    if use_VH_pol:
        cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*HV', 'VV'],dtype='S')
    else:
        cov.attrs['pol'] = np.array(['HH', 'sqrt(2)*VH', 'VV'],dtype='S')
    print('kapok.fsar.load | Covariance matrix calculation complete. ('+time.ctime()+')     ')


    # Load geolocation data.
    print('kapok.fsar.load | Loading geolocation data and setting up coordinate grids. ('+time.ctime()+')')

    geo_az_pattern = os.path.join(track_path_glob[0], 'GTC/GTC-LUT/sr2geo_az*.rat')
    geo_az_file = glob(geo_az_pattern)
    if len(geo_az_file) == 1:
        geo_az_data = load_rat(geo_az_file[0])
    elif len(geo_az_file) > 1:
        print('kapok.fsar.load | Too many sr2geo_az files matching pattern: "'+geo_az_pattern+'".  Aborting.')
        # return
    else:
        print('kapok.fsar.load | Cannot find sr2geo_az file matching pattern: "'+geo_az_pattern+'".  Aborting.')
        # return

    geo_rg_pattern = os.path.join(track_path_glob[0], 'GTC/GTC-LUT/sr2geo_rg*.rat')
    geo_rg_file = glob(geo_rg_pattern)
    if len(geo_rg_file) == 1:
        geo_rg_data = load_rat(geo_rg_file[0])
    elif len(geo_rg_file) > 1:
        print('kapok.fsar.load | Too many sr2geo_rg files matching pattern: "'+geo_rg_pattern+'".  Aborting.')
        # return
    else:
        print('kapok.fsar.load | Cannot find sr2geo_rg file matching pattern: "'+geo_rg_pattern+'".  Aborting.')
        # return

    geo_h0_pattern = os.path.join(track_path_glob[0], 'GTC/GTC-LUT/sr2geo3d_h0*.rat')
    geo_h0_file = glob(geo_h0_pattern)
    if len(geo_h0_file) == 1:
        geo_h0_data = load_rat(geo_h0_file[0])
    elif len(geo_h0_file) > 1:
        print('kapok.fsar.load | Too many h0 files matching pattern: "'+geo_h0_pattern+'".  Aborting.')
        # return
    else:
        print('kapok.fsar.load | Cannot find h0 file matching pattern: "'+geo_h0_pattern+'".  Aborting.')
        # return

    # Set up input and output grids for pyresample.
    geo_header_file = geo_az_file[0] + '.hdr'
    geo_header = load_rat_header(geo_header_file)
    f.attrs['wavelength'] = geo_header['wavelength']
    geo_grid_lines = geo_header['lines']
    geo_grid_samples = geo_header['samples']
    geo_grid_northing_origin = geo_header['max_northing']
    geo_grid_northing_spacing = geo_header['pixel_spacing_north'] * -1
    geo_grid_easting_origin = geo_header['min_easting']
    geo_grid_easting_spacing = geo_header['pixel_spacing_east']
    geo_grid_projection_zone = geo_header['projection_zone']
    # if 'north' in geo_header['map info'].lower():
    #     geo_grid_northern = True
    #     geo_north_string = 'N'
    # elif 'south' in geo_header['map info'].lower():
    #     geo_grid_northern = False
    #     geo_north_string = 'S'
    # else:
    #     geo_grid_northern = True # by default assume North if not set
    #     geo_north_string = 'N'
    if geo_grid_projection_zone < 0:
        # Southern
        geo_grid_projection_zone = np.abs(geo_grid_projection_zone)
        geo_grid_northern = False
        geo_north_string = 'S'
    else:
        # Northern
        geo_grid_northern = True
        geo_north_string = 'N'

    
    print('kapok.fsar.load | Data geolocation grid is in UTM Zone {}{}.'.format(geo_grid_projection_zone, geo_north_string))

    geo_grid_northing = np.tile(np.arange(geo_grid_northing_origin, geo_grid_northing_origin + geo_grid_northing_spacing*geo_grid_lines, geo_grid_northing_spacing),
                                (geo_grid_samples, 1)).T
    geo_grid_northing = geo_grid_northing.astype('float32')

    geo_grid_easting = np.tile(np.arange(geo_grid_easting_origin, geo_grid_easting_origin + geo_grid_easting_spacing*geo_grid_samples, geo_grid_easting_spacing),
                                (geo_grid_lines, 1))
    geo_grid_easting = geo_grid_easting.astype('float32')

    no_data_mask = (geo_az_data == -9999) | (geo_rg_data == -9999) | (geo_h0_data == -9999)
    geo_az_masked = np.ma.array(geo_az_data, mask=no_data_mask, copy=True)
    geo_rg_masked = np.ma.array(geo_rg_data, mask=no_data_mask, copy=True)
    geo_h0_masked = np.ma.array(geo_h0_data, mask=no_data_mask, copy=True)

    # Scale azimuth and range coordinates and define input grid.
    max_dim = np.ceil(np.nanmax(geo_az_masked))
    geo_az_masked /= max_dim
    geo_rg_masked /= max_dim
    swath = pr.geometry.SwathDefinition(lons=geo_rg_masked, lats=geo_az_masked)

    # Set up output grid (SLC azimuth and range indices of each multi-looked
    # pixel in covariance matrix).
    slc_grid_az = np.tile(np.arange(1, azsize_slc+1), (rngsize_slc, 1)).T
    cov_grid_az = mlook(slc_grid_az, mlwin)
    cov_grid_az /= max_dim
    slc_grid_rg = np.tile(np.arange(1, rngsize_slc+1), (azsize_slc, 1))
    cov_grid_rg = mlook(slc_grid_rg, mlwin)
    cov_grid_rg /= max_dim
    area = pr.geometry.GridDefinition(lons=cov_grid_rg, lats=cov_grid_az)

    dx = 6378137. * np.radians(geo_grid_easting_spacing / max_dim)
    scale = pr.utils.fwhm2sigma(dx)

    # Resample UTM northing coordinates:
    print('kapok.fsar.load | Resampling Northing values. ('+time.ctime()+')')
    northing_res = pr.kd_tree.resample_gauss(swath, geo_grid_northing, area,
                                            radius_of_influence=3*dx,
                                            sigmas=scale, neighbours=36, segments=1)
    
    # Resample UTM easting coordinates:
    print('kapok.fsar.load | Resampling Easting values. ('+time.ctime()+')')
    easting_res = pr.kd_tree.resample_gauss(swath, geo_grid_easting, area,
                                            radius_of_influence=3*dx,
                                            sigmas=scale, neighbours=36, segments=1)
    
    # Convert UTM coordinates to lat/lon coordinates to save in HDF5 file.
    print('kapok.fsar.load | Converting UTM coordinates to lat/lon. ('+time.ctime()+')')
    lat_res = np.zeros((azsize, rngsize), dtype='float32')
    lon_res = np.zeros((azsize, rngsize), dtype='float32')
    ind_valid_geo = (easting_res != 0) | (northing_res != 0)

    if np.any(ind_valid_geo):
        lat_res[ind_valid_geo], lon_res[ind_valid_geo] = utm.to_latlon(easting_res[ind_valid_geo], northing_res[ind_valid_geo], int(geo_grid_projection_zone), northern=geo_grid_northern)
    
    if np.any(~ind_valid_geo):
        lat_res[~ind_valid_geo] = np.nan
        lon_res[~ind_valid_geo] = np.nan

    lat = f.create_dataset('lat', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lat[:] = lat_res
    lat.attrs['units'] = 'degrees'
    lat.attrs['description'] = 'Latitude'

    lon = f.create_dataset('lon', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    lon[:] = lon_res
    lon.attrs['units'] = 'degrees'
    lon.attrs['description'] = 'Longitude'

    # Initialize DEM height dataset and import values:
    print('kapok.fsar.load | Resampling geolocated ellipsoid heights. ('+time.ctime()+')')
    h0_res = pr.kd_tree.resample_gauss(swath, geo_h0_masked, area,
                                       radius_of_influence=3*dx,
                                       sigmas=scale, neighbours=36, segments=1)

    dem = f.create_dataset('dem', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    dem[:] = h0_res
    dem.attrs['units'] = 'meters'
    dem.attrs['description'] = 'Ellipsoid Height used to generate F-SAR geolocation.'

    f.flush()
    del geo_grid_northing, geo_grid_easting, northing_res, easting_res, lat_res, lon_res, h0_res

    # Incidence Angle of Reference Track
    print('kapok.fsar.load | Importing incidence angles from reference track RGI product. ('+time.ctime()+')')
    incidence_pattern = os.path.join(track_path_glob[0], 'RGI/RGI-SR/incidence*.rat')
    incidence_file = glob(incidence_pattern)

    if len(incidence_file) == 1:
        incidence_data = load_rat(incidence_file[0])
    elif len(incidence_file) > 1:
        print('kapok.fsar.load | Too many incidence files matching pattern: "'+incidence_pattern+'".  Aborting.')
        return
    else:
        print('kapok.fsar.load | Cannot find incidence file matching pattern: "'+incidence_pattern+'".  Aborting.')
        return

    # Initialize incidence angle dataset (stores reference track incidence angle):
    inc = f.create_dataset('inc', (azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    inc.attrs['units'] = 'radians'
    inc.attrs['description'] = 'Reference Track Local Incidence Angle'
    inc.attrs['slope_corrected'] = True # F-SAR incidence angle is local incidence angle including terrain
    inc[:] = mlook(incidence_data, mlwin)[0:azsize, :]

    # Initialize kz dataset:
    kz = f.create_dataset('kz', (num_tracks, azsize, rngsize), dtype='float32', compression=compression, compression_opts=compression_opts)
    kz.attrs['units'] = 'radians/meter'
    kz.attrs['description'] = 'Interferometric Vertical Wavenumber'
    kz.attrs['indexing'] = 'track'
    kz.attrs['slope_corrected'] = False # is F-SAR kz slope corrected?  it does not appear to be...

    # Load kz files.
    for tr in range(1,num_tracks):
        print('kapok.fsar.load | Importing kz between track '+str(tr)+' and reference track. ('+time.ctime()+')')
        kz_file_pattern = os.path.join(track_path_glob[tr], 'INF/INF-SR/kz*hh*.rat')
        kz_file = glob(kz_file_pattern)
        if len(kz_file) == 1:
            kz_data = load_rat(kz_file[0], min_lines=azsize_slc)
            kz[tr,:,:] = mlook(kz_data, mlwin)
        elif len(kz_file) > 1:
            print('kapok.fsar.load | Too many kz files matching pattern: "'+kz_file_pattern+'".  Aborting.')
            return
        else:
            print('kapok.fsar.load | Cannot find kz file matching pattern: "'+kz_file_pattern+'".  Aborting.')
            return

    # Close the file, then return it as a Scene object.
    f.close()
    print('kapok.fsar.load | Complete. ('+time.ctime()+')     ')
    return kapok.Scene(outfile)


def load_rat_header(header_file):
    """ Load .rat header file and return information as a dict. """
    header_dict = {}

    with open(header_file, 'r') as fd:
        header_list = fd.read().split('\n')

    for n in range(len(header_list)):
        split_line = header_list[n].split('=')
        if len(split_line) > 1:
            key = split_line[0].strip()
            val = split_line[-1].strip()
            try:
                val = float(val) # if we can convert the string to a number, do so
                if (val - np.floor(val)) == 0:
                    val = int(val)  # if it's an integer, convert it to one (e.g., number of samples)
            except: # if we can't convert the string to a number, leave it as a string
                pass

            header_dict[key] = val

    return header_dict


def load_rat(file, min_lines=None):
    """ Load RAT file and return as a NumPy array. """
    if os.path.isfile(file + '.hdr'):
        header_file = file + '.hdr'
    elif os.path.isfile(file.replace('.rat', '.hdr')):
        header_file = file.replace('.rat', '.hdr')
    else:
        print('kapok.fsar.load_rat | Error: RAT header file not found for {}!'.format(file))

    header_dict = load_rat_header(header_file)

    file_dim = (header_dict['lines'], header_dict['samples'])

    if header_dict['data type'] == 4:
        file_dtype = 'float32'
    elif header_dict['data type'] == 6:
        file_dtype = 'complex64'

    data = np.fromfile(file, dtype=file_dtype, offset=1000)
    data = data.reshape(file_dim)

    if min_lines is not None:
        if file_dim[0] < min_lines:
            pad_length = min_lines - file_dim[0]
            data = np.pad(data, ((0, pad_length), (0,0)))

    return data
