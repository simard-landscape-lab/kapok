# -*- coding: utf-8 -*-
"""Basic Kapok Processing Example

This Python script demonstrates a basic sequence of processing steps to
estimate and export forest canopy heights from a UAVSAR SLC stack dataset
using the Kapok Python library.  Examples of basic data visualization using
the kapok.vis and kapok.region modules are also given.

UAVSAR SLC stack data is available for download at
https://uavsar.jpl.nasa.gov/.  UAVSAR data courtesy NASA/JPL-Caltech.  A free
user registration is required to download SLC stack data.  In the Data Search,
under "Processing Modes", make sure to select "TomoSAR".

Source and destination filenames are hardcoded near the top of this script.
If you wish to run this script, make sure to change these parameters to match
the location where you have downloaded the UAVSAR SLC stack you are working
with, and the location where you wish to save the created files.  There are
also other user-specified options you may wish to change based on your
preferences.  For a full list of possible processing options, please refer to
the Kapok documentation and user's manual.
    
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
###########
# IMPORTS #
###########
import os
import os.path

import numpy as np

import kapok



###################################
# FILE LOCATIONS AND USER OPTIONS #
###################################
# Below is a path and filename pointing to the annotation (.ann) metadata file
# for a UAVSAR stack:
annfile = '/data/stack/filename.ann'

# This is the path and filename pointing to the HDF5 file where the imported
# data and derived parameters should be stored:
datafile = '/data/kapokfile.hdf5'

# This is an output path pointing to a folder where plotted figures and
# geocoded output products will be saved:
outpath = '/data/kapok_output/'

# Dimensions of the multi-looking window applied to the data.
# First index is the azimuth size, second index is the range size.
mlwin = [20,5]



#############
# LOAD DATA #
#############
# If the output path does not exist, create it.
if not os.path.exists(outpath):
    os.makedirs(outpath)

# First, get the Kapok Scene object.  If the file already exists, load it.
# Otherwise, import from UAVSAR data.
if os.path.isfile(datafile):
    scene = kapok.Scene(datafile)
else:
    import kapok.uavsar
    scene = kapok.uavsar.load(annfile,datafile,mlwin=mlwin)



#######################
# OPTIMIZE COHERENCES #
#######################
# Perform phase diversity coherence optimization.  This can easily be
# performed using the Scene object's .opt method.
# The resulting optimized coherences are saved into the HDF5 file.
# After optimization, they are accessible using Scene.pdcoh.
scene.opt()



#####################
# PLOTTING EXAMPLES #
#####################
# Display and save some basic overview images.
# Plotting of raster images is done through the Scene object's .show method.
scene.show('paulirgb', savefile=outpath+'paulirgb.png') # Pauli RGB color composite.
scene.show('coh', pol='high', bl=0, savefile=outpath+'coh_high_bl0.png') # Image of the high coherence.
scene.show('coh', pol='low', bl=0, savefile=outpath+'coh_low_bl0.png') # Image of the low coherence.
scene.show('coh mag', pol='high', bl=0, savefile=outpath+'coh_high_mag_bl0.png') # Magnitude image of the high cohernece.
scene.show('kz', savefile=outpath+'kz_bl0.png') # Image of the kz Values derived from the platform and viewing geometry.


# Instead of displaying the whole image, we can display and save some images for a subset of the scene bounded by azimuth indices 2000-3500.
scene.show('paulirgb',bounds=(2000,3500),savefile=outpath+'paulirgb_az_2000_3500.png')
scene.show('power',pol='HV',bounds=(2000,3500),savefile=outpath+'hvpower_az_2000_3500.png') # Backscattered HV power, in dB.
scene.show('coh', pol='high', bounds=(2000,3500), savefile=outpath+'coh_high_bl0_az_2000_3500.png')
scene.show('coh', pol='low', bounds=(2000,3500), savefile=outpath+'coh_low_bl0_az_2000_3500.png')


# The following closes all open figures.  Useful if we are plotting a lot of
# things, and saving them to files, and do not want them to clutter the
# screen after the plotting is finished:
scene.show('close')

# Interactive Coherence Region Plot for Pixel With Coordinates (2500,50)
scene.region(2500,50,mode='interactive')



##############################
# MASK LOW BACKSCATTER AREAS #
##############################
# Now, we create a mask which identifies low HV backscatter areas.
mask = scene.power('HV') # Get the HV backscattered power (in linear units).
mask[mask <= 0] = 1e-10 # Get rid of zero-valued power.
mask = 10*np.log10(mask) # Convert to dB.
mask = mask > -22 # Find pixels above/below -22 dB threshold.

# If this mask is provided to the model inversion, only pixels with HV
# sigma-nought over -22 dB will be considered valid pixels for the forest
# height estimation.  This will also save some computation time, since
# these pixels will be skipped over by the algorithm.



##########################
# ESTIMATE FOREST HEIGHT #
##########################
# Model inversion is performed through the .inv method of the Scene object.
# The name and desc keywords allow us to name and give a short description of
# the estimated forest heights.  The forest heights are saved to the HDF5 file,
# and are tagged with many identifying attributes.
rvog = scene.inv('rvog', name='rvog', desc='RVoG, hv and ext. free parameters, no temporal decorrelation.', mask=mask, overwrite=True)

# If we had wanted to do sinc model inversion, we could do so using
# scene.inv('sinc').  There are also many other options for the model
# inversion process (with different options being valid for different
# models).  See the scene.inv function header for more details.

# Since we named the results of this inversion 'rvog' using the name keyword
# of the .inv method, the estimated parameters are now stored in the HDF5 file
# in the group: /products/rvog/.  The estimated forest height is stored in the
# dataset 'products/rvog/hv'.  The estimated extinction parameter values are
# stored in 'products/rvog/ext'.  The estimated complex ground coherence is
# stored in 'products/rvog/ground'.



##################
# OUTPUT RESULTS #
##################
# Display and save an image of the estimated forest heights for a subset:
scene.show('rvog/hv', bounds=(2000,3500), vmax=30, savefile=outpath+'rvog_hv_az_2000_3500.png')

# Output geocoded forest height map as an ENVI grd/hdr file.
scene.geo('rvog/hv', outpath+'rvog.grd')