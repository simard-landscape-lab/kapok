# -*- coding: utf-8 -*-
"""AfriSAR Canopy Height Script for Pongara

This Python script demonstrates how the publicly released canopy height
products for the AfriSAR campaign were created using the Kapok Python library.
This script contains the code used to generate the products for the Pongara
National Park study area in the country of Gabon.

UAVSAR SLC stack data is available for download at
https://uavsar.jpl.nasa.gov/.  UAVSAR data courtesy NASA/JPL-Caltech.  A free
user registration is required to download SLC stack data.  In the Data Search,
under "Processing Modes", make sure to select "TomoSAR".

The Pongara SLC stack used to generate the released products can be found at:

https://uavsar.jpl.nasa.gov/cgi-bin/product.pl?jobName=pongar_TM275_01

Source and destination filenames are hardcoded near the top of this script.
If you wish to run this script, make sure to change these parameters to match
the location where you have downloaded the UAVSAR SLC stack for Pongara, and
the location where you wish to save the created files.  There are also other
user-specified options you may wish to change based on your preferences.
For a full list of possible processing options, please refer to the Kapok
documentation and user's manual.
    
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
nptodb = 20/np.log(10)
from scipy import ndimage

import kapok
import kapok.topo
from kapok.lib import calcslope, linesegmentdist



###################################
# FILE LOCATIONS AND USER OPTIONS #
###################################

# Path and filename of an annotation (.ann) file in the downloaded UAVSAR
# stack.
inputann = '/path/to/stack/pongar_TM275_16009_002_160227_L090HH_01_BC.ann'

# Path to save the generated files.
savepath = 'path/to/save/pongara/'

# Name of the Kapok HDF5 file to create containing the covariance matrix,
# coherences, and other data.  It will be saved to the savepath specified
# above.
kapokfile = 'pongara_kapok.h5'

# String identifying the name of the site.  Used in output filenames.
site = 'pongara'

# Size of the multilooking window to use when estimating the covariance
# matrix from the SLC data, in (azimuth, range) dimensions.
mlwin = (20,5)

# If you wish to use the .kz files downloaded as part of the UAVSAR SLC stack,
# set calculate_flat_kz = False.  If you wish to have Kapok calculate
# the kz assuming a flat earth, without terrain correction, set
# calculate_flat_kz = True.  For the Pongara dataset, we set
# calculate_flat_kz to True, as that area has relatively minor topographic
# variation.  For the Lope dataset, we set calculate_kz = False, as that area
# displays significant topography.  This option also determines if the
# terrain slope is used in the forest height estimation.
calculate_flat_kz = True

# A fixed extinction value to use for inversion of the random volume over
# ground model, if desired.  If you wish to solve for the extinction as a
# free parameter, set ext = None (default).  Note that Kapok uses Np/m as the
# units for the extinction parameter, so if you wish to use dB/m, you must
# divide by nptodb, which is equal to 20/log(10).  For example:
# ext = 0.1/nptodb # 0.1 dB/m fixed extinction
ext = None



#############
# LOAD DATA #
#############

if not os.path.exists(savepath):
    os.makedirs(savepath)
    os.makedirs(savepath+'supplementary_products/')

# Load the Kapok HDF5 file if it already exists.  If not, import the UAVSAR
# SLC stack to create it.
kapokfile = savepath + kapokfile
if os.path.isfile(kapokfile):
    scene = kapok.Scene(kapokfile)
else:
    import kapok.uavsar
    # Note: num_blocks in the below function call can be increased if you
    # have memory problems loading in the data.
    scene = kapok.uavsar.load(inputann,kapokfile,mlwin=mlwin,num_blocks=50,kzcalc=calculate_flat_kz)



#######################
# OPTIMIZE COHERENCES #
#######################

# The canopy-dominated and ground-dominated coherences are estimated using a
# coherence optimization algorithm implemented in the kapok.cohopt.pdopt()
# function.
scene.opt()



##############################
# MASK LOW BACKSCATTER AREAS #
##############################

# We wish to identify water and other low backscatter areas, as well as pixels
# with a high dynamic range of HV backscatter in their 3x3 local area
# (e.g., edges).  These areas will be excluded from the canopy height
# estimation.
hv_power = scene.power('hv')
hv_power[hv_power <= 1e-10] = 1e-10
hv_power = 10*np.log10(hv_power)

hv_localmax = ndimage.generic_filter(hv_power, np.nanmax, size=3)
hv_localmin = ndimage.generic_filter(hv_power, np.nanmin, size=3)

inc = np.degrees(scene.inc[:])
mask = hv_power > -30
mask[(inc < 35)] = hv_power[(inc < 35)] > -15
mask[(inc < 45) & (inc >= 35)] = hv_power[(inc < 45) & (inc >= 35)] > -21
mask[(inc < 55) & (inc >= 45)] = hv_power[(inc < 55) & (inc >= 45)] > -24
mask[(inc >= 55)] = hv_power[(inc >= 55)] > -28

mask_edges = (np.abs(hv_localmax-hv_localmin) < 20)

mask = mask & mask_edges



#################################
# CREATE CANOPY HEIGHT PRODUCTS #
#################################

# Name for the canopy height dataset that will be created in the HDF5 file.
if ext is not None:
    canopyheight_dataset_name = 'rvog_mb_fixedext'+str(ext)
else:
    canopyheight_dataset_name = 'rvog_mb'
    
# Should we correct the model inversion for the range terrain slope angle?
if calculate_flat_kz:
    rangeslope = None
else:
    rangeslope, azimuthslope = calcslope(scene.dem, scene.spacing, scene.inc)

# Perform the random volume over ground model inversion to estimate the canopy
# heights and other parameters.
# Note: If the RVoG inversion has already been run, the next line of code will
# not overwrite it!  If you wish to overwrite a previous run, set the
# overwrite keyword in the line below to True.
canopyheightdataset = scene.inv(method='rvog', name=canopyheight_dataset_name, desc='Multi-baseline RVoG model inversion.  Baseline selection performed using coherence line product criteria.',
                      bl='all', blcriteria='line', hv_min=0, hv_max=60, ext=ext, mask=mask, rngslope=rangeslope, overwrite=False)

# Get array of canopy heights.
canopyheight = scene.get(canopyheight_dataset_name+'/hv')

# Export geocoded canopy heights.
canopyheight[canopyheight < 2] = 0
scene.geo(canopyheight,savepath+site+'_afrisar_canopyheight.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Get array of baseline indices which were selected for each pixel, for use in
# supplementary products later.
baselinemap = scene.get(canopyheight_dataset_name+'/bl')



#################################
# CREATE SUPPLEMENTARY PRODUCTS #
#################################

# RADAR BACKSCATTER

# Export average HV backscatter across all tracks.
# Note: We export the power in linear units, so that it is in linear units
# when it's resampled.  We convert the geocoded backscatter to dB units
# afterward.
hv_power = np.zeros(scene.dim,dtype='float32')
for tr in range(scene.num_tracks):
    hv_power += scene.power('hv',tr=tr)/scene.num_tracks
hv_power[hv_power <= 1e-10] = 1e-10
scene.geo(hv_power,savepath+'supplementary_products/'+site+'_afrisar_HVbackscatter_linear.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')
del hv_power


# GEOMETRY PRODUCTS

# Export look angle (angle between reference track look vector and sensor
# nadir).
lookangle = np.degrees(scene.inc[:])
scene.geo(lookangle,savepath+'supplementary_products/'+site+'_afrisar_lookangle.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export ground range and azimuth terrain slope angles from SRTM DEM.
rangeslope, azimuthslope = calcslope(scene.dem, scene.spacing, scene.inc)
scene.geo(np.degrees(rangeslope),savepath+'supplementary_products/'+site+'_afrisar_rangeslopeangle.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')
scene.geo(np.degrees(azimuthslope),savepath+'supplementary_products/'+site+'_afrisar_azimuthslopeangle.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export merged kz values (the kz value of the baselines selected for each
# pixel).  The absolute value of the kz values is exported, with the sign
# ignored.
kzmerged = np.zeros(scene.dim,dtype='float32')
for bl in np.unique(baselinemap):
    kzmerged[baselinemap == bl] = scene.kz(bl)[baselinemap == bl]
kzmerged[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(np.abs(kzmerged),savepath+'supplementary_products/'+site+'_afrisar_kzmerged.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export ambiguity height for the merged kz values, equal to abs(2*pi/kz).
ambheight = np.abs(2*np.pi/kzmerged)
ambheight[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(ambheight,savepath+'supplementary_products/'+site+'_afrisar_ambiguityheight.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')


# POLINSAR COHERENCE PRODUCTS

# Get the observed canopy-dominated ("high"), and ground-dominated ("low")
# coherences.  Then, use them to estimate the ground coherence using a line
# fit procedure implemented in kapok.topo.groundsolver().
coh_high = np.ones(scene.dim,dtype='complex64')
coh_low = np.ones(scene.dim,dtype='complex64')
coh_ground = np.zeros(scene.dim,dtype='complex64')
for bl in np.unique(baselinemap):
    ground, groundalt, volindex = kapok.topo.groundsolver(scene.pdcoh[bl], kz=scene.kz(bl), returnall=True, silent=True)
    coh_high[baselinemap == bl] = np.where(volindex,scene.pdcoh[bl,1],scene.pdcoh[bl,0])[baselinemap == bl]
    coh_low[baselinemap == bl] = np.where(volindex,scene.pdcoh[bl,0],scene.pdcoh[bl,1])[baselinemap == bl]
    coh_ground[baselinemap == bl] = ground[baselinemap == bl]
del ground, groundalt, volindex

# Export high and low coherence magnitudes.
scene.geo(np.abs(coh_high),savepath+'supplementary_products/'+site+'_afrisar_canopycoh_mag.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')
scene.geo(np.abs(coh_low),savepath+'supplementary_products/'+site+'_afrisar_groundcoh_mag.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export high and low phase center heights.
def remove_negative_heights(pch, kz):
    ind = (pch < 0)
    while np.any(ind):
        pch[ind] += np.abs(2*np.pi/kz[ind])
        ind = (pch < 0)        
    return pch
    
high_pch = remove_negative_heights(np.angle(coh_high*np.conj(coh_ground))/kzmerged,kzmerged)
high_pch[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(high_pch,savepath+'supplementary_products/'+site+'_afrisar_canopycoh_pch.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

low_pch = remove_negative_heights(np.angle(coh_low*np.conj(coh_ground))/kzmerged,kzmerged)
low_pch[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(low_pch,savepath+'supplementary_products/'+site+'_afrisar_groundcoh_pch.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

del high_pch, low_pch


# DATA QUALITY PRODUCTS

# Export coherence separation mask.
sep = np.abs(coh_high - coh_low)
sep[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(sep,savepath+'supplementary_products/'+site+'_afrisar_masksep.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export length between origin and coherence line mask.
dist = linesegmentdist(0, coh_high, coh_low)
dist[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
scene.geo(dist,savepath+'supplementary_products/'+site+'_afrisar_maskloc.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')

# Export Cramer-Rao lower bound phase standard deviation, converted to height
# standard deviation.
nlooks = mlwin[0] * mlwin[1] / 1.44
cohsq = np.abs(coh_high) ** 2
err = np.sqrt((1-cohsq)/2/nlooks/cohsq)/np.abs(kzmerged)
err[canopyheight <= 0] = 0 # Mask out pixels without a canopy height.
err[err > 100] = 100 # Clip error to 100 m maximum.
scene.geo(err,savepath+'supplementary_products/'+site+'_afrisar_maskerr.tif',nodataval=-99,tr=0.000277777777777778,outformat='GTiff',resampling='pyresample')