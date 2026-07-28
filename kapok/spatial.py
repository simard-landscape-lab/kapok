# -*- coding: utf-8 -*-
"""Spatial Filtering Module.

    This module contains implementations of filters for smoothing the
    covariance matrix using spatial filters other than the standard
    boxcar moving average.  Note that if you do wish to filter the
    covariance matrix using a boxcar moving average, this can be performed
    during the data import process.  This module is for more sophisticated
    filters which consider the similarity of pixels before averaging.
    
    Currently implemented filters:
        Non-Local Means: Implemented in kapok.spatial.nlmfilter().
            Essentially just an interface to the nlsartoolbox Python
            library, which is available from:
            https://www.math.u-bordeaux.fr/~cdeledal/nlsar.php
        Segmentation: Uses segmentation as implemented in the rsgislib
            Python library to segment the radar scene using certain
            features.  Then averages the covariance matrix within each
            feature.  The rsgislib library is available from:
            http://www.rsgislib.org/

    The actual implementation of this module is in Cython for improved
    speed, in the file spatial.pyx.  This file is just a wrapper that
    imports the Cython code.    
    
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

import numpy as np
import h5py

import kapok

try: # Import Cython Implementation
    import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
    from .spatialc import segfilter, nlmfilter
except ImportError: # Cython Import Failed
    print('kapok.spatial | WARNING: Cython import failed.  Aborting.  (This module has no native Python implementation.)')
