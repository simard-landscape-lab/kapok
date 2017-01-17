# -*- coding: utf-8 -*-
"""UAVSAR Data Import

    Module for importing UAVSAR data.  Imports the SLCs and calculates the
    covariance matrix, imports the UAVSAR annotation (metadata) file, and
    imports/calculates the necessary parameters for the viewing geometry
    (incidence angle, kz).  The imported data is saved as an HDF5 file
    which can be loaded into a Scene object using the main Kapok module.
    
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
import numpy as np

try: # Import Cython Implementation
    import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
    from .uavsarc import load, Ann, findsegment, getslcblock, quicklook
except ImportError: # Cython Import Failed
    print('kapok.uavsar | WARNING: Cython import failed.  Using native Python (will be slow).')
    from .uavsarp import load, Ann, findsegment, getslcblock, quicklook