# -*- coding: utf-8 -*-
"""Coherence Optimization Module.

    Currently contains an implementation of the phase diversity coherence
    optimization algorithm which finds the two coherences with the largest
    separation in the complex plane.  The actual code for this module is in
    cohoptc.pyx.  This file is just a wrapper that imports the Cython code.
    
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
    from .cohoptc import pdopt, pdopt_pixel
except ImportError: # Cython Import Failed
    print('kapok.cohopt | WARNING: Cython import failed.  Running in native Python (will be slow!).')
    from .cohoptp import pdopt, pdopt_pixel