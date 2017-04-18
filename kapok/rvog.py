# -*- coding: utf-8 -*-
"""Random Volume Over Ground (RVoG) Forest Model Inversion

    Contains functions for the forward RVoG model, and inversion.  RVoG model
    is formulated to include real-valued volumetric temporal decorrelation,
    as described in:
    
    S. R. Cloude and K. P. Papathanassiou, "Three-stage inversion process
    for polarimetric SAR interferometry," in IEE Proceedings - Radar, Sonar
    and Navigation, vol. 150, no. 3, pp. 125-134, 2 June 2003.
    doi: 10.1049/ip-rsn:20030449
    
    Model inversion can be accessed procedurally through this module's
    functions directly, or in an object-oriented fashion via the Scene class
    method 'inv'.
    
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
    from .rvogc import rvoginv, rvogfwdvol, rvogblselect
except ImportError: # Cython Import Failed
    print('kapok.rvog | WARNING: Cython import failed.  Using native Python (will be slow).')
    from .rvogp import rvoginv, rvogfwdvol, rvogblselect