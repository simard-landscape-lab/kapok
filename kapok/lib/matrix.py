# -*- coding: utf-8 -*-
"""Covariance matrix helper functions.

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


def makehermitian(m):
    """Given a matrix with elements below the diagonal equal to zero, fill
        those elements assuming the matrix is Hermitian. Note: Changes array
        in place!  (But also returns it.)
        
        Arguments:
            m (array): Matrices to make Hermitian.  Should have dimensions:
                [az, rng, n, n].  n is the number of rows and columns in the
                matrix (which need to be equal).
                
        Returns:
            m (array): Hermitian symmetric form of input.
    
    """
    if m.ndim == 4:
        for row in range(1,m.shape[2]):
            for col in range(0,row):
                m[:,:,row,col] = np.conj(m[:,:,col,row])
    else:
        for row in range(1,m.shape[0]):
            for col in range(0,row):
                m[row,col] = np.conj(m[col,row])
            
    return m