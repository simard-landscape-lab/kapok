# -*- coding: utf-8 -*-
"""Covariance matrix manipulation functions.

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
                (az, rng, n, n).  n is the number of rows and columns in the
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
    
    
def zerobelowdiag(m):
    """Given a Hermitian symmetric covariance matrix with all elements filled,
        set the redundant elements below the diagonal to zero, in order
        to save storage space.
    
        Note: Changes array in place!  (But also returns it.)
        
        Arguments:
            m (array): Matrices to cancel out the below diagonal elements.
                Should have dimensions: (az, rng, n, n).  n is the number of
                rows and columns in the matrix (which need to be equal).
                
        Returns:
            m (array): Compress form of input matrix.
    
    """
    if m.ndim == 4:
        for row in range(1,m.shape[2]):
            for col in range(0,row):
                m[:,:,row,col] = 0
    else:
        for row in range(1,m.shape[0]):
            for col in range(0,row):
                m[row,col] = 0
            
    return m
    

def convertT3toC3(m):
    """Change the polarization basis of a 3x3 Pauli coherency matrix
        to a 3x3 Lexicographic covariance matrix.
        
        Arguments:
            m (array): The matrices to change the basis of, with dimensions of
                (az, rng, 3, 3).  The third dimension contains the rows
                of the matrices, and the fourth dimension contains the columns.
                The matrices need to be 3x3.
                
        Returns:
            mc (array): The matrix with a Lexicographic polarization basis.
    
    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mc = np.zeros(x.shape, dtype=x.dtype)
    
    srt = np.sqrt(2)
    
    mc[0,0] = x[0,0] + x[1,0] + x[0,1] + x[1,1]
    mc[0,1] = srt*(x[0,2] + x[1,2])
    mc[0,2] = x[0,0] + x[1,0] - x[0,1] - x[1,1]
    mc[1,0] = srt*(x[2,0] + x[2,1])
    mc[1,1] = 2*x[2,2]
    mc[1,2] = srt*(x[2,0]-x[2,1])
    mc[2,0] = x[0,0] - x[1,0] + x[0,1] - x[1,1]
    mc[2,1] = srt*(x[0,2] - x[1,2])
    mc[2,2] = x[0,0] - x[1,0] - x[0,1] + x[1,1]
        
    mc /= 2
    mc = np.rollaxis(np.rollaxis(mc,-1),-1)
    return mc
    
    
def convertC3toT3(m):
    """Change the polarization basis of a 3x3 Lexicographic covariance matrix
        to a 3x3 Pauli coherency matrix.
        
        Arguments:
            m (array): The matrices to change the basis of, with dimensions of
                (az, rng, 3, 3).  The third dimension contains the rows
                of the matrices, and the fourth dimension contains the columns.
                The matrices need to be 3x3.
                
        Returns:
            mc (array): The matrix with a Pauli polarization basis.
    
    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mc = np.zeros(x.shape, dtype=x.dtype)
    
    srt = np.sqrt(2)
    
    mc[0,0] = x[0,0] + x[2,0] + x[0,2] + x[2,2]
    mc[0,1] = x[0,0] + x[2,0] - x[0,2] - x[2,2]
    mc[0,2] = srt*(x[0,1] + x[2,1])
    mc[1,0] = x[0,0] - x[2,0] + x[0,2] - x[2,2]
    mc[1,1] = x[0,0] - x[2,0] - x[0,2] + x[2,2]
    mc[1,2] = srt*(x[0,1] - x[2,1])
    mc[2,0] = srt*(x[1,0] + x[1,2])
    mc[2,1] = srt*(x[1,0] - x[1,2])
    mc[2,2] = 2*x[1,1]
        
    mc /= 2
    mc = np.rollaxis(np.rollaxis(mc,-1),-1)
    return mc
    
    
def rotateT3(m, psi):
    """Given a 3x3 Pauli basis coherency matrix and a rotation angle, apply
        a rotation in the polarization plane.
    
        Arguments:    
            m: An array containing the input coherency matrices, with
                dimensions of (az, rng, 3, 3).
            psi: The desired rotation angle, in radians.
        
        Returns:   
            mr: The coherency matrix rotated by psi.
    
    """
    x = np.rollaxis(np.rollaxis(m,-1),-1)
    mr = np.zeros(x.shape, dtype=x.dtype)    
    
    c = np.cos(2*psi)
    s = np.sin(2*psi)
    
    mr[0,0] = x[0,0]
    mr[0,1] = x[0,1]*c + x[0,2]*s
    mr[0,2] = x[0,1]*(-s) + x[0,2]*c
    mr[1,0] = x[1,0]*c + x[2,0]*s
    mr[1,1] = c*(x[1,1]*c + x[2,1]*s) + s*(x[1,2]*c + x[2,2]*s)
    mr[1,2] = (-s)*(x[1,1]*c + x[2,1]*s) + c*(x[1,2]*c + x[2,2]*s)
    mr[2,0] = x[1,0]*(-s) + x[2,0]*c
    mr[2,1] = c*(x[1,1]*(-s) + x[2,1]*c) + s*(x[1,2]*(-s) + x[2,2]*c)
    mr[2,2] = (-s)*(x[1,1]*(-s) + x[2,1]*c) + c*(x[1,2]*(-s) + x[2,2]*c)
    
    mr = np.rollaxis(np.rollaxis(mr,-1),-1)
    return mr