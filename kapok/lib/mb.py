# -*- coding: utf-8 -*-
"""Multibaseline indexing functions.

    Conventions:    
    Covariance matrix dimensions: [az, rg, pol*tr, pol*tr]
    Baseline-independent data dimensions (e.g., inc): [az, rng]
    Baseline-dependent data dimensions (e.g., kz): [bl, az, rng]
    Phase diversity coherence dimensions: [bl, 2, az, rng]
    
    Baselines are numbered such that adding new tracks does not change the
    ordering of currently existing baselines.
    
    Omega matrix locations for each baseline index (baselines considered
    for i<j):
    
        [ T11  0   1   3   6   10  ... ]
        [     T22  2   4   7   11  ... ]
        [         T33  5   8   12  ... ]
        [             T44  9   13  ... ]
        [                 T55  14  ... ]
        [                     T66  ... ]
    
    All baseline and track and polarization indices start with 0.

    Author: Maxim Neumann
    
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


def mb_num_baselines(n_tr):
    """Returns number of baselines, based on given number of tracks."""
    return int(n_tr * (n_tr-1) / 2)
    
    
def mb_num_tracks(n_bl):
    """Returns number of tracks, based on given number of baselines."""
    return ((1 + np.sqrt(1 + 8*n_bl))/2).astype('int')
    

def mb_tr_index(bl):
    """Returns tuple of track indices for a specified baseline.

    Based on the triangular number calculations.
    """
    if bl >= 0:
        j = np.floor((1+np.sqrt(1+8*(bl)))/2).astype(int)
        i = (bl - j*(j-1)/2).astype(int)
        return (i,j)
    else:
        return None


def mb_bl_index(tr1, tr2):
    """Returns the baseline index for given track indices.

    By convention, tr1 < tr2. Otherwise, a warning is printed,
    and same baseline returned.
    """
    if tr1 == tr2:
        print("ERROR: No baseline between same tracks.")
        return None
    if tr1 > tr2:
        print("WARNING: tr1 expected < than tr2.")
    mx = max(tr1, tr2)
    bl = np.array(mx*(mx-1)/2 + min(tr1, tr2))
    return bl.astype(int)


def mb_cov_index(bl, pol=0, pol2=None, n_pol=3):
    """Returns row,col covariance matrix indices for given baseline and
        polarization index.

        If polb is not given, then same polarization is assumed (e.g. HH-HH,
        vs. HH-VV).
    """
    t1, t2 = mb_tr_index(bl)
    p1, p2 = pol, pol if pol2 is None else pol2

    return (t1*n_pol + p1, t2*n_pol + p2)