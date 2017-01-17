# -*- coding: utf-8 -*-
"""Terrain slope calculation, and ground range spacing calculation from
    DEM.

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


def calcslope(dem, spacing, inc):
    """Calculate terrain slope angles.
    
    Given an input DEM in radar (azimuth, slant range) coordinates, the DEM
    pixel spacing (in azimuth and slant range), and the incidence angle,
    calculate and return the azimuth and ground range slope angles (in
    radians).
    
    Arguments:
        dem: an array containing the DEM heights.
        spacing: a tuple containing the (azimuth, slant range) pixel spacing
            of the DEM, in meters.
        inc: the incidence angle, in radians.
      
    Returns:
        rngslope: the terrain slope angle in the ground range direction
        azslope: the slope angle in the azimuth direction
    
    """    
    (azslope,rngslope) = np.gradient(dem)
    
    azslope = np.arctan(azslope / spacing[0])
    rngslope = np.arctan(rngslope / ((spacing[1]/np.sin(inc)) + (rngslope/np.tan(inc))))
        
    return rngslope, azslope
    
    
def calcgrspacing(dem, spacing, inc):
    """Calculate ground range pixel spacing.
    
    Given an input DEM in radar (azimuth, slant range) coordinates, the DEM
    pixel spacing (in azimuth and slant range), and the incidence angle,
    calculate and return the ground range spacing.
    
    Arguments:
        dem: an array containing the DEM heights.
        spacing: the slant range spacing of the DEM, in meters.
        inc: the incidence angle, in radians.
      
    Returns:
        grspacing: Ground range spacing for each pixel, in meters.
    
    """    
    (azgrad,srgrad) = np.gradient(dem)
    grspacing = ((spacing/np.sin(inc)) + (srgrad/np.tan(inc)))
    
    return grspacing