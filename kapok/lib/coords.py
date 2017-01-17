# -*- coding: utf-8 -*-
"""Coordinate system and transformation library.

    All units are in radians and meters.
    
    Classes:
        Ellipsoid
        Peg: Longitude, Latitude, Heading, Local Radius, in [rad, rad, rad, m]
        LLH: Longitude, Latitude, Height, in [rad, rad, h]
        XYZ: ECEF/XYZ system (Z=North, X=prime meridian, Y), in [m, m, m]
        SCH: Along-track (along Peg-heading), cross-track, and height, in
            [m, m, m]
        ENU: Local East, North, UP: x, y, z, relative to reference position
            (Peg)
    
    Notes:
        1. LLH are represented as Longitude, Latitude, and Height (right-hand
            system, x-y-z).
        2. In this module Longitude and Latitude, as well as all other angles,
            are in radians.
        3. Heading is from North, in clockwise direction.
    
    References:
        T. H. Meyer. Introduction to geometrical and physical geodesy:
        foundations of geomatics. ESRI Press, Redlands, California, 2010.
    
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
from numpy import (array, sqrt, sin, cos, arcsin, arctan, arctan2, degrees)


class Ellipsoid:
    """Ellipsoid representation."""
    def __init__ (self, semimajor_axis=None, eccentricity_sq=None,
                  semiminor_axis=None, select='WGS84', name=""):
        """Expects 'semimajor_axis' and ('eccentricity_sq' or
        'semiminor_axis'), or a predefined ellipsoid with 'select'."""
        if semimajor_axis is None:
            if select == 'WGS84':
                semimajor_axis = 6378137.
                eccentricity_sq = 0.00669437999015
                name = select
            else:
                print("Unknown select ellipsoid")
                raise Exception
        if eccentricity_sq is None and semiminor_axis is not None:
            eccentricity_sq = 1. - (semiminor_axis / semimajor_axis)**2
        self.a = self.semimajor_axis = semimajor_axis
        self.e2 = self.eccentricity_sq = eccentricity_sq
        self.e = self.eccentricy = sqrt(self.e2)
        self.b = self.semiminor_axis = self.a * sqrt(1. - self.e2)
        self.f = self.flattening = 1. - sqrt(1. - self.e2)
        self.ep2 = self.ep_squared = self.e2 / (1. - self.e2)
        self.name = name

    def radius_east(self, lat):
        """Radius of curvature in the east direction (lat in radians).
        Also called 'Radius of curvature in the prime vertical' N."""
        return self.a / sqrt(1. - self.e2 * sin(lat)**2)

    def radius_north(self, lat):
        """Radius of curvature in the north direction (lat in radians).
        Also called 'Radius of curvature in the meridian' M."""
        return (self.a*(1.-self.e2) / (1.-self.e2*sin(lat)**2)**1.5)

    def radius_local(self, lat, hdg):
        """Local radius of curvature along heading (lat, hdg in radians)
        Heading is from North (y, lat)!
        It is related to the 'Radius of curvature in the normal section',
        except of the different definition for the hdg/azimuth angle
        (shifted by 90 degrees).
        """
        er = self.radius_east(lat)
        nr = self.radius_north(lat)
        return er * nr / (er * cos(hdg)**2 + nr * sin(hdg)**2)


# Default ellipsoid
WGS84 = Ellipsoid(select="WGS84")


class Peg:
    """Peg representation.

    Attributes
    ----------
    lon : radians
        Longitude, in radians
    lat : radians
        Geodetic Latitude, in radians
    hdg : radians
        Heading, in radians, from North, probably in clockwise direction(?).
    radius : m
        Local Earth radius, in m
    ellipsoid : Ellipsoid, optional
        Ellipsoid object. Default is WGS-84.
    """

    def __init__(self, lon, lat, hdg, ellipsoid=WGS84):
        self.lon = lon
        self.lat = lat
        self.hdg = hdg
        self.radius = ellipsoid.radius_local(lat, hdg)
        self.ellipsoid = ellipsoid
        self._update_transformations()
    def _update_transformations(self):
        slon, clon = sin(self.lon), cos(self.lon)
        slat, clat = sin(self.lat), cos(self.lat)
        shdg, chdg = sin(self.hdg), cos(self.hdg)
        xyzP_to_enu = array([[0, shdg, -chdg],
                             [0, chdg,  shdg],
                             [1,    0,     0]])
        enu_to_xyz = enu_to_xyz_matrix(self.lon, self.lat)
        self.rotation_matrix = enu_to_xyz.dot(xyzP_to_enu)
        re = self.ellipsoid.radius_east(self.lat)
        p = array([re * clat * clon,
                   re * clat * slon,
                   re * (1.0 - self.ellipsoid.e2) * slat])
        up = self.radius * enu_to_xyz[:,2] # just take the third up vector
        self.translation_vector = p - up
    def __call__(self):
        return array([self.lon, self.lat, self.hdg])
    def __repr__(self):
        return("Peg Lon: {:.3f} deg; Lat: {:.3f}; Heading: {:.1f} deg"
               .format(degrees(self.lon),degrees(self.lat),degrees(self.hdg)))


class LLH:
    """Longitude, geodetic Latitude, Height (lon, lat in radians).

    Parameters
    ----------
    lon : float, array_like
        Longitude, in radians([-180, 180])
    lat : float, array_like
        Geodetic latitude, in radians([-90, 90])
    h : float, array_like
        Height above ellipsoid, in meters
    """
    def __init__(self, lon, lat, h=None):
        self.lon = lon
        self.lat = lat
        self.h = h if h is not None else (lon * 0.)
    def __call__(self):
        return array([self.lon, self.lat, self.h])
    def __repr__(self):
        return("Lon: {:.3f} deg; Lat: {:.3f}; Height: {:.1f} m"
               .format(degrees(self.lon),degrees(self.lat),self.h))
    def xyz(self, ellipsoid=WGS84):
        """Transform to ECEF XYZ coordinates."""
        r = ellipsoid.radius_east(self.lat)
        x = (r + self.h) * cos(self.lat) * cos(self.lon)
        y = (r + self.h) * cos(self.lat) * sin(self.lon)
        z = (r * (1. - ellipsoid.e2) + self.h) * sin(self.lat)
        return XYZ(x, y, z, ellipsoid)
    def enu(self, o_xyz=None, o_llh=None, ellipsoid=WGS84):
        """Transform to ENU, given ENU origin point o."""
        if o_xyz is not None: ellipsoid = o_xyz.ellipsoid
        return self.xyz(ellipsoid).enu(o_xyz=o_xyz,o_llh=o_llh)
    def sch(self, peg):
        """Transform to SCH coordinates, given Peg."""
        return self.xyz(peg.ellipsoid).sch(peg)


class XYZ:
    """ECEF XYZ cartesian geocentric coordinates.

    Parameters
    ----------
    x : float, array_like
        In direction of prime meridian (lon=0, lat=0).
    y : float, array_like
        In direction lon=90, lat=0
    z : float, array_like
        Close to the rotation axis, with direction North (lat=90).
    """
    def __init__(self, x, y, z, ellipsoid=WGS84):
        self.x = x
        self.y = y
        self.z = z
        self.ellipsoid = ellipsoid
    def __repr__(self):
        return("x: {} y: {} z: {}".format(*self()))
    def __call__(self):
        return array([self.x, self.y, self.z])
    def sch(self, peg):
        """Transform to SCH coordinates, given Peg."""
        xyzP = peg.rotation_matrix.T.dot(
            array([self.x,self.y,self.z])-peg.translation_vector)
        r = np.linalg.norm(xyzP)
        h = r - peg.radius
        c = peg.radius * arcsin(xyzP[2] / r)
        s = peg.radius * arctan2(xyzP[1], xyzP[0])
        return SCH(peg, s, c, h)
    def llh(self):
        lon = arctan2(self.y, self.x)
        pr = sqrt(self.x**2 + self.y**2) # projected radius
        alpha = arctan(self.z / (pr * sqrt(1.-self.ellipsoid.e2)))
        lat = arctan(
            (self.z + self.ellipsoid.ep2 * self.ellipsoid.b * sin(alpha)**3)
            /(pr - self.ellipsoid.e2 * self.ellipsoid.a * cos(alpha)**3))
        h = pr / cos(lat) - self.ellipsoid.radius_east(lat)
        return LLH(lon,lat,h)
    def enu(self, o_xyz=None, o_llh=None):
        """Transform to ENU, given ENU origin point o (in XYZ!).
        At least o_xyz or o_llh have to be provided!"""
        if o_llh is None: o_llh = o_xyz.llh()
        if o_xyz is None: o_xyz = o_llh.xyz(ellipsoid=self.ellipsoid)
        enu_to_xyz = enu_to_xyz_matrix(o_llh.lon, o_llh.lat)
        return ENU(*enu_to_xyz.T.dot(self()-o_xyz()),o_llh=o_llh,o_xyz=o_xyz)


class ENU:
    """ENU cartesian coordinate system.

    East-North-Up (ENU) coordinate system: cartesian similar to XYZ, just
    translated to origin point O, and rotated to align with the ENU axes.

    Parameters
    ----------
    e : float
    n : float
    u : float
    o_llh : LLH
        Origin given in LLH.
    o_xyz : XYZ
        Origin given in XYZ.
    """
    def __init__(self, e, n, u, o_llh=None, o_xyz=None, ellipsoid=WGS84):
        """At least one of the origin points, o_llh and o_xyz,
        have to be provided."""
        self.e = e
        self.n = n
        self.u = u
        if o_llh is None: o_llh = o_xyz.llh()
        if o_xyz is None: o_xyz = o_llh.xyz(ellipsoid)
        self.o_llh = o_llh
        self.o_xyz = o_xyz
    def xyz(self):
        enu_to_xyz = enu_to_xyz_matrix(self.o_llh.lon, self.o_llh.lat)
        return XYZ(*enu_to_xyz.dot(self())+self.o_xyz,
                   ellipsoid=self.o_xyz.ellipsoid)
    def llh(self):
        return self.xyz().llh()


class SCH:
    """Radar-related spherical coordinate system.

    It is referenced to the Peg position, determining the s and c directions,
    and height values.

    Parameters
    ----------
    peg: Peg
        Peg object, determing the directions of s and c coordinates.
    s : float, array_like
        Along-track curved distance, at the ground, in m.
    c : float, array_like
        Cross-track curved distance, at the ground, in m. Positive is left of s.
    h : float, array_like
        Height above peg sphere (?), in m.

    """
    def __init__(self, peg, s=None, c=None, h=None):
        self.peg = peg
        self.s = s
        self.c = c
        self.h = h
    def __repr__(self):
        return("s: {} c: {} h: {}".format(*self()))
    def __call__(self):
        return array([self.s, self.c, self.h])
    def llh(self):
        """Transform to LLH coordinates."""
        return self.xyz().llh()
    def xyz(self):
        """Transform SCH point to XYZ ECEF point."""
        c_angle = self.c / self.peg.radius
        s_angle = self.s / self.peg.radius
        r = self.peg.radius + self.h
        # from spherical to cartesian
        xyz_local = array ([r * cos (c_angle) * cos (s_angle),
                            r * cos (c_angle) * sin (s_angle),
                            r * sin (c_angle)])
        # from local xyz to ECEF xyz
        xyz = self.peg.rotation_matrix.dot(xyz_local) + self.peg.translation_vector
        return XYZ(xyz[0], xyz[1], xyz[2], self.peg.ellipsoid)


class LookVectorSCH(SCH):
    """Geometry of a look vector given in SCH coordinates.

    Meant only as reference. This look vector is given from platform to target:
    lv = sch_target - sch_platform

    Note: by default, the full 3d, non-normalized, vector is considered. Either
    project to incidence plane individually or use appropriate methods.

    Parameterizing the look vector in SCH coordinates:
             [ sin(inc_l) sin(az) ]     [ S ]
    \hat\l = [ sin(inc_l) cos(az) ]  =  [ C ]
             [    -cos(inc_l)     ]     [ H ]
    """
    def __init__(self, sch):
        SCH.__init__(self, sch.peg, sch.s, sch.c, sch.h)
    def range(self):
        if "r" not in self.__dict__:
            self.r = np.linalg.norm(self())
        return self.r
    def incidence_plane_look_angle(self):
        return np.arccos(-self.h/r)


def enu_to_xyz_matrix(lon, lat):
    """ENU to XYZ rotation matrix.

    Also the rotation matrix from sch_hat to xyz_prime (with lon=S, lat=C,
    both S and C in radians, i.e. s/r, and c/r).

    """
    slon, clon = sin(lon), cos(lon)
    slat, clat = sin(lat), cos(lat)
    enu_to_xyz = array([[-slon, -slat * clon, clat * clon],
                        [ clon, -slat * slon, clat * slon],
                        [ 0,     clat,        slat       ]])
    return enu_to_xyz
    
    
def sch2enu(s, c, h, peglat, peglon, peghdg):
    """Quick conversion from SCH to ENU coordinates.
    
    Convert an array of points in the SCH coordinate system to ENU
    (East-North-Up) coordinates, given the SCH peg location and heading.
    
    Arguments:
        s: Array containing S values, in meters.
        c: Array containing C values, in meters.
        h: Array containing H values, in meters.
        peglat: Latitude of the SCH peg, in radians.
        peglon: Longitude of the SCH peg, in radians.
        peghdg: Heading of the SCH peg, in radians.
        
    Returns:
        enu: Array containing ENU values, in meters.  Has dimensions
        (s.shape,3) if s, c, and h are arrays.  If s, c, and h are
        scalar, enu has dimensions of (3).
    
    """
    s = np.array(s)
    c = np.array(c)
    h = np.array(h)
    
    # Create origin peg object:
    peg = np.array([peglon, peglat, peghdg])
    peg = Peg(*peg)
    
    # Make images of the S and C angles (s/peg.radius and c/peg.radius)
    s_angle = s / peg.radius
    c_angle = c / peg.radius
    r = peg.radius + h
    xl = r * np.cos(c_angle) * np.cos(s_angle)
    yl = r * np.cos(c_angle) * np.sin(s_angle)
    zl = r * np.sin(c_angle)

    # From local XYZ to ECEF XYZ:    
    x = (peg.rotation_matrix[0,0]*xl + peg.rotation_matrix[0,1]*yl + peg.rotation_matrix[0,2]*zl) + peg.translation_vector[0]
    y = (peg.rotation_matrix[1,0]*xl + peg.rotation_matrix[1,1]*yl + peg.rotation_matrix[1,2]*zl) + peg.translation_vector[1]
    z = (peg.rotation_matrix[2,0]*xl + peg.rotation_matrix[2,1]*yl + peg.rotation_matrix[2,2]*zl) + peg.translation_vector[2]
    del xl,yl,zl
    
    # From ECEF XYZ to ENU:
    originllh = LLH(peg()[0],peg()[1],0)
    originxyz = originllh.xyz(ellipsoid=peg.ellipsoid)
    
    x -= originxyz()[0]
    y -= originxyz()[1]
    z -= originxyz()[2]    
    
    enumatrix = enu_to_xyz_matrix(peg()[0], peg()[1])
    enumatrix = enumatrix.T
       
    e = enumatrix[0,0]*x + enumatrix[0,1]*y + enumatrix[0,2]*z
    n = enumatrix[1,0]*x + enumatrix[1,1]*y + enumatrix[1,2]*z
    u = enumatrix[2,0]*x + enumatrix[2,1]*y + enumatrix[2,2]*z
    
    if len(e.shape) > 0:       
        enu = np.zeros((np.append(3,s.shape)),dtype='float32')
        enu[0] = e
        enu[1] = n
        enu[2] = u
        enu = np.moveaxis(enu,0,-1)
    else:
        enu = np.zeros((3),dtype='float32')
        enu[0] = e
        enu[1] = n
        enu[2] = u

    return enu
    
    
def llh2enu(lon, lat, h, peglat, peglon, peghdg):
    """Quick conversion from LLH to ENU coordinates.
    
    Convert an array of points in the LLH coordinate system to ENU
    (East-North-Up) coordinates, given the SCH peg location and heading.
    
    Arguments:
        lon: Array containing longitude values, in radians.
        lat: Array containing latitude values, in radians.
        h: Array containing height values, in meters.
        peglat: Latitude of the SCH peg, in radians.
        peglon: Longitude of the SCH peg, in radians.
        peghdg: Heading of the SCH peg, in radians.
        
    Returns:
        enu: Array containing ENU values, in meters.  Has dimensions
        (lon.shape,3) if lon, lat, and h are arrays.  If lon, lat, and h are
        scalar, enu has dimensions of (3).
    
    """
    lon = np.array(lon)
    lat = np.array(lat)
    h = np.array(h)
    
    # Create origin peg object:
    peg = np.array([peglon, peglat, peghdg])
    peg = Peg(*peg)
    
    # Transform LLH to ECEF XYZ coordinates.
    r = peg.ellipsoid.radius_east(lat)
    x = (r + h) * np.cos(lat) * np.cos(lon)
    y = (r + h) * np.cos(lat) * np.sin(lon)
    z = (r * (1.0 - peg.ellipsoid.e2) + h) * np.sin(lat)
    
    # From ECEF XYZ to ENU:
    originllh = LLH(peg()[0],peg()[1],0)
    originxyz = originllh.xyz(ellipsoid=peg.ellipsoid)
    
    x -= originxyz()[0]
    y -= originxyz()[1]
    z -= originxyz()[2]    
    
    enumatrix = enu_to_xyz_matrix(peg()[0], peg()[1])
    enumatrix = enumatrix.T
       
    e = enumatrix[0,0]*x + enumatrix[0,1]*y + enumatrix[0,2]*z
    n = enumatrix[1,0]*x + enumatrix[1,1]*y + enumatrix[1,2]*z
    u = enumatrix[2,0]*x + enumatrix[2,1]*y + enumatrix[2,2]*z
    
    if len(e.shape) > 0:
        enu = np.zeros((np.append(3,lon.shape)),dtype='float32')
        enu[0] = e
        enu[1] = n
        enu[2] = u
        enu = np.moveaxis(enu,0,-1)
    else:
        enu = np.zeros((3),dtype='float32')
        enu[0] = e
        enu[1] = n
        enu[2] = u

    return enu