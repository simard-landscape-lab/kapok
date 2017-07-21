[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.167040.svg)](https://doi.org/10.5281/zenodo.167040)

# Kapok: An Open Source Python Library for PolInSAR Forest Height Estimation Using UAVSAR Data

Kapok is a Python library created for the purposes of estimating forest height and structure using data collected by NASA's Uninhabited Aerial Vehicle Synthetic Aperture Radar (UAVSAR) instrument.  The library contains implementations of basic algorithms for processing of polarimetric SAR interferometry (PolInSAR) data, and allows easy import of UAVSAR SLC (single-look complex) stacks (UAVSAR data from multiple repeat-pass flights).

Software primarily designed and written by Michael Denbina.  Brian Hawkins contributed a geocoding function using the pyresample library.  Maxim Neumann contributed a number of ideas regarding the overall structure and data organization of Kapok, as well as library functions for coordinate transformations and indexing of baselines.  See individual source code files for more detailed author information.

If you use this software in a published work, please cite it using the following DOI: https://doi.org/10.5281/zenodo.167040

For reference, also see the following journal articles for PolInSAR forest height estimation results using this software:

M. Simard and M. Denbina, "An Assessment of Temporal Decorrelation Compensation Methods for Forest Canopy Height Estimation Using Airborne L-Band Same-Day Repeat-Pass Polarimetric SAR Interferometry," IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, Submitted, 2017.

M. Denbina and M. Simard, "The Effects of Topography on Forest Height and Structure Estimation from PolInSAR," IEEE Transactions on Geoscience and Remote Sensing, Submitted, 2017.

This library is dependent on the following open source software libraries:

* Numerical Python (NumPy)
* Scientific Python (SciPy)
* HDF5 For Python (h5py)
* matplotlib
* Cython
* Geospatial Data Abstraction Library (GDAL)
* pyresample

See docs/manual.pdf for a user's manual and basic tutorial.  The scripts/ folder contains a number of example scripts demonstrating how to use the software.  The docs/ folder also contains installation guides for Mac OSX and Windows.

Copyright 2016 California Institute of Technology.  All rights reserved.  United States Government Sponsorship acknowledged.

This software is released under the GNU General Public License.  For details, see the file LICENSE.txt included with this program, or visit http://www.gnu.org/licenses/.  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.