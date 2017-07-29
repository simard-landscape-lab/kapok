# -*- coding: utf-8 -*-
"""Kapok Setup File

    See README.md for a description of this software.  See docs/install.pdf
    for installation instructions.

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
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file.       
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = long_description.replace("\r","")
except ImportError:
    print("Pandoc not found. Long_description conversion failure.")
    with open(path.join(here, 'README.md'), encoding="utf-8") as f:
        long_description = f.read()


setup(
    name='Kapok',
    version='0.2.0',

    description='A Python library for PolInSAR forest height estimation using UAVSAR data.',
    long_description=long_description,
    url='https://github.com/mdenbina/kapok',

    author='Michael Denbina',
    author_email='michael.w.denbina@jpl.nasa.gov',

    license='GPLv3+',

    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='radar remotesensing ecology forestry biomass polinsar polarimetry',

    packages=find_packages(exclude=['docs', 'tests', 'scripts']),
    
    # Required Packages
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py', 'cython', 'gdal', 'pyresample'],
    
    # Cython Modules
    ext_modules = cythonize(['kapok/cohoptc.pyx', 'kapok/rvogc.pyx', 'kapok/uavsarc.pyx']),
    include_dirs=[np.get_include()],
)