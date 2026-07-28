# -*- coding: utf-8 -*-
"""Library containing small utility/convenience functions called by the
       main Kapok modules.

"""
from .coords import *
from .gdal import *
from .interp import *
from .matrix import *
from .mb import *
from .slope import *
from .win import *

# import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()})
# from .matrixc import *