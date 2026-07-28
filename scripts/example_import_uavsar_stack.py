import os
import os.path

import numpy as np
from scipy import ndimage
import seaborn
seaborn.set_style('white')
seaborn.set_context('poster',font_scale=0.7,rc={'lines.markeredgewidth': 0.5})
hveg_cmap = seaborn.cubehelix_palette(8, start=1.5, rot=0.3, dark=0, light=1, reverse=True, as_cmap=True)

import kapok

annfile = '/Users/mdenbina/mnt/stacks/rabi/1/rabifo_TM130_16010_000_160228_L090HH_02_BC.ann'
kapokfile = '/Users/mdenbina/Sites/rabi/kapok/rabi_BC_20_5_prockz.hdf5'
mlwin = [20,5]
kzcalc = False


# First, get the Kapok Scene object.  If the file already exists, load it.
# Otherwise, import from UAVSAR data.
if os.path.isfile(kapokfile):
    scene = kapok.Scene(kapokfile)
else:
    import kapok.uavsar
    scene = kapok.uavsar.load(annfile,kapokfile,mlwin=mlwin,num_blocks=50,kzcalc=kzcalc,overwrite=True)




# Coherence Optimization
scene.opt(saveall=True)

