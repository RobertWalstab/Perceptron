import numpy as np
import building_blocks
import genplotting
import ownconstants as ocst

# %matplotlib  # Magic command, execute if supported
d = {'f': 1.93e14, 'tau': ocst.tau, 'psi0': 0.0}
val_fkt = genplotting.val_splitter(building_blocks.i_tr, **d)

a = np.arange(0, 2.2, 0.2)

ax1 = None
for v in a:
    ax1 = genplotting.plotting_fkt(*(val_fkt(v)), ax1)

