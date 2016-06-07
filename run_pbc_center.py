from gromacs_utils import pbc_center, write_pdbs
import glob
import numpy as np


''' This is a short script for writing out pdbs of aligned trajectories. The pdbs are
    primarily for visualization. It is a simple way to check whether the protein has
    moved through the periodic box (and screwed up gmindist calculations).'''

pathname = '/home/jmh5sf/ngon/HPC_runs/cg/cg_540mus/'

xtcs = np.sort(glob.glob(pathname=pathname+'total_*.xtc'))
tprs = np.sort(glob.glob(pathname=pathname+'prod_*.tpr'))
ndx = pathname+'entropy.ndx'
orig_xtcs = []
n_files = 0
for xtc in xtcs:
    if ("whole" not in xtc) and ("cntr" not in xtc):
        orig_xtcs.append(xtc)
        n_files += 1

pbc_center(xtcs=orig_xtcs, tprs=tprs, ndx=ndx)
new_xtcs = [orig_xtc[0:-4]+'_cntr.xtc' for orig_xtc in orig_xtcs]
pdbs = [orig_xtc[0:-4] + '.pdb' for orig_xtc in orig_xtcs]
write_pdbs(xtcs=new_xtcs, tprs=tprs, ndx=ndx, pdbs=pdbs, rewrite=True)
