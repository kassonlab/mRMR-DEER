from gromacs_utils import pbc_center, write_pdbs, strip_filenames
import glob
import numpy as np


''' This is a short script for writing out pdbs of aligned trajectories. The pdbs are
    primarily for visualization. It is a simple way to check whether the protein has
    moved through the periodic box (and screwed up gmindist calculations).'''

pathname = '/home/jmh5sf/dimer_ngon/ngon/HPC_runs/cg/cg_540mus/'

xtcs = np.sort(glob.glob(pathname=pathname+'total_*.xtc'))
tprs = np.sort(glob.glob(pathname=pathname+'prod_*.tpr'))
ndx = pathname+'entropy.ndx'
orig_filenames = strip_filenames(xtcs)
orig_xtcs = [orig_file+'.xtc' for orig_file in orig_filenames]
n_files = len(orig_xtcs)
print tprs
pbc_center(xtcs=orig_xtcs, tprs=tprs, ndx=ndx)
new_xtcs = [orig_file + '_cntr.xtc' for orig_file in orig_filenames]
pdbs = [orig_file + '.pdb' for orig_file in orig_filenames]
write_pdbs(xtcs=new_xtcs, tprs=tprs, ndx=ndx, pdbs=pdbs, rewrite=True)
