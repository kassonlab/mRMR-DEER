import os
import glob
import numpy as np

# This is a working test run of the mRMR algorithm using 540 microsecond CG trajectory

pathname = '/home/jmh5sf/dimer_ngon/ngon/HPC_runs/cg/cg_540mus/'

xtcs = np.sort(glob.glob(pathname=pathname+'total_*.xtc'))
tprs = np.sort(glob.glob(pathname=pathname+'prod_*.tpr'))


# You need to exclude globbed files that include "whole" and "cntr", otherwise you will
# end up generating files that go like "_cntr_cntr.pdb", etc.

orig_xtcs = []
n_files = 0
for xtc in xtcs:
    if ("whole" not in xtc) and ("cntr" not in xtc):
        orig_xtcs.append(xtc)
        n_files += 1

# Get the right numbering for output xvgs.
traj_nums = [orig_xtc[-6:-4] for orig_xtc in orig_xtcs]
xvgs = [pathname+'dist_'+traj_nums[i]+'.xvg' for i in range(n_files)]


ndx = pathname+'entropy.ndx'
iters = 3
binsize = 0.2
entropy_mat_name = pathname + 'entropy.txt'
hmRMR_name = pathname + 'highmRMR.txt'
restrict = [2.0, 5.0]  # Do not include pairs with distances < 20 A or > 50 A
weights = [1.0, 0.0]  # Exclusively MID (subtraction algorithm).


os.system('python run_mRMR.py -f %s -s %s -n %s -od %s -iters %i -binsize %f -oe %s '
          '-om %s -rd %f %f -rr %s -w %f %f -cg' %
          (" ".join(xtcs), " ".join(tprs), ndx, " ".join(xvgs), iters, binsize,
           entropy_mat_name, hmRMR_name, restrict[0], restrict[1], 'PRO', weights[0],
           weights[1]))
