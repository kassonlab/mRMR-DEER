import os
import glob
import numpy as np
import sys
from gromacs_utils import strip_filenames

pathname = '/home/jmh5sf/dimer_ngon/ngon/HPC_runs/cg/cg_540mus/'
iters = 3
binsize = 0.2
restrict = [2.0, 5.0]  # Do not include pairs with distances < 20 A or > 50 A
weights = [1.0, 0.0]  # Exclusively MID (subtraction algorithm).
rr = 'PRO'

if sys.argv[-1] == 'large':

    xtcs = np.sort(glob.glob(pathname=pathname+'total_*.xtc'))
    tprs = np.sort(glob.glob(pathname=pathname+'prod_*.tpr'))

    orig_xtcs = strip_filenames(xtcs)
    n_files = len(orig_xtcs)

    # Get the right numbering for output xvgs.
    traj_nums = [orig_xtc[-2:] for orig_xtc in orig_xtcs]
    xvgs = [pathname+'dist_'+traj_nums[i]+'.xvg' for i in range(n_files)]

    ndx = 'large_test/entropy.ndx'
    entropy_mat_name = 'large_test/entropy.p'
    hmRMR_name = 'large_test/highmRMR.txt'

elif sys.argv[-1] == 'small':

    pathname = '/home/jmh5sf/dimer_ngon/ngon/HPC_runs/cg/cg_540mus/'

    xtcs = [pathname + 'total_cg_10.xtc', pathname + 'total_cg_11.xtc']
    tprs = [pathname + 'prod_center_10.tpr', pathname + 'prod_center_11.tpr']

    orig_xtcs = strip_filenames(xtcs)
    n_files = len(orig_xtcs)

    # Get the right numbering for output xvgs.
    traj_nums = [orig_xtc[-2:] for orig_xtc in orig_xtcs]
    xvgs = [pathname + 'dist_' + traj_nums[i] + '.xvg' for i in range(n_files)]

    ndx = 'small_test/entropy.ndx'
    entropy_mat_name = 'small_test/entropy.p'
    hmRMR_name = 'small_test/highmRMR.txt'

else:
    raise ValueError('Must specify large or small test run.')


os.system('python run_mRMR.py -f %s -s %s -n %s -od %s -iters %i -binsize %f -oe %s '
          '-om %s -rd %f %f -rr %s -w %f %f -cg' %
          (" ".join(xtcs), " ".join(tprs), ndx, " ".join(xvgs), iters, binsize,
           entropy_mat_name, hmRMR_name, restrict[0], restrict[1], 'PRO', weights[0],
           weights[1]))
