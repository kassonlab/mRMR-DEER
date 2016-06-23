import os
from gromacs_utils import strip_filenames
# This is a small working test run of the mRMR algorithm using a subset of 540 microsecond CG trajectory

pathname = '/home/jmh5sf/dimer_ngon/ngon/HPC_runs/cg/cg_540mus/'

xtcs = [pathname + 'total_cg_10.xtc', pathname + 'total_cg_11.xtc']
tprs = [pathname + 'prod_center_10.tpr', pathname + 'prod_center_11.tpr']

orig_xtcs = strip_filenames(xtcs)
n_files = len(orig_xtcs)

# Get the right numbering for output xvgs.
traj_nums = [orig_xtc[-2:] for orig_xtc in orig_xtcs]
xvgs = [pathname+'dist_'+traj_nums[i]+'.xvg' for i in range(n_files)]


ndx = pathname+'entropy_10_11.ndx'
iters = 3
binsize = 0.2
entropy_mat_name = pathname + 'entropy_10_11.p'
hmRMR_name = pathname + 'highmRMR.txt'
restrict = [2.0, 5.0]  # Do not include pairs with distances < 20 A or > 50 A
weights = [1.0, 0.0]  # Exclusively MID (subtraction algorithm).
rr = 'PRO'  # Ignore Prolines


os.system('python run_mRMR.py -f %s -s %s -n %s -od %s -iters %i -binsize %f -oe %s '
          '-om %s -rd %f %f -rr %s -w %f %f -cg -control' %
          (" ".join(xtcs), " ".join(tprs), ndx, " ".join(xvgs), iters, binsize,
           entropy_mat_name, hmRMR_name, restrict[0], restrict[1], rr, weights[0],
           weights[1]))
