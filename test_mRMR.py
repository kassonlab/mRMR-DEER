import os
import glob
import numpy as np

pathname = '/home/jmh5sf/ngon/HPC_runs/cg/cg_540mus/'

xtcs = np.sort(glob.glob(pathname=pathname+'total_*.xtc'))
tprs = np.sort(glob.glob(pathname=pathname+'prod_*.tpr'))
print xtcs
print tprs

xvgs = [pathname+'dist_'+str(i)+'.xvg' for i in range(len(xtcs))]
ndx = pathname+'entropy.ndx'
iters = 3
binsize = 0.2
entropy_mat_name = pathname+'entropy.txt'
hmRMR_name = pathname + 'highmRMR.txt'
restrict = [2.0, 5.0]
weights = [1.0, 0.0]


os.system('python run_mRMR.py -f %s -s %s -n %s -od %s -iters %i -binsize %f -oe %s -om %s '
          '-restrict %f %f -w %f %f -cg' %
          (" ".join(xtcs), " ".join(tprs), ndx, " ".join(xvgs), iters, binsize, entropy_mat_name, hmRMR_name,
           restrict[0], restrict[1], weights[0], weights[1]))
