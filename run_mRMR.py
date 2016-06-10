from gromacs_utils import *
from mRMR import *
import argparse


# Anal-retentive Jennifer color codes outputs.
class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Command line arguments
parser = argparse.ArgumentParser(description='Runs mRMR algorithm on a list of trajectories. '
                                             'When you provide lists of xtcs, tprs, and xvgs, ordering matters!')
parser.add_argument('-f', nargs='*', help='list of xtcs')
parser.add_argument('-s', nargs='*', help='list of tprs')
parser.add_argument('-n', help='ndx file for output')
parser.add_argument('-od', nargs='*', help='list of xvgs for output')
parser.add_argument('-oe', help='output txt file of entropy', default='entropy.txt')
parser.add_argument('-om', help='output txt file of high mRMR pairs', default='highmRMR.txt')
parser.add_argument('-iters', help='number of times to perform mRMR', type=int)
parser.add_argument('-binsize', help='bin size in nm', type=float, default=0.2)
parser.add_argument('-rd', nargs='*', type=float, help='lower and upper distance limits in nm',
                    default=[2.0, 5.0])
parser.add_argument('-rr', nargs='*', help='Residue names to ignore', default='PRO')
parser.add_argument('-w', nargs='*', type=float, help='weights for MID (subtraction) and MIQ (division) '
                                                      'in mRMR calculation, respectively',
                    default=[1.0, 0.0])
parser.add_argument('-cg', action='store_true', help='Use this flag for coarse-grained simulations')
args = parser.parse_args()

xtc_names = args.f
tpr_names = args.s
ndx_name = args.n
xvg_names = args.od
entropy_name = args.oe
mRMR_name = args.om
iters = args.iters
bin_size = args.binsize
dist_restrict = args.rd
resi_restrict = args.rr
weights = args.w
aa = not args.cg

# Align all the trajectories and put them in the center of the periodic box so that gmx mindist
# can calculate distances properly.

print colors.BOLD + '\n\tPreprocessing trajectories' + colors.ENDC
data, ndx_dict = pre_process(xtcs=xtc_names, tprs=tpr_names, ndx=ndx_name, xvgs=xvg_names,
                             bin_size=bin_size, AA=aa)
n_samples, n_pairs = data.shape

# Calculate the entropies of each distance pair. If an entropy file already exists, the calculation is
# skipped and the file is loaded into memory.

print colors.BOLD + '\n\tEntropy of each pair' + colors.ENDC

if os.path.exists(entropy_name):
    print "The file", entropy_name, "already exists: delete/move this file to rewrite"
    entropy_matrix = np.loadtxt(entropy_name)
else:
    entropy_matrix = [entropy(data[:, i]) for i in range(n_pairs)]
    np.savetxt(entropy_name, entropy_matrix, fmt='%g')
    print "Finished calculating the entropies."


print colors.BOLD + '\n\tmRMR calculation with ' + str(iters) + ' iterations' + colors.ENDC
raveled_resis = mRMR(mat=data, iters=iters, H=entropy_matrix, weights=[1, 0],
                     dist_restrict=dist_restrict, resi_restrict=resi_restrict,
                     resi_tpr=tpr_names[0], AA=aa, bin_size=bin_size)
print "Highest mRMR pair indices", raveled_resis


# Getting the length of the two proteins so that the flattened indices of raveled_resis can be turned
# into non-flattened indices. In other words, pair 5 -> (ceacam 1, opa 5).
if aa:
    mol1_len = len(ndx_dict['mol_1_and_name_CA'])
    mol2_len = len(ndx_dict['mol_2_and_name_CA'])
else:
    mol1_len = len(ndx_dict['mol_1_and_name_BB'])
    mol2_len = len(ndx_dict['mol_2_and_name_BB'])

print "Length of molecule 1:", mol1_len, "\nLength of molecule 2:", mol2_len

# Converts flattened indices to matrix indices.
resis = np.unravel_index(raveled_resis, (mol1_len, mol2_len))
print "Converted to residue numbers (Molecule 1, Molecule 2):", resis
print "Please remember that your residues are now zero-indexed!"
print "Saving to file", mRMR_name
np.savetxt(mRMR_name, resis, fmt='%g')


