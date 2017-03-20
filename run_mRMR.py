from gromacs_utils import *
from mRMR import *
import argparse
import numpy as np


def convert_to_pairs(tri_resis, num_residues):
    row_idx, col_idx = np.triu_indices(num_residues, k=1)
    pairs = []
    for tri_resi in tri_resis:
        pairs.append([row_idx[tri_resi], col_idx[tri_resi]])
    return np.add(pairs, 1)

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description='Runs mRMR algorithm on csv of distance data.')
    parser.add_argument('-f', nargs='*', help='Configuration file')
    parser.add_argument('-i', type=int, help='Number of mRMR pairs to calculate')
    parser.add_argument('-n', type=int, help='Number of residues in protein')
    args = parser.parse_args()
    iters = args.i

    print colors.BOLD + '\n\tmRMR calculation with ' + str(iters) + ' iterations' + colors.ENDC
    mRMR_config = MRMRConfig(args.f, rewrite=False)
    if mRMR_config.chains == 1:
        num_residues = args.n
    else:
        raise ValueError("No support for multiple chains yet")

    resis = mRMR_config.mRMR(iters, store_in_memory=False)
    print 'MID Weight:', mRMR_config.weights[0]
    print 'MIQ Weight:', mRMR_config.weights[1]
    print "Highest mRMR pair indices", resis

    # Converts flattened indices to matrix indices.
    pairs = convert_to_pairs(resis, num_residues)
    print "Converted to residue numbers (Molecule 1, Molecule 2):", pairs
    #print colors.BOLD + "Please remember that your residues are now zero-indexed!" + colors.ENDC
    final_filename = "%s-MID%0.2f-MIQ%0.2f.txt" % (mRMR_config.mRMR_filename[:-4], mRMR_config.weights[0],
                                                   mRMR_config.weights[1])
    np.savetxt(final_filename, pairs, fmt="%i")


