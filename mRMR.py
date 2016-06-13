import numpy as np
from gromacs_utils import make_ndx, read_ndx, warn_file_exists, colors
import os
import cPickle
import time


def entropy(arr):
    '''
    Calculates the entropy of a distribution:
        S = - SUM_i (p_i * ln p_i)
    This is the true entropy if arr is a pdf, but notice that you will have to divide by an appropriate normalization if
    p_i's are counts rather than probabilities.
    :param arr: a probability distribution
    :return: entropy of arr
    '''
    return np.sum([-p * np.log(p) if p else 0 for p in arr])


def h_fast(x):
    '''
    Calculates the column entropies of the matrix x. This assumes that the columns of x have been binned but NOT
    histogrammed. That is, it will perform the following calculation:
        S = -SUM_i [n_i/N ln(n_i/N)]
    This is actually implemented as:
        S = ln(N) - 1/N * SUM_i[n_i ln n_i]
    If X is mxn dimensional, then this will return a list of length n.
    :param x: An mxn matrix of observations. m is the number of observations, and n is the number of observables. In the
    case of mRMR, m is the number of frames and n is the number of pairs of residues (x is essentially a processed xvg
    from gmx mindist).
    :return: column entropy (n dimensional array)
    '''
    n_samples, _ = x.shape
    bin_range = [np.min(x), np.max(x)]
    n_bins = bin_range[1] - bin_range[0]

    # Bincount along columns
    hist = np.apply_along_axis(lambda y: np.histogram(y, bins=n_bins, range=bin_range)[0], axis=0, arr=x)
    # Compute: -SUM_i[n_i ln n_i] along cols
    count_entropy = np.apply_along_axis(entropy, axis=0, arr=hist)
    # Compute: ln(N) - 1/N * SUM_i[n_i ln n_i]
    return np.add(np.divide(count_entropy, n_samples), np.log(n_samples))


def mi_fast(x, y):
    '''
    A clever algorithm for calculating the mutual information of the arrays x and y. Pulled from Kasson git: numpy_mi.py
    :param x: an array that has been binned but not yet been histogrammed.
    :param y: an array that has been binned but not yet been histogrammed.
    :return: mutual information of x and y.
    '''
    n_samples = len(x)
    bin_range = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    n_bins = bin_range[1] - bin_range[0]

    xy_c = np.histogram2d(x, y, bins=n_bins, range=[bin_range, bin_range])[0]
    x_c = np.sum(xy_c, 1)
    y_c = np.sum(xy_c, 0)
    Hxy = entropy(np.ravel(xy_c)) / n_samples + np.log(n_samples)
    Hx = entropy(np.ravel(x_c)) / n_samples + np.log(n_samples)
    Hy = entropy(np.ravel(y_c)) / n_samples + np.log(n_samples)
    return Hx + Hy - Hxy


def get_permitted_pairs(mat, dist_range=None, resi_select=None, resi_tpr=None, AA=True, bin_size=0):
    '''
    This function allows you to restrict the pairs you wish to consider in the mRMR algorithm based on their average
    distances and/or residue name. DEER measurements are reliable for distances only in the range of about 20 - 50 nm.
    Therefore, we wish to only consider pairs with an average distance in simulation of 20 - 50 nm. Similarly, we may
    not wish to mutate certain residues that are integral to secondary structure (or lack of secondary structure) e.g.,
    PRO. Therefore, we may eliminate all PRO residues from the distance calculation as well.
    :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i.
    :param dist_range: the range of allowed distances as a list. I.e., [2.0, 5.0] for the DEER case described above.
    :param resi_select: a list of residue name selections, e.g. ['PRO', 'GLY', ...], that you wish to eliminate in the
    mRMR calculation. Use PDB residue naming scheme.
    :param resi_tpr: a single tpr so that the residue name can be mapped to atom index and therefore to pair number.
    :param bin_size: This is clunky. Although the matrix has already been binned, we need the bin_size to convert from
    the bin number back to absolute distance in nm.
    :return: a list of permitted pair numbers.
    '''

    _, num_pairs = mat.shape

    permitted_pairs = [i for i in range(num_pairs)]

    # EXCLUDE BASED ON DISTANCE
    if dist_range is not None:
        bin_range = np.divide(dist_range, bin_size)
        print "Choosing pairs with distances between", dist_range[0], "and", dist_range[1], "nm"
        print "That corresponds to bin numbers between", bin_range[0], "and", bin_range[1]
        bin_mean = np.mean(mat, axis=0)
        for i in range(num_pairs):
            if not (bin_range[0] < bin_mean[i] < bin_range[1]):
                permitted_pairs.remove(i)

    # EXCLUDE BASED ON RESIDUE NAME
    if resi_select is not None:

        # Generate two index files:
        # 1. CA/BB that are PRO
        # 2. All CA/BB
        full_selection = "(resname " + " or resname ".join(resi_select) + ")"
        if AA:
            selection = 'mol 1 and name CA and ' + full_selection + \
                        '; mol 2 and name CA and ' + full_selection
        else:
            selection = 'mol 1 and name BB and ' + full_selection + \
                        '; mol 2 and name BB and ' + full_selection

        print "Eliminating pairs with selection:", resi_select

        make_ndx(tpr_filename=resi_tpr, ndx_filename='selected_restrict_resis.ndx', selection=selection)
        make_ndx(tpr_filename=resi_tpr, ndx_filename='all_resis.ndx', AA=AA)
        restrict_dict = read_ndx('selected_restrict_resis.ndx')
        all_dict = read_ndx('all_resis.ndx')
        mol1all, mol2all, _, _ = all_dict.keys()
        mol1restrict, mol2restrict = restrict_dict.keys()

        pair_count = 0

        for resi1 in all_dict[mol1all]:
            for resi2 in all_dict[mol2all]:
                if ((resi1 in restrict_dict[mol1restrict]) or
                        (resi2 in restrict_dict[mol2restrict])) and \
                                (pair_count in permitted_pairs):
                    permitted_pairs.remove(pair_count)
                pair_count += 1

    return permitted_pairs


def mRMR(mat, iters, entropy_filename, weights=[1, 0], dist_restrict=None, resi_restrict=None,
         resi_tpr=None, AA=True, bin_size=0, rewrite=False):

    '''
    This method is an implementation of the mRMR algorithm described in http://dx.doi.org/10.1142/S0219720005001004
    (Here, however, we use a weighted sum of both the MID and MIQ criteria, not just one or the other).

    :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i. The matrix MUST be binned.
    Unfortunately, because of memory concerns, the xvg must be read in as bin numbers (integer) instead of
    raw distances (floats).
    :param iters: Number of mRMR pairs you wish to calculate.
    :param entropy_filename: a file for storing all distance entropies. This is a binary cPickle file (this is done so
    that if you have already calculated the entropies, the entropies can be read in quickly instead of recalculated).
    :param weights: weights for MID and MIQ criteria. This should be a list: [MID_weight, MIQ_weight]
    :param dist_restrict: the range of allowed distances as a list. I.e., [2.0, 5.0] for the usual DEER case.
    :param resi_restrict: a list of residue name selections, e.g. ['PRO', 'GLY', ...], that you wish to eliminate in the
    mRMR calculation. Use PDB residue naming scheme.
    :param resi_tpr:  a single tpr so that the residue name can be mapped to atom index and therefore to pair number.
    :param bin_size: This is a little clunky. The distance matrix mat must be pre-binned, but we need to know the bin
    size that was used. That way we can map bin number to absolute distance with get_permitted_pairs()
    :param rewrite: boolean to indicate whether you want to rewrite the entropy file.
    :return: a list of pair numbers, ordered from highest to lowest mRMR.
    '''

    _, n_pairs = mat.shape  # Get the number of pairs in your matrix

    permitted_resis = get_permitted_pairs(mat, dist_range=dist_restrict, resi_select=resi_restrict, resi_tpr=resi_tpr,
                                          AA=AA, bin_size=bin_size)

    print "Saving permitted pairs to permitted_pairs.txt in this directory..."
    np.savetxt('permitted_pairs.txt', permitted_resis)

    # If no entropy has been provided, entropy must be calculated for permitted residues.
    print colors.HEADER + "\tENTROPY CALCULATION" + colors.ENDC
    if os.path.exists(entropy_filename) and not rewrite:
        warn_file_exists(entropy_filename)
        H = cPickle.load(open(entropy_filename))
    else:
        print "No entropy file was found or you have chosen to rewrite. Calculating entropies..."
        H = h_fast(mat)
        print "Dumping entopies to", entropy_filename
        cPickle.dump(H, open(entropy_filename, 'w'))

    # Initialize a set omega. At each iteration you will pull a pair from this set
    # based on the mRMR criterion of your choice (subtraction or division). The pair
    # will be placed into the set s. s is the final set of highest mRMR pairs.

    omega = [i for i in permitted_resis]  # At first, all permitted pairs are present in omega
    s = []  # s starts as an empty set
    mi_mat = np.zeros(shape=(n_pairs, n_pairs))  # To-do: make mi_mat only as big as num of permitted resis

    print colors.HEADER + "\n\tBEGINNING mRMR CALCULATION" + colors.ENDC
    for m in range(iters):  # Number of times you want to perform mRMR
        print "m =", m, "calculation"
        start_time = time.time()
        if m == 0:  # At the first step, choose the highest entropy pair.
            temp_H = [H[i] for i in permitted_resis]
            s.append(omega.pop(np.argmax(temp_H)))
            # Now you have lost the nice property that omega[i] = i. You must be
            # careful in all following calculations to ensure you pop the right
            # element of omega.

        # Must calculate the mutual information between pairs of pairs.
        # MI = H(Y_i) + H(Y_j) - H(Y_i, Y_j)
        # Then the final mRMR score is H(Y_test) -/ mean of MI(Y_test, Y_s)
        else:
            # high_idx are the high mRMR pairs currently in the set s. You
            # will evaluate the mutual information between each of these
            # pairs and a test pair. The test pair is test_idx

            mRMR_scores = []

            for test_idx in omega:  # Loop through test pairs
                pair_pair_mi = 0
                for high_idx in s:  # Loop through mRMR pairs

                    # First, check to see if you need to store joint entropy - you don't want
                    # to recalculate if you can use symmetry properties and previously stored
                    # values.
                    if (mi_mat[test_idx, high_idx] == 0) and (mi_mat[high_idx, test_idx] == 0):
                        mi_mat[test_idx, high_idx] = mi_fast(mat[:, test_idx], mat[:, high_idx])
                        mi_mat[high_idx, test_idx] = mi_mat[test_idx, high_idx]

                    pair_pair_mi += mi_mat[test_idx, high_idx]
                mRMR_scores.append(weights[0]*(H[test_idx] - pair_pair_mi/m) +
                                   weights[1]*H[test_idx]/pair_pair_mi/m)
            print "Minimum mRMR score:", np.min(mRMR_scores)
            print "Maximum mRMR score:", np.max(mRMR_scores)
            omega_idx = np.argmax(mRMR_scores)  # Find the test pair with the highest mRMR value
            s.append(omega.pop(omega_idx))  # Pop that test pair from omega, place in s
        end_time = time.time()
        print "Time elapsed for m =", m, "calculation:", end_time-start_time
    return np.array(s)



