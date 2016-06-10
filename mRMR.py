import math
import numpy as np
from gromacs_utils import make_ndx, read_ndx


def entropy(X):
    '''
    Calculates the entropy of a series of observations X. If using this function with
    mRMR algorithm, X must be a series of BINNED distances.
    :param X: set of observations
    :return: entropy of X
    '''
    s = 0
    probs = [(X == x).mean() for x in set(X)]
    for p in probs:
        s += -p * math.log(p)
    return s


def joint_entropy(X, Y):
    '''
    Calculates the joint entropy of two distributions.
    :param X: A series of observations. If using this function with mRMR algorithm,
    X must be a series of BINNED distances.
    :param Y: Same as X.
    :return: joint entropy of X and Y.
    '''

    probs = []
    joint_H = 0
    for x in set(X):
        for y in set(Y):
            probs.append(np.logical_and(X == x, Y == y).mean())
    for p in probs:
        if p != 0:
            joint_H += - p * math.log(p)
    return joint_H


def get_permitted_pairs(mat, dist_range=None, resi_select=None, resi_tpr=None, AA=True, bin_size=0):
    '''
    This function allows you to restrict the pairs you wish to consider in the mRMR
    algorithm based on their average distances. DEER measurements are reliable for
    distances only in the range of about 20 - 50 nm. Therefore, we only wish to consider
    pairs with an average distance in simulation of 20 - 50 nm.
    :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i.
    :param dist_range: the range of allowed distances as a list. I.e., [2.0, 5.0] for
    the DEER case described above.
    :param bin_size: This is a little clunky. The distance matrix mat must be pre-binned,
    but we need to know the bin size that was used. That way we can map bin number to
    absolute distance.
    :return: a list of permitted pair numbers.
    '''

    _, num_pairs = mat.shape

    permitted_pairs = [i for i in range(num_pairs)]

    if dist_range is not None:
        print "Choosing pairs with distances between", dist_range[0], "and", dist_range[1],"nm"
        dists = np.mean(mat, axis=0)
        if bin_size != 0:
             dists = np.multiply(dists, bin_size)
        for i in range(num_pairs):
            if not (dist_range[0] < dists[i] < dist_range[1]):
                permitted_pairs.remove(i)

    if resi_select is not None:
        full_selection = "(resname " + " or resname ".join(resi_select) + ")"
        if AA:
            selection = 'mol 1 and name CA and ' + full_selection + \
                        '; mol 2 and name CA and ' + full_selection
        else:
            selection = 'mol 1 and name BB and ' + full_selection + \
                        '; mol 2 and name BB and ' + full_selection

        print "Eliminating pairs with selection:", resi_select

        make_ndx(tpr_filename=resi_tpr, ndx_filename='selected_restrict_resis.ndx',
                 selection=selection)
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


def mRMR(mat, iters, H=None, weights=[1, 0], dist_restrict=None, resi_restrict=None,
         resi_tpr=None, AA=True, bin_size=0):

    '''
    This method is an implementation of the mRMR algorithm described in
    http://dx.doi.org/10.1142/S0219720005001004
    Here, however, it is possible to use a weighted sum of both the MID and MIQ criteria.

    :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i. Distances
    should be binned.
    :param iters: Number of mRMR pairs you wish to calculate.
    :param H: you may provide an entropy matrix if you have already calculated one.
    :param weights: weights for MID and MIQ criteria. This should be a list:
    [MID_weight, MIQ_weight]
    :param dist_restrict: the range of allowed distances as a list. I.e., [2.0, 5.0] for
    the usual DEER case.
    :param resi_restrict:
    :param bin_size: This is a little clunky. The distance matrix mat must be pre-binned,
    but we need to know the bin size that was used. That way we can map bin number to
    absolute distance with restrict_distance()
    :return: a list of pair numbers, ordered from highest to lowest mRMR.
    '''

    _, n_pairs = mat.shape  # Get the number of pairs in your matrix

    permitted_resis = get_permitted_pairs(mat, dist_range=dist_restrict,
                                          resi_select=resi_restrict, resi_tpr=resi_tpr,
                                          AA=AA, bin_size=bin_size)

    np.savetxt('permitted_pairs.txt', permitted_resis)

    # If no entropy has been provided, entropy must be calculated for permitted residues.
    if H is None:
        H = [entropy(mat[:, i]) for i in range(n_pairs)]

    # Initialize a set omega. At each iteration you will pull a pair from this set
    # based on the mRMR criterion of your choice (subtraction or division). The pair
    # will be placed into the set s. s is the final set of highest mRMR pairs.

    omega = [i for i in permitted_resis]  # At first, all permitted pairs are present in omega
    s = []  # s starts as an empty set
    jointH = np.zeros(shape=(n_pairs, n_pairs))  # To-do: make jointH only as big as num of permitted resis

    for m in range(iters):  # Number of times you want to perform mRMR
        print "m =", m, "calculation"
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
                    if (jointH[test_idx, high_idx] == 0) and (jointH[high_idx, test_idx] == 0):
                        jointH[test_idx, high_idx] = joint_entropy(mat[:, test_idx], mat[:, high_idx])
                        jointH[high_idx, test_idx] = jointH[test_idx, high_idx]
                    elif jointH[test_idx, high_idx] == 0:
                        jointH[test_idx, high_idx] = jointH[high_idx, test_idx]

                    pair_pair_mi += H[test_idx] + H[high_idx] - jointH[test_idx, high_idx]

                mRMR_scores.append(weights[0]*(H[test_idx] - pair_pair_mi/m) +
                                   weights[1]*H[test_idx]/pair_pair_mi/m)

            omega_idx = np.argmax(mRMR_scores)  # Find the test pair with the highest mRMR value
            s.append(omega.pop(omega_idx))  # Pop that test pair from omega, place in s

    return np.array(s)



