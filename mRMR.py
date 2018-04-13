import numpy as np
from gromacs_utils import make_ndx, read_ndx, warn_file_exists, colors
import os
import cPickle
import time
from ConfigParser import RawConfigParser


def generate_bins(vector, incr):
    """
    Categorizes a vector based on a set increment
    :param vector: vector to categorize
    :param incr: spacing between the bins
    :return: categorized vector
    """
    mx = np.max(vector)
    mn = np.min(vector)
    spread = mx - mn
    curr = mn
    return np.arange(mn - incr, mx + incr, incr)

def digitize_vector(vector, incr):
    return np.digitize(vector, generate_bins(vector, incr)).astype('int16')

def mi_all_to_one(x, y):
    """Runs mutual information on all columns."""
    dat1 = x
    #print x
    dat2 = y
    ncols1 = dat1.shape[1]
    ncols2 = dat2.shape[1] if dat2.ndim > 1 else 1
    mimat = np.zeros([ncols1, ncols2])
    for i in range(ncols1):
      if ncols2 > 1:
        for j in range(ncols2):
            mimat[i, j] = mi_fast(dat1[:, i], dat2[:, j])
      else:
        # case for 1-dimensional dat2
        mimat[i] = mi_fast(dat1[:, i], dat2)
      #print 'Finished %d' % i
    return mimat

def entropy(arr):
    """
    Calculates the entropy of a distribution:
        S = - SUM_i (p_i * ln p_i)
    This is the true entropy if arr is a pdf, but notice that you will have to divide by an appropriate normalization if
    p_i's are counts rather than probabilities.
    :param arr: a probability distribution
    :return: entropy of arr
    """
    return np.sum([-p * np.log(p) if p else 0 for p in arr])


def H(arr):
    """
    Calculates the entropy of a distribution:
        S = - SUM_i (p_i * ln p_i)
    This is the true entropy if arr is a pdf, but notice that you will have to divide by an appropriate normalization if
    p_i's are counts rather than probabilities.
    :param arr: a probability distribution
    :return: entropy of arr
    """
    x = arr
    y = arr
    n_samples = len(x)
    bin_range = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    n_bins = bin_range[1] - bin_range[0] + 1

    xy_c = np.histogram2d(x, y, bins=n_bins, range=[bin_range, bin_range])[0]
    x_c = np.sum(xy_c, 1)
    y_c = np.sum(xy_c, 0)
    Hx = entropy(np.ravel(x_c)) / n_samples + np.log(n_samples)
    return Hx


def h_fast(x):
    """
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
    """
    n_samples, _ = x.shape
    bin_range = [np.min(x), np.max(x)]
    n_bins = bin_range[1] - bin_range[0] + 1

    # Bincount along columns
    hist = np.apply_along_axis(lambda y: np.histogram(y, bins=n_bins, range=bin_range)[0], axis=0, arr=x)
    # Compute: -SUM_i[n_i ln n_i] along cols
    count_entropy = np.apply_along_axis(entropy, axis=0, arr=hist)
    # Compute: ln(N) - 1/N * SUM_i[n_i ln n_i]
    return np.add(np.divide(count_entropy, n_samples), np.log(n_samples))


def mi_fast(x, y):
    """
    A clever algorithm for calculating the mutual information of the arrays x and y. Pulled from Kasson git: numpy_mi.py
    :param x: an array that has been binned but not yet been histogrammed.
    :param y: an array that has been binned but not yet been histogrammed.
    :return: mutual information of x and y.
    """
    n_samples = len(x)
    bin_range = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    n_bins = bin_range[1] - bin_range[0]+1

    xy_c = np.histogram2d(x, y, bins=n_bins, range=[bin_range, bin_range])[0]
    x_c = np.sum(xy_c, 1)
    y_c = np.sum(xy_c, 0)
    Hxy = entropy(np.ravel(xy_c)) / n_samples + np.log(n_samples)
    Hx = entropy(np.ravel(x_c)) / n_samples + np.log(n_samples)
    Hy = entropy(np.ravel(y_c)) / n_samples + np.log(n_samples)
    return Hx + Hy - Hxy


def mi_fast_norm(x, y):
    """
    A clever algorithm for calculating the mutual information of the arrays x and y. Pulled from Kasson git: numpy_mi.py
    :param x: an array that has been binned but not yet been histogrammed.
    :param y: an array that has been binned but not yet been histogrammed.
    :return: mutual information of x and y.
    """
    n_samples = len(x)
    bin_range = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
    n_bins = bin_range[1] - bin_range[0] + 1

    xy_c = np.histogram2d(x, y, bins=n_bins, range=[bin_range, bin_range])[0]
    x_c = np.sum(xy_c, 1)
    y_c = np.sum(xy_c, 0)
    Hxy = entropy(np.ravel(xy_c)) / n_samples + np.log(n_samples)
    Hx = entropy(np.ravel(x_c)) / n_samples + np.log(n_samples)
    Hy = entropy(np.ravel(y_c)) / n_samples + np.log(n_samples)
    minH = np.min((Hx,Hy))
    if minH != 0:
        return (Hx + Hy - Hxy)/minH
    else:
        return 0


class MRMRConfig(object):

    def __init__(self, configuration_file, rewrite=False, pre_binned=True):
        if isinstance(configuration_file , str) :
            parser = RawConfigParser()
            parser.read(configuration_file)
            dist_filename = parser.get('files', 'distances-filename')
            entropy_filename = parser.get('files', 'entropy-filename')
            self.mRMR_filename = parser.get('files', 'mRMR-filename')
            self.num_bins = parser.getint('parameters', 'num-bins')
            self.bin_width = parser.getfloat('parameters', 'bin-width')
            self.aa = parser.getboolean('parameters', 'aa')
            self.chains = parser.getint('parameters', 'chains')
            self.dist_range = [float(i) for i in parser.get('parameters', 'distance-range').split()]
            self.weights = [float(i) for i in parser.get('parameters', 'weights').split()]
            self.resi_select = parser.get('parameters', 'excluded-residues').split()
            self.category_vect = None
        elif isinstance(configuration_file , dict):
            self.num_bins = configuration_file['num-bins']
            self.mRMR_filename = configuration_file['mRMR-filename']
            self.bin_width = configuration_file['bin-width']
            self.aa = configuration_file['aa']
            self.chains = configuration_file['chains']
            self.dist_range = configuration_file['distance-range']
            self.weights = configuration_file['weights']
            self.resi_select = configuration_file['excluded-residues']
            self.dists = configuration_file['distances']
            if 'categories' in configuration_file.keys(): #Allows for a all-to-categorical vs all-to-all MI calculation
                self.category_vect = configuration_file['categories']
            else:
                self.category_vect = None
            dist_filename = None
        else:
            raise ValueError('ERROR: Configuration needs to be an accepted format')
            exit()
        self.pre_binned = pre_binned

        if not self.resi_select:
            self.resi_select = None
        if dist_filename is not None:
            print "Loading distance file %s" % dist_filename
            init = time.time()
            self.dists = cPickle.load(open(dist_filename))
            print "Loaded %s in %f seconds" % (dist_filename, time.time() - init)
        else:
            print "Distances provided through dictionary"
        # If no entropy has been provided, entropy must be calculated for permitted residues.
        if not self.pre_binned:
            print "Distances will be binned at %f increments" % (self.bin_width)
            binned_distance_vector = []
            for i in range(self.dists.shape[1]):
                binned_distance_vector.append(digitize_vector(self.dists[:,i], self.bin_width))
            self.dists = np.vstack(binned_distance_vector).T
        if self.category_vect is None:
            print colors.HEADER + "\tENTROPY CALCULATION" + colors.ENDC
            if os.path.exists(entropy_filename) and not rewrite:
                warn_file_exists(entropy_filename)
                self.entropy = cPickle.load(open(entropy_filename))
            else:
                print "No entropy file was found or you have chosen to rewrite. Calculating entropies..."
                self.entropy = h_fast(self.dists)
                print "Dumping entopies to", entropy_filename
                cPickle.dump(self.entropy, open(entropy_filename, 'w'))
        else:
            print colors.HEADER + "\tINITAL MI CALCULATION" + colors.ENDC
            self.entropy = mi_all_to_one(self.dists, self.category_vect)




    def _get_permitted_pairs(self, resi_tpr=None):
        """
        This function allows you to restrict the pairs you wish to consider in the mRMR algorithm based on their average
        distances and/or residue name. DEER measurements are reliable for distances only in the range of about 20 - 50 nm.
        Therefore, we wish to only consider pairs with an average distance in simulation of 20 - 50 nm. Similarly, we may
        not wish to mutate certain residues that are integral to secondary structure (or lack of secondary structure) e.g.,
        PRO. Therefore, we may eliminate all PRO residues from the distance calculation as well.
        :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i.
        :param resi_select: a list of residue name selections, e.g. ['PRO', 'GLY', ...], that you wish to eliminate in the
        mRMR calculation. Use PDB residue naming scheme.
        :param resi_tpr: a single tpr so that the residue name can be mapped to atom index and therefore to pair number.
        :return: a list of permitted pair numbers.
        """

        _, num_pairs = self.dists.shape

        permitted_pairs = [i for i in range(num_pairs)]

        # EXCLUDE BASED ON DISTANCE
        if self.dist_range is not None:
            bin_range = np.divide(self.dist_range, self.bin_width)
            print "Choosing pairs with distances between", self.dist_range[0], "and", self.dist_range[1], "nm"
            print "That corresponds to bin numbers between", bin_range[0], "and", bin_range[1]
            bin_mean = np.mean(self.dists, axis=0)
            for i in range(num_pairs):
                if not (bin_range[0] < bin_mean[i] < bin_range[1]):
                    permitted_pairs.remove(i)

        # EXCLUDE BASED ON RESIDUE NAME
        if self.resi_select is not None:

            # Generate two index files:
            # 1. CA/BB that are PRO
            # 2. All CA/BB
            full_selection = "(resname " + " or resname ".join(self.resi_select) + ")"
            if self.aa:
                selection = 'mol 1 and name CA and ' + full_selection + \
                            '; mol 2 and name CA and ' + full_selection
            else:
                selection = 'mol 1 and name BB and ' + full_selection + \
                            '; mol 2 and name BB and ' + full_selection

            print "Eliminating pairs with selection:", self.resi_select

            make_ndx(tpr_filename=resi_tpr, ndx_filename='selected_restrict_resis.ndx', selection=selection)
            make_ndx(tpr_filename=resi_tpr, ndx_filename='all_resis.ndx', AA=self.aa)
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

    def mRMR(self, iters, store_in_memory=True, resi_tpr=None, permitted_pairs=None):

        """
        This method is an implementation of the mRMR algorithm described in http://dx.doi.org/10.1142/S0219720005001004
        (Here, however, we use a weighted sum of both the MID and MIQ criteria, not just one or the other).

        :param mat: A matrix of distances. mat[i,j] = distance of pair j at frame i. The matrix MUST be binned.
        Unfortunately, because of memory concerns, the xvg must be read in as bin numbers (integer) instead of
        raw distances (floats).
        :param iters: Number of mRMR pairs you wish to calculate.
        :param entropy_filename: a file for storing all distance entropies. This is a binary cPickle file (this is done so
        that if you have already calculated the entropies, the entropies can be read in quickly instead of recalculated).
        :param self.weights: self.weights for MID and MIQ criteria. This should be a list: [MID_weight, MIQ_weight]
        :param dist_restrict: the range of allowed distances as a list. I.e., [2.0, 5.0] for the usual DEER case.
        :param resi_restrict: a list of residue name selections, e.g. ['PRO', 'GLY', ...], that you wish to eliminate in the
        mRMR calculation. Use PDB residue naming scheme.
        :param resi_tpr:  a single tpr so that the residue name can be mapped to atom index and therefore to pair number.
        :param self.bin_width: This is a little clunky. The distance matrix mat must be pre-binned, but we need to know the bin
        size that was used. That way we can map bin number to absolute distance with get_permitted_pairs()
        :param rewrite: boolean to indicate whether you want to rewrite the entropy file.
        :return: a list of pair numbers, ordered from highest to lowest mRMR.
        """

        _, n_pairs = self.dists.shape  # Get the number of pairs in your matrix
        print "The number of pairs is", n_pairs

        if permitted_pairs is None:
            permitted_resis = self._get_permitted_pairs(resi_tpr=resi_tpr)
        else:
            permitted_resis = permitted_pairs

        print "Saving permitted pairs to permitted_pairs.txt in this directory..."
        np.savetxt('permitted_pairs.txt', permitted_resis)

        # Initialize a set omega. At each iteration you will pull a pair from this set
        # based on the mRMR criterion of your choice (subtraction or division). The pair
        # will be placed into the set s. s is the final set of highest mRMR pairs.

        omega = [i for i in permitted_resis]  # At first, all permitted pairs are present in omega
        s = []  # s starts as an empty set
        if store_in_memory:
            mi_mat_norm = np.zeros(shape=(n_pairs, n_pairs))  # To-do: make mi_mat only as big as num of permitted resis
            mi_mat = np.zeros(shape=(n_pairs, n_pairs))

        print colors.HEADER + "\n\tBEGINNING mRMR CALCULATION" + colors.ENDC
        mRMR_file = open(self.mRMR_filename, "w")
        for m in range(iters):  # Number of times you want to perform mRMR
            print "m =", m, "calculation"
            start_time = time.time()
            if m == 0:  # At the first step, choose the highest entropy pair.
                temp_H = [self.entropy[i] for i in permitted_resis]
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
                        if store_in_memory:
                            # First, check to see if you need to store joint entropy - you don't want
                            # to recalculate if you can use symmetry properties and previously stored
                            # values.
                            if (mi_mat[test_idx, high_idx] == 0) and (mi_mat[high_idx, test_idx] == 0):
                                mi_mat[test_idx, high_idx] = mi_fast(self.dists[:, test_idx], self.dists[:, high_idx])
                                mi_mat[high_idx, test_idx] = mi_mat[test_idx, high_idx]
                                pair_pair_mi += mi_mat[test_idx, high_idx]
                        else:
                            pair_pair_mi += mi_fast(self.dists[:, test_idx], self.dists[:, high_idx])

                    mRMR_scores.append(self.weights[0] * (self.entropy[test_idx] - pair_pair_mi / m))
                omega_idx = np.nanargmax(mRMR_scores)  # Find the test pair with the highest mRMR value
                print 'shape: ', np.array(mRMR_scores).shape
                if m == 1:
			mRMR_stack = [mRMR_scores]
			mRMR_stack_norm = [omega[:]]
		else:
			mRMR_stack.append(mRMR_scores)
			mRMR_stack_norm.append(omega[:])
		sa = np.sort(np.concatenate(mRMR_scores))[::-1]
                print 'value: ', mRMR_scores[omega_idx]
                print sa
                if np.where(np.isnan(sa))[0].shape[0] > 0:
                    print sa[np.max(np.where(np.isnan(sa))) + 1] - sa[np.max(np.where(np.isnan(sa))) + 2]
                else:
                    print sa[0] - sa[1]
                s.append(omega.pop(omega_idx))  # Pop that test pair from omega, place in s
            mRMR_file.write("%i\n" % s[m])
            end_time = time.time()
            print "Time elapsed for m =", m, "calculation:", end_time-start_time
        mRMR_file.close()
        return np.array(s), mRMR_stack, mRMR_stack_norm



