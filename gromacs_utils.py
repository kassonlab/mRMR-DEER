import os
import numpy as np
import time


class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def warn_file_exists(filename):
    print colors.WARNING + "The file " + filename + " already exists: use rewrite=True to override" + colors.ENDC


def strip_filenames(filenames):
    '''
    This strips off "_cntr" and "_whole" substrings in each filename of a list of filenames. The "_cntr" and "_whole"
    substrings are used to name pbc -whole and pbc -cntr output xtcs, so it is good to be able to strip off those
    extras. BEWARE this will sort the files!
    :param filenames: A list of filenames from which "_cntr" and "_whole" will be removed.
    :return: A list of the edited filenames, no extensions.
    '''
    stripped_filenames = []
    for filename in filenames:
        if '_cntr' in filename:
            stripped_filenames.append(filename.replace('_cntr', '')[0:-4])
        elif '_whole' in filename:
            stripped_filenames.append(filename.replace('_whole', '')[0:-4])
        else:
            stripped_filenames.append(filename[0:-4])
    return np.unique(stripped_filenames)


def make_ndx(tpr_filename, ndx_filename, selection=None, rewrite=False, AA=True):
    '''
    Generates an index file using gmx select. The output index file will have the following groups:
        mol_1_and_name_CA
        mol_2_and_name_CA
        (mol_1_and_name_CA)_or_(mol_2_and_name_CA)
        mol_1_or_mol_2
    Do NOT alter this selection if you are using this with mRMR, except you may flag whether the tpr is a coarse-grained
    or all-atom simulation. If all-atom, alpha carbons are selected, and if coarse-grained, backbone atoms are selected.
    You may provide your own selection if you are not using this with mRMR.
    :param tpr_filename: path to a single tpr
    :param ndx_filename: path to output index file
    :param rewrite: boolean to indicate whether you want to overwrite an existing index file of the same name
    :return: None. os.system sends the appropriate gmx select command to console.
    '''

    if rewrite or not os.path.exists(ndx_filename):
        if selection is None:
            if AA:
                selection = 'mol 1 and name CA; mol 2 and name CA; ' \
                            '(mol 1 and name CA) or (mol 2 and name CA); mol 1 or mol 2'
            else:
                selection = 'mol 1 and name BB; mol 2 and name BB; ' \
                            '(mol 1 and name BB) or (mol 2 and name BB); mol 1 or mol 2'
        os.system('gmx select -s %s -select "%s" -on %s' % (tpr_filename, selection, ndx_filename))
    else:
        warn_file_exists(ndx_filename)


def read_ndx(ndx_filename):
    '''
    Reads a Gromacs ndx file into a dictionary. The dictionary keys are the group names. If used with mRMR algorithm,
    group names should be the names discussed in make_ndx
    :param ndx_filename: path to ndx file
    :return: dictionary of the form index_dict[group id] = [atom ids in group id]
    '''

    index_dict = {}
    index_file = open(ndx_filename, 'r')
    while 1:
        newline = index_file.readline()
        if not newline:
            break

        split_line = newline.split()

        if split_line:
            if split_line[0] == '[':
                current_group = split_line[1]
                index_dict[current_group] = []
            else:
                index_dict[current_group].append([int(i) for i in split_line])

    for key in index_dict.keys():
        index_dict[key] = np.concatenate(index_dict[key], axis=1)

    return index_dict


def make_xvg(tpr_filename, xtc_filename, ndx_filename, xvg_filename, length_mol1, rewrite=False):
    '''
    Generates an xvg file of distances using an updated gmx mindist which can output rectangular distance matrices.
    Calculates the distances between all alpha carbons/ backbone atoms of molecule 1 to all alpha carbons/ backbone
    atoms of molecule 2. Distances are calculated at each time step.
    The xvg will therefore have the form:
    t_0 distance(mol1atom1,mol2atom1) distance(mol1atom2, mol2atom1) ... distance(mol1atomN, mol2atomM)
    t_1 distance(mol1atom1,mol2atom1) ...                                distance(mol1atomN, mol2atomM)
    ...

    :param tpr_filename: a single tpr for gmx mindist input
    :param xtc_filename: a single xtc for gmx mindist input
    :param ndx_filename: a single ndx for gmx mindist input (it MUST have the form described in previous function
    make_ndx.)
    :param xvg_filename: a single xvg for gmx mindist output
    :param length_mol1: The size of the first protein, mol_1_and_name_CA. Input for gmx mindist.
    :param rewrite: boolean to indicate whether you want to overwrite an existing xvg file of the same name
    :return: None. os.system sends the appropriate gmx mindist command to console.
    '''
    if rewrite or not os.path.exists(xvg_filename):
        os.system('echo 2 | gmx mindist -f %s -s %s -n %s -rectmatrix %i -ng 0 -od %s' %
                  (xtc_filename, tpr_filename, ndx_filename, length_mol1, xvg_filename))
    else:
        warn_file_exists(xvg_filename)


def read_xvg(xvg_filenames, bin_size=0, contact=0, one_traj=True, skip_time=True):
    '''
    Reads a series of xvg files into a single distance matrix. You may keep each trajectory separate, or concatenate all
    the trajectories together for use with mRMR.
    :param xvg_filenames: a list of xvg filenames
    :param bin_size: if you wish to bin the distances for use with mRMR, provide a bin size in nm. If bin_size is 0,
    distances will be kept as floating point.
    :param contact: if you want to get contact maps instead of distances, specify a contact length in nm.
    :param one_traj: boolean to indicate whether trajectories are concatenated. Must be True
    for use with mRMR algorithm.
    :param skip_time: whether you want to read in the time step as well as the distances. This was added for
    flexibility, but isn't useful for mRMR. Need to set True.
    :return: array of distances. data[i,j] = pair j distance value at frame number i if you are not keeping track of
    time or the trajectories. If you are keeping track of the trajectory number, data[i,0] = traj_num. If you are
    keeping track of the time, data[i,1]=time.
    '''

    data = []
    traj_num = 0

    if bin_size != 0:
        print colors.HEADER + "Distances will be binned with bin size: " + str(bin_size) + colors.ENDC
    if contact != 0:
        print colors.HEADER + "Contact maps will be loaded with contact cutoff: " + str(contact) + " nm" + colors.ENDC

    for xvg_filename in xvg_filenames:
        start_time = time.time()
        print "Reading file " + xvg_filename + '...'
        inputfile = open(xvg_filename, 'r')

        while 1:
            newline = inputfile.readline()
            if not newline:
                break
            if newline[0] not in ['@', '#']:
                if one_traj:
                    temp_vec = []
                else:
                    temp_vec = [traj_num]  # The first column of the data should be trajectory number
                col_num = 0
                for element in newline.split():
                    if skip_time:
                        if col_num > 0:
                            if bin_size != 0:
                                dist = int(float(element)/bin_size)
                            else:
                                dist = float(element)
                            temp_vec.append(dist)
                    else:
                        dist = float(element)
                        temp_vec.append(dist)

                    if (contact > 0) and (col_num > 0):
                        if dist < contact:
                            dist = 1
                        else:
                            dist = 0
                        temp_vec.append(dist)
                    col_num += 1

                data.append(temp_vec)
        end_time = time.time()
        print "Time elapsed to read trajectory", traj_num, ":", end_time-start_time
        traj_num += 1

    return np.array(data)


def dump_frame(xtc, tpr, ndx, time, pdb):
    '''
    Uses gmx trjconv -dump to pull the frame at "time" and dump it to a pdb.
    This is useful for restarting all-atom simulations from coarse-grained
    simulations. For use with the file cluster_dump.py.
    :param xtc: A single xtc.
    :param tpr: A single tpr.
    :param ndx: An index file, not necessarily of the form in make_ndx
    :param time: The time (in ps) you want to dump.
    :param pdb: The name of the output pdb.
    :return: None. os.system sends appropriate gmx command to console
    '''
    os.system('gmx trjconv -f %s -s %s -n %s -dump %f -o %s' % (xtc, tpr, ndx, time, pdb))


def pbc_center(xtcs, tprs, ndx, rewrite=False):
    '''
    This function is used for pre-processing trajectories that have pbc issues.
    It does the following:
        1. Make any split atoms whole using gmx trjconv -pbc whole
        2. Center on a single protein. Ideally, we would center on both proteins.
           However, there are problems with gmx trjconv -center on multiple
           chains. See
           https://mailman-1.sys.kth.se/pipermail/gromacs.org_gmx-users/2010-August/053340.html
           To do: option to center on one or the other protein. Right now we just
           center on the second protein, Opa.
    :param xtcs: A list of xtcs for pre-processing
    :param tprs: A list of tprs. Must be in the same order as the xtc list.
    :param ndx: A single ndx filename. The ndx file will be written using make_ndx if it
    does not already exist.
    :param rewrite: Whether to rewrite the ndx file or the intermediate xtc files.
    :return: None. Writes appropriate gmx commands to console.
    '''

    make_ndx(tpr_filename=tprs[0], ndx_filename=ndx, rewrite=rewrite)
    stripped = strip_filenames(xtcs)
    orig_xtcs = [strip +'.xtc' for strip in stripped]
    n_files = len(orig_xtcs)

    for i in range(n_files):
        whole_name = stripped[i] + '_whole.xtc'
        cntr_name = stripped[i] + '_cntr.xtc'

        if os.path.exists(whole_name) and not rewrite:
            print "The file", whole_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 3 3 | gmx trjconv -f %s -s %s -pbc whole -n %s -o %s' %
                      (orig_xtcs[i], tprs[i], ndx, whole_name))

        if os.path.exists(cntr_name) and not rewrite:
            print "The file", cntr_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 1 3 | gmx trjconv -f %s -s %s -pbc mol -ur compact -center -n %s -o %s' %
                      (whole_name, tprs[i], ndx, cntr_name))


def write_pdbs(xtcs, tprs, ndx, pdbs, dt=100000, rewrite=False):
    '''
    Uses gmx trjconv to write out pdbs at intervals of dt. The pdbs are written with connect
    records. This is really for visualization to make sure centered trajectories are not
    moving throught the periodic box.
    :param xtcs: A list of xtcs.
    :param tprs: A list of tprs. MUST BE IN CORRECT ORDER
    :param ndx: An index file of the form described in the method make_ndx
    :param pdbs: A list of output pdbs. Must be the same length as your xtc list.
    :param dt: Intervals at which you wish to write out pdbs. See GROMACS documentation of
    trjconv.
    :param rewrite: Whether to overwrite the pdbs.
    :return: None. Writes appropriate gmx commands to console.
    '''
    n_files = len(xtcs)
    for i in range(n_files):
        if os.path.exists(pdbs[i]) and not rewrite:
            warn_file_exists(pdbs[i])
        else:
            os.system('echo 3 | gmx trjconv -f %s -s %s -n %s -conect -dt %i -o %s' %
                      (xtcs[i], tprs[i], ndx, dt, pdbs[i]))


def pre_process(xtcs, tprs, ndx, xvgs, bin_size=0, AA=True, one_traj=True, rewrite=False):
    '''
    This function:
        1. Makes a single ndx file using a single tpr.
        2. Reads that ndx file to determine the size of the first protein.
        3. Takes care of pbc problems by clustering and centering with pbc_center.
        4. Makes a bunch of xvg files using a list of xtcs and tprs.
        5. Reads in those xvg files to a single distance matrix.
        6. Returns the distance matrix.
    :param xtcs: list of xtcs
    :param tprs: list of tprs
    :param ndx: single index file name. If the index file does not already exist, one will
    be created using make_ndx
    :param xvgs: a list of xvg file names. If any one of the xvgs does not already exist, it
    will be created using make_xvg.
    :param bin_size: 0 if you do not want to bin the distance data, otherwise the bin size in nm.
    :param AA: indicates whether the trajectories are all-atom or coarse-grained.
    :param one_traj: whether you want to concatenate the trajectories. You must concatenate for
    use with mRMR algorithm.
    :param rewrite: boolean to indicate whether you want to overwrite an existing index file of the
    same name
    :return: distance matrix.
    '''

    cntr_xtcs = []
    n_files = 0

    make_ndx(tprs[0], ndx, rewrite=rewrite, AA=AA)
    mol_ndx = read_ndx(ndx_filename=ndx)
    if AA:
        size_mol1 = len(mol_ndx['mol_1_and_name_CA'])
    else:
        size_mol1 = len(mol_ndx['mol_1_and_name_BB'])

    pbc_center(xtcs, tprs, ndx)

    for xtc in xtcs:
        if 'cntr' in xtc:
            cntr_xtcs.append(xtc)
            n_files += 1

    for i in range(n_files):

        make_xvg(tpr_filename=tprs[i], xtc_filename=cntr_xtcs[i], ndx_filename=ndx,
                 xvg_filename=xvgs[i], length_mol1=size_mol1, rewrite=rewrite)

    return read_xvg(xvg_filenames=xvgs, one_traj=one_traj, bin_size=bin_size), mol_ndx


