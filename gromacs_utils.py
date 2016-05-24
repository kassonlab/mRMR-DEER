import os
import numpy as np


def make_ndx(tpr_filename, ndx_filename, rewrite=False, AA=True):
    '''
    Generates an index file using gmx select. The output index file will have the following groups:
        mol_1_and_name_CA
        mol_2_and_name_CA
        (mol_1_and_name_CA)_or_(mol_2_and_name_CA)
        mol_1_or_mol_2
    There is NO flexibility in this selection, except you may flag whether the tpr is a coarse-grained
    or all-atom simulation. If all-atom, alpha carbons are selected, and if coarse-grained, backbone
    atoms are selected.
    :param tpr_filename: path to a single tpr
    :param ndx_filename: path to output index file
    :param rewrite: boolean to indicate whether you want to overwrite an existing index file of the
    same name
    :return: None. os.system sends the appropriate gmx select command to console.
    '''

    if rewrite or not os.path.exists(ndx_filename):
        if AA:
            selection = 'mol 1 and name CA; mol 2 and name CA; ' \
                        '(mol 1 and name CA) or (mol 2 and name CA); mol 1 or mol 2'
        else:
            selection = 'mol 1 and name BB; mol 2 and name BB; ' \
                        '(mol 1 and name BB) or (mol 2 and name BB); mol 1 or mol 2'
        os.system('gmx select -s %s -select "%s" -on %s' % (tpr_filename, selection, ndx_filename))
    else:
        print "The file " + ndx_filename + " already exists: use rewrite=True to override"


def read_ndx(ndx_filename):
    '''
    Reads a Gromacs ndx file into a dictionary. The dictionary keys are the group names.
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
    Generates an xvg file of distances using an updated gmx mindist which can output rectangular
    distance matrices. Calculates the distances between all alpha carbons/ backbone atoms of molecule
    1 to all alpha carbons/ backbone atoms of molecule 2. Distances are calculated at each time step.
    The xvg will therefore have the form:
    t_0 distance(mol1atom1,mol2atom1) distance(mol1atom2, mol2atom1) ... distance(mol1atomN, mol2atomM)
    t_1 distance(mol1atom1,mol2atom1) ...                                distance(mol1atomN, mol2atomM)
    ...

    :param tpr_filename: a single tpr for gmx mindist input
    :param xtc_filename: a single xtc for gmx mindist input
    :param ndx_filename: a single ndx for gmx mindist input (it MUST have the form described in previous
    function make_ndx. If it does not, everything will BLOW UP)
    :param xvg_filename: a single xvg for gmx mindist output
    :param length_mol1: The size of the first protein, mol_1_and_name_CA. Input for gmx mindist
    :param rewrite: boolean to indicate whether you want to overwrite an existing xvg file of the
    same name
    :return: None. os.system sends the appropriate gmx mindist command to console.
    '''
    if rewrite or not os.path.exists(xvg_filename):
        os.system('echo 2 | gmx mindist -f %s -s %s -n %s -rectmatrix %i -ng 0 -od %s' %
                  (xtc_filename, tpr_filename, ndx_filename, length_mol1, xvg_filename))
    else:
        print "The file " + xvg_filename + " already exists: use rewrite=True to override"


def read_xvg(xvg_filenames, bin_size=0, one_traj=True, skip_time=True):
    '''
    Reads a series of xvg files into a single distance matrix. You may keep each trajectory
    separate, or concatenate all the trajectories together for use with mRMR.
    :param xvg_filenames: a list of xvg filenames
    :param bin_size: if you wish to bin the distances for use with mRMR, provide a bin size in
    nm. If bin_size is 0, distances will be kept as floating point.
    :param one_traj: boolean to indicate whether trajectories are concatenated. Must be True
    for use with mRMR algorithm.
    :param skip_time: whether you want to read in the time step as well as the distances.
    This was added for flexibility, but isn't useful for mRMR. Need to set True.
    :return: array of distances. dist[i,j] = pair j distance value at frame number i.
    '''
    all_data = []
    for xvg_filename in xvg_filenames:
        print "Reading file " + xvg_filename + '...'
        data = []
        inputfile = open(xvg_filename, 'r')

        while 1:
            newline = inputfile.readline()
            if not newline:
                break
            if newline[0] not in ['@', '#']:
                temp_vec = []
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

                    col_num += 1
                data.append(temp_vec)
        all_data.append(data)
    if one_traj:
        all_data = np.concatenate(all_data, axis=0)
    return np.array(np.squeeze(all_data))


def pbc_center(xtcs, tprs, ndx, rewrite=False):
    '''
    This function is used for pre-processing trajectories that have pbc issues.
    It does the following:
        1. Make any split atoms whole using gmx trjconv -pbc whole
        2. Eliminate any jumps across the periodic box using gmx trjconv -pbc nojump
        3. Move the center of mass of the protein into the box using gmx trjconv -pbc mol
        4. Center the proteins in the box using gmx trjconv -center
    :param xtcs: A list of xtcs for pre-processing
    :param tprs: A list of tprs. Must be in the same order as the xtc list.
    :param ndx: A single ndx filename. The ndx file will be written using make_ndx if it
    does not already exist.
    :param rewrite: Whether to rewrite the ndx file or the intermediate xtc files.
    :return:
    '''

    make_ndx(tpr_filename=tprs[0], ndx_filename=ndx, rewrite=rewrite)

    orig_xtcs = []
    n_files = 0
    for xtc in xtcs:
        if ("whole" not in xtc) and ("nojump" not in xtc) and ("mol" not in xtc) \
                and ("cntr" not in xtc):
            orig_xtcs.append(xtc)
            n_files += 1

    stripped_xtcs = [orig_xtc[0:-4] for orig_xtc in orig_xtcs]

    for i in range(n_files):
        whole_name = stripped_xtcs[i] + '_whole.xtc'
        noj_name = stripped_xtcs[i] + '_nojump.xtc'
        mol_name = stripped_xtcs[i] + '_mol.xtc'
        cntr_name = stripped_xtcs[i] + '_cntr.xtc'

        if os.path.exists(whole_name) and not rewrite:
            print "The file", whole_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 3 3 | gmx trjconv -f %s -s %s -pbc whole -n %s -o %s' %
                      (orig_xtcs[i], tprs[i], ndx, whole_name))

        if os.path.exists(noj_name) and not rewrite:
            print "The file", mol_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 3 3 | gmx trjconv -f %s -s %s -pbc nojump -n %s -o %s' %
                      (whole_name, tprs[i], ndx, noj_name))

        if os.path.exists(mol_name) and not rewrite:
            print "The file", mol_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 3 3 | gmx trjconv -f %s -s %s -pbc mol -n %s -o %s' %
                      (noj_name, tprs[i], ndx, mol_name))

        if os.path.exists(cntr_name) and not rewrite:
            print "The file", cntr_name, "already exists: use rewrite=True to override"
        else:
            os.system('echo 3 3 | gmx trjconv -f %s -s %s -center -n %s -o %s' %
                      (mol_name, tprs[i], ndx, cntr_name))


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


