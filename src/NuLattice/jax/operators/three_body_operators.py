"""
This module provides functions to define three body operators on the 3D lattice
"""

__authors__ = ["Thomas Papenbrock, Maxwell Rothman"]
__credits__ = ["Thomas Papenbrock, Maxwell Rothman"]
__copyright__ = "(c) Thomas Papenbrock, Maxwell Rothman"
__license__ = "BSD-3-Clause"
__date__ = "2026-05-15"


import scipy.sparse as sparse
import sys
from concurrent.futures import ProcessPoolExecutor
import os

import NuLattice.jax.operators.one_body_operators as ob_ops
import NuLattice.jax.lattice as lat


def rho_cubed_NO(rho_n, mult, min_val, max_mem=0):
    """
    Computes the density cubed and normal ordered at a lattice site n

    :param rho_n:   Array storing the density at lattice site n
    :type rho_n:    scipy.sparse.coo_array
    :param mult:    Factor to scale the calculation by
    :type mult:
    :param min_val: minimum value to be saved in the operator
    :type min_val:  float
    :param max_mem: Optional; Maximum memory size for the value array
                    to get to before compressing into a sparse format.
                    If left as 0, no limit to the memory will be set
    :type max_mem:  int
    :returns:       The density operator squared and normal ordered, for
                    V^{ij}_{kl}, it only stores i<j and k<l
    :rtype:         scipy.sparse.csr_array
    """
    tmp_col = []
    tmp_row = []
    tmp_val = []
    dim = rho_n.shape[0]
    ret = sparse.csr_array((dim**3, dim**3))
    for a, d, v in zip(rho_n.row, rho_n.col, rho_n.data):
        for b, e, w in zip(rho_n.row, rho_n.col, rho_n.data):
            for c, f, x in zip(rho_n.row, rho_n.col, rho_n.data):
                matele = mult * v * w * x
                if abs(matele) < min_val:
                    continue
                if a < b and b < c and d < e and e < f:
                    # ret.append([a, b, c, d, e, f, matele])
                    tmp_col.append(a + b * dim + c * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(matele)

                if b < a and a < c and d < e and e < f:
                    # ret.append([b, a, c, d, e, f, -matele])
                    tmp_col.append(b + a * dim + c * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(-matele)

                if c < a and a < b and d < e and e < f:
                    # ret.append([c, a, b, d, e, f, matele])
                    tmp_col.append(c + a * dim + b * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(matele)

                if a < c and c < b and d < e and e < f:
                    # ret.append([a, c, b, d, e, f, -matele])
                    tmp_col.append(a + c * dim + b * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(-matele)

                if c < b and b < a and d < e and e < f:
                    # ret.append([c, b, a, d, e, f, -matele])
                    tmp_col.append(c + b * dim + a * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(-matele)

                if b < c and c < a and d < e and e < f:
                    # ret.append([b, c, a, d, e, f, matele])
                    tmp_col.append(b + c * dim + a * dim**2)
                    tmp_row.append(d + e * dim + f * dim**2)
                    tmp_val.append(matele)
                if (
                    max_mem != 0
                    and sys.getsizeof(tmp_val)
                    + sys.getsizeof(tmp_col)
                    + sys.getsizeof(tmp_row)
                    > max_mem
                ):
                    ret += sparse.csr_array(
                        (tmp_val, (tmp_row, tmp_col)), shape=(dim**3, dim**3)
                    )
                    tmp_val = []
                    tmp_row = []
                    tmp_col = []
    return ret + sparse.csr_array((tmp_val, (tmp_row, tmp_col)), shape=(dim**3, dim**3))


def shortRangeV_3body(
    lattice,
    myL,
    sL,
    sNL,
    c0,
    spin=2,
    isospin=2,
    verbose=False,
    min_val=0,
    max_mem=0,
    sites=None,
):
    """
    Generates the leading-order short-range interaction defined in equation 13

    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param c0:      strength of the short range interaction in MeV
    :type c0:       float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param max_mem: Optional; maximum memory for all of the temporary float arrays to take
                    up before compressing into a sparse format. If set to 0, it will completely
                    fill the array before compressing. NOTE: this utilizes a multiprocess to
                    parallelize the loop over all sites, so the memory for each site will be
                    max_mem / cpu_count and you should set the memory limit accordingly
    :param sites:   Optional; Give default value or None in order to compute the interaction at
                    all sites, or give a list of sites in the format [i, j, k] to only compute it
                    at the given sites
    :type sites:    list[int,int,int]
    :return:        A sparse csr_array representing V^{abc}_{ijk} in MeV with the row
                    indicies corresponding to a + b * spin*isospin*L^3 and column
                    indices corresponding to i + j * spin*isospin*L^3
    :rtype:         scipy.sparse.csr_array()
    """
    rho_smeared = ob_ops.get_smeared_dens(
        lattice,
        myL,
        sL,
        sNL,
        op1b=None,
        spin=spin,
        isospin=isospin,
        verbose=verbose,
        sites=sites,
    )
    dim = myL**3 * spin * isospin
    ret = sparse.csr_array((dim**3, dim**3))
    if verbose:
        print("Generating Interaction...", end="")
    with ProcessPoolExecutor() as executor:
        size = len(rho_smeared)
        max_mem_proc = max_mem / os.cpu_count()
        for val in executor.map(
            rho_cubed_NO,
            rho_smeared,
            [c0] * size,
            [min_val] * size,
            [max_mem_proc] * size,
        ):
            ret += val
    if verbose:
        print("Interaction Generated")
    return ret


def sparse_to_list_3body(sparse_int, myL, spin=2, isospin=2):
    """
    Takes sparse 3-body interaction and converts it to a list

    :param sparse_int:  Interaction stored as a sparse matrix
    :type sparse_int:   scipy.sparse.csr_array
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spin:         int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of lists [a,b,c,d,e,f,v] where a, b, c, d, e, and f
                        are indices and v is the value for V^abc_def
    :rtype:             list[[int, int, int, int, int, int float]]
    """
    ret = []
    dim = myL**3 * spin * isospin
    sparse_int_coo = sparse_int.tocoo()
    for i, j, v in zip(sparse_int_coo.row, sparse_int_coo.col, sparse_int_coo.data):
        a = i % dim
        b = ((i - a) // dim) % dim
        c = (((i - a) // dim) - b) // dim
        d = j % dim
        e = ((j - d) // dim) % dim
        f = (((j - d) // dim) - e) // dim
        ret.append([a, b, c, d, e, f, v])


def change_lat_3body(inter, origL, newL, spin=2, isospin=2):
    """
    Changes a three-body interaction in list format for a given L to a new L

    :param inter:   interaction stored as a list of lists [a,b,c,d,e,f,v]
                    where a, b, c, and d are indices and v is the value
                    for V^abc_def
    :type inter:    list[(int, int, int, int, int, int, float)]
    :param origL:   original L for the basis of inter
    :type origL:    int
    :param newL:    new L to return the basis of inter
    :type newL:     int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        interaction in the basis of the new L in the same
                    list format
    :rtype:         list[(int, int, int, int, int, int, float)]
    """
    new_inter = [[] for _ in range(len(inter))]
    for i in range(len(inter)):
        a, b, c, d, e, f, val = inter[i]
        lst = []
        ind_lst = [a, b, c, d, e, f]
        for ind in ind_lst:
            lst.append(
                lat.state2index(
                    lat.index2state(ind, origL, spin, isospin), newL, spin, isospin
                )
            )
        lst.append(val)
        new_inter[i] = lst
    return new_inter
