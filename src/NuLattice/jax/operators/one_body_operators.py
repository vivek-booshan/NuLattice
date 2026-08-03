"""
This module provides functions to define one body operators on the 3D lattice
"""

__authors__ = ["Thomas Papenbrock, Maxwell Rothman"]
__credits__ = ["Thomas Papenbrock, Maxwell Rothman"]
__copyright__ = "(c) Thomas Papenbrock, Maxwell Rothman"
__license__ = "BSD-3-Clause"
__date__ = "2026-05-15"

import numpy as np
import scipy.sparse as sparse
import math

import NuLattice.jax.lattice as lat
import NuLattice.utils.constants as consts


def list_to_sparse1b(mylist, sparsetype="csr"):
    """
    transforms a list of matrix elements of a 1-body operator to a sparse format
    :param mylist:     the one-body operator
    :type mylist:      list of lists [[p,q, val, ...] with int p, q, and real or complex val
    :param sparsetype: the desired format (only "csr" and "coo" implemented as of now)
    :type sparsetype:  string (only "csr" and "coo" implemented as of now)
    :return:           the sparse matrix of mylist
    :rtype:            scipy.sparse.csr_array or scipy.sparse.coo_array
    """
    row = [item[0] for item in mylist]
    col = [item[1] for item in mylist]
    val = [item[2] for item in mylist]
    if sparsetype == "coo":
        return sparse.coo_array((val, (row, col)))
    else:
        return sparse.csr_array((val, (row, col)))


def indConv(ind, myL):
    """
    Gets x,y,z indices from combined index

    :param ind: index in lattice, given by x + y * myL + z * myL ^ 2
    :type ind:  int
    :param myL: number of lattice sites in each direction
    :type myL:  int
    :return:    list of x, y, z
    :rtype:     list[int]
    """
    x = ind % myL
    y = ((ind - x) // myL) % myL
    z = ((ind - x) // myL - y) // myL
    return x, y, z


def tKin(myL, Nk, a_lat, spin=2, isospin=2, mass=consts.MASS):
    """
    computes 1-body kinetic energy matrix elements.

    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param Nk:      number of neighbors along each axis to use(1 for nearest-neighbor, 2 for
                    next-to-nearest-neighbor, etc)
    :type Nk:       int
    :param a_lat:   lattice spacing divided by hbar c
    :type a_lat:    float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :param mass:    Optional; nucleon mass
    :type mass:     float
    :return:        list of tuples [i, j, value] where i and j are indices in the single-particle
                    basis, and value is the value of the matrix element Tij in MeV
    :rtype:         list[(int, int, float)]
    """
    h = -1.0 / 2.0 / (mass * a_lat)

    KK = np.zeros([myL**3, myL**3])
    cf0 = 0.0

    r = np.arange(myL**3)
    nx = np.mod(r, myL)
    ny = np.mod((r - nx) // myL, myL)
    nz = (r - nx - ny * myL) // (myL**2)

    for k in range(1, Nk + 1):
        cf = (
            (-1) ** (k + 1)
            * 2.0
            * (math.factorial(Nk) / math.factorial(Nk - k))
            / (math.factorial(Nk + k) / math.factorial(Nk))
            / k**2
            * h
        )
        cf0 -= 2 * cf

        rxp = np.mod(nx + k, myL) + ny * myL + nz * myL**2
        rxm = np.mod(nx - k, myL) + ny * myL + nz * myL**2
        ryp = nx + np.mod(ny + k, myL) * myL + nz * myL**2
        rym = nx + np.mod(ny - k, myL) * myL + nz * myL**2
        rzp = nx + ny * myL + np.mod(nz + k, myL) * myL**2
        rzm = nx + ny * myL + np.mod(nz - k, myL) * myL**2

        KK[r, rxp] += cf
        KK[r, rxm] += cf
        KK[r, ryp] += cf
        KK[r, rym] += cf
        KK[r, rzp] += cf
        KK[r, rzm] += cf

    cf0 *= 3
    KK[r, r] += cf0
    ret = []
    for i in range(myL**3):
        for j in range(myL**3):
            val = KK[i][j]
            if val == 0:
                continue
            indx1, indy1, indz1 = indConv(i, myL)
            indx2, indy2, indz2 = indConv(j, myL)
            for tz in range(isospin):
                for sz in range(spin):
                    state1 = [indx1, indy1, indz1, tz, sz]
                    ind1 = lat.state2index(state1, myL, spin, isospin)
                    state2 = [indx2, indy2, indz2, tz, sz]
                    ind2 = lat.state2index(state2, myL, spin, isospin)
                    ret.append([ind1, ind2, val / a_lat])
    return ret


def nonLocOp(site, myL, sNL, sz, tz, spin=2, isospin=2):
    """
    Generates the non-local creation/annihilation operator as definedin equations 3/4

    :param site:    x,y,z coordinate on the lattice
    :type site:     [int,int,int]
    :param myL:     size of lattice
    :type myL:      int
    :param sNL:     strength of the non-local smearing
    :type sNL:      float
    :param sz:      spin
    :type sz:       int
    :param tz:      isospin
    :type sz:       int
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :return:        1D list equivalent to the non-local creation/annihilation operator at
                    a given site
    :rtype:         numpy list[float]
    """
    ret = np.zeros(myL**3 * spin * isospin)
    pos = lat.state2index([site[0], site[1], site[2], tz, sz], myL, spin, isospin)
    rx = lat.state2index(
        [lat.right(site[0], myL), site[1], site[2], tz, sz], myL, spin, isospin
    )
    ry = lat.state2index(
        [site[0], lat.right(site[1], myL), site[2], tz, sz], myL, spin, isospin
    )
    rz = lat.state2index(
        [site[0], site[1], lat.right(site[2], myL), tz, sz], myL, spin, isospin
    )
    lx = lat.state2index(
        [lat.left(site[0], myL), site[1], site[2], tz, sz], myL, spin, isospin
    )
    ly = lat.state2index(
        [site[0], lat.left(site[1], myL), site[2], tz, sz], myL, spin, isospin
    )
    lz = lat.state2index(
        [site[0], site[1], lat.left(site[2], myL), tz, sz], myL, spin, isospin
    )
    ret[pos] += 1
    ret[rx] += sNL
    ret[lx] += sNL
    ret[ry] += sNL
    ret[ly] += sNL
    ret[rz] += sNL
    ret[lz] += sNL
    return ret


def rho_op(site, myL, op1b=None, sNL=0, op_fac=1.0, spin=2, isospin=2):
    """
    Generates the (non-)local density operator as defined in equation 9

    :param site:    location in lattice
    :type site:     [int, int, int]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param op1b:    one-body operator that acts between a-dagger and a on the indicated site
    :type op1b:     scipy.sparse.csr_array
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param op_fac:  factor that multiplies the operator
    :type op_fac:   float
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :returns:       (non-)local density operator for the given site
    :rtype:         scipy.sparse.coo_array()
    """
    dim = myL**3 * spin * isospin
    if op1b is None:
        op_sparse = op_fac * sparse.csr_array(np.identity(dim))
    else:
        op_sparse = op_fac * op1b

    ret = sparse.csr_array(np.zeros([dim, dim]))
    for sz in range(spin):
        for tz in range(isospin):
            op_nl = nonLocOp(site, myL, sNL, sz, tz, spin, isospin)
            matvec = op_sparse @ op_nl
            ret += sparse.csr_array(np.outer(op_nl, matvec))
    return ret.tocoo()


def lattice_one(lattice, myL, spin=2, isospin=2):
    """
    computes elements of the 1-body identity / unit / one operator on the lattice

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    res = []
    dim = myL**3 * spin * isospin
    for i in range(dim):
        res.append([i, i, 1.0])
    return res


def x_k(my_basis, k):
    """
    returns the 1-body matrix representation (as a list) of the operator x_k
    my_basis: list of basis states
    k:        direction k=1, 2, or 3
    return: 1-body matrix representation (as a list) of the operator x_k
    """
    x = []
    if k < 1 or k > 3:
        return x
    for i, state in enumerate(my_basis):
        pos = state[k - 1]
        x.append([i, i, pos])
    return x

def tau(lattice, myL, component, spin=2, isospin=2):
    component = component.lower()
    sites = np.asarray(lattice)

    basis_size = myL**3 * isospin * spin
    internal_size = isospin * spin

    # Spatial contribution to the basis index, shape: (# sites, 1, 1)
    site_base = (
        ((sites[:, 0] * myL + sites[:, 1]) * myL + sites[:, 2])
        * internal_size
    )[:, None, None]

    # Shapes are chosen so broadcasting produces:
    # (# sites, iso, spin).
    tz = np.arange(isospin, dtype=np.int64)[None, :, None]
    sz = np.arange(spin, dtype=np.int64)[None, None, :]

    # Column indices are input states
    columns_3d = site_base + tz * spin + sz

    if component == "z":
        rows = columns_3d.ravel()
        columns = rows

        values = np.broadcast_to(
            tz.astype(np.float64) - 0.5,
            columns_3d.shape,
        ).ravel()

    else:
        # tau_x and tau_y flip the two-state isospin index.
        flipped_tz = 1 - tz
        rows_3d = site_base + flipped_tz * spin + sz

        rows = rows_3d.ravel()
        columns = columns_3d.ravel()

        if component == "x":
            values = np.full(
                columns.size,
                0.5,
                dtype=np.float64,
            )
        else:
            # Preserves the sign convention in the original tau_y:
            # tz=0 -> -0.5j
            # tz=1 -> +0.5j
            tz_values = np.where(tz == 0, -0.5j, 0.5j)

            values = np.broadcast_to(
                tz_values,
                columns_3d.shape,
            ).ravel()

    operator = sparse.coo_array(
        (values, (rows, columns)),
        shape=(basis_size, basis_size),
    )

    return operator.tocsr()

# def tau_x(lattice, myL, spin=2, isospin=2):
#     """
#     computes matrix elements for 1-body isospin-x operator.

#     :param lattice:     list of lattice sites returned by get_lattice
#     :type lattice:      list[(int, int, int)]
#     :param myL:         number of lattice sites in each direction
#     :type myL:          int
#     :param spin:        Optional; number of spin degrees of freedom
#     :type spinL:        int
#     :param isospin:     Optional; number of isospin degrees of freedom
#     :type isospin:      int
#     :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
#                         basis, and value is the value of the matrix element Tij
#     :rtype:             list[(int, int, float)]
#     """
#     mat = []
#     for site in lattice:
#         i = site[0]
#         j = site[1]
#         k = site[2]
#         for tz in range(isospin):
#             tzp = 1 - tz
#             val = 0.5
#             for sz in range(spin):
#                 state1 = [i, j, k, tz, sz]
#                 indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
#                 state2 = [i, j, k, tzp, sz]
#                 indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
#                 mat.append([indx2, indx1, val])
#     #
#     return mat


# def tau_y(lattice, myL, spin=2, isospin=2):
#     """
#     computes matrix elements for 1-body isospin-y operator.

#     :param lattice:     list of lattice sites returned by get_lattice
#     :type lattice:      list[(int, int, int)]
#     :param myL:         number of lattice sites in each direction
#     :type myL:          int
#     :param spin:        Optional; number of spin degrees of freedom
#     :type spinL:        int
#     :param isospin:     Optional; number of isospin degrees of freedom
#     :type isospin:      int
#     :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
#                         basis, and value is the value of the matrix element Tij
#     :rtype:             list[(int, int, complex)]
#     """
#     mat = []
#     for site in lattice:
#         i = site[0]
#         j = site[1]
#         k = site[2]
#         for tz in range(isospin):
#             sgn = np.sign(tz - 0.5)
#             tzp = 1 - tz
#             val = sgn * 0.5j
#             for sz in range(spin):
#                 state1 = [i, j, k, tz, sz]
#                 indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
#                 state2 = [i, j, k, tzp, sz]
#                 indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
#                 mat.append([indx2, indx1, val])
#     #
#     return mat


# def tau_z(lattice, myL, spin=2, isospin=2):
#     """
#     computes matrix elements for 1-body isospin-z operator.

#     :param lattice:     list of lattice sites returned by get_lattice
#     :type lattice:      list[(int, int, int)]
#     :param myL:         number of lattice sites in each direction
#     :type myL:          int
#     :param spin:        Optional; number of spin degrees of freedom
#     :type spinL:        int
#     :param isospin:     Optional; number of isospin degrees of freedom
#     :type isospin:      int
#     :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
#                         basis, and value is the value of the matrix element Tij
#     :rtype:             list[(int, int, float)]
#     """
#     mat = []
#     for site in lattice:
#         i = site[0]
#         j = site[1]
#         k = site[2]
#         for tz in range(isospin):
#             val = tz - 0.5
#             for sz in range(spin):
#                 state = [i, j, k, tz, sz]
#                 indx = lat.state2index(state, myL=myL, spin=spin, isospin=isospin)
#                 mat.append([indx, indx, val])
#     #
#     return mat

def tau_x(
    lattice,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    """Construct the one-body isospin-x operator."""
    return tau(
        lattice,
        myL,
        component="x",
        spin=spin,
        isospin=isospin,
    )


def tau_y(
    lattice,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    """Construct the one-body isospin-y operator."""
    return tau(
        lattice,
        myL,
        component="y",
        spin=spin,
        isospin=isospin,
    )


def tau_z(
    lattice,
    myL: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    """Construct the one-body isospin-z operator."""
    return tau(
        lattice,
        myL,
        component="z",
        spin=spin,
        isospin=isospin,
    )

def spin_x(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body spin-x operator.

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        for tz in range(isospin):
            for sz in range(spin):
                szp = 1 - sz
                val = 0.5
                state1 = [i, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                state2 = [i, j, k, tz, szp]
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx2, indx1, val])
    #
    return mat


def spin_y(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body spin-y operator.

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, complex)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        for tz in range(isospin):
            for sz in range(spin):
                sgn = np.sign(sz - 0.5)
                szp = 1 - sz
                val = sgn * 0.5j
                state1 = [i, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                state2 = [i, j, k, tz, szp]
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx2, indx1, val])
    #
    return mat


def spin_z(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body spin-z operator.

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        for tz in range(isospin):
            for sz in range(spin):
                val = sz - 0.5
                state = [i, j, k, tz, sz]
                indx = lat.state2index(state, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx, indx, val])
    #
    return mat


def pauli_spin_x(lattice, myL, spin=2, isospin=2):
    return [(p, q, 2 * x) for p, q, x in spin_x(lattice, myL, spin, isospin)]


def pauli_spin_y(lattice, myL, spin=2, isospin=2):
    return [(p, q, 2 * x) for p, q, x in spin_y(lattice, myL, spin, isospin)]


def pauli_spin_z(lattice, myL, spin=2, isospin=2):
    return [(p, q, 2 * x) for p, q, x in spin_z(lattice, myL, spin, isospin)]


def pauli_tau_x(
    lattice,
    L: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    return 2 * tau_x(lattice, L, spin, isospin)


def pauli_tau_y(
    lattice,
    L: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    return 2 * tau_y(lattice, L, spin, isospin)


def pauli_tau_z(
    lattice,
    L: int,
    spin: int = 2,
    isospin: int = 2,
) -> sparse.csr_array:
    return 2 * tau_z(lattice, L, spin, isospin)


def p_x(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_x. Really: -i times d_x

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        # hop to the right in x
        r = lat.right(i, myL=myL)  # r,j,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [r, j, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def p_y(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_y. Really: -i times d_y

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #
        # hop to the right in y
        r = lat.right(j, myL=myL)  # i,r,k
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, r, k, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def p_z(lattice, myL, spin=2, isospin=2):
    """
    computes matrix elements for 1-body momentum operator p_z. Really: -i times d_z

    :param lattice:     list of lattice sites returned by get_lattice
    :type lattice:      list[(int, int, int)]
    :param myL:         number of lattice sites in each direction
    :type myL:          int
    :param spin:        Optional; number of spin degrees of freedom
    :type spinL:        int
    :param isospin:     Optional; number of isospin degrees of freedom
    :type isospin:      int
    :return:            list of tuples [i, j, value] where i and j are indices in the single-particle
                        basis, and value is the value of the matrix element Tij
    :rtype:             list[(int, int, float)]
    """
    mat = []
    for site in lattice:
        i = site[0]
        j = site[1]
        k = site[2]
        #
        # hop to the right in z
        r = lat.right(k, myL=myL)  # i,j,r
        val = -0.5j
        for tz in range(isospin):
            for sz in range(spin):
                state1 = [i, j, k, tz, sz]
                state2 = [i, j, r, tz, sz]
                indx1 = lat.state2index(state1, myL=myL, spin=spin, isospin=isospin)
                indx2 = lat.state2index(state2, myL=myL, spin=spin, isospin=isospin)
                mat.append([indx1, indx2, val])
                mat.append(
                    [indx2, indx1, -val]
                )  # adds a hop-to-the left matrix element
    #
    return mat


def change_lat_1body(inter, origL, newL, spin=2, isospin=2):
    """
    Changes a one-body interaction in list format for a given L to a new L

    :param inter:   interaction stored as a list of lists [a,b,v]
                    where a and b are indices and v is the value
                    for V^a_b
    :type inter:    list[(int, int, float)]
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
    :rtype:         list[(int, int, float)]
    """
    new_inter = [[] for _ in range(len(inter))]
    for i in range(len(inter)):
        a, b, val = inter[i]
        lst = []
        ind_lst = [a, b]
        for ind in ind_lst:
            lst.append(
                lat.state2index(
                    lat.index2state(ind, origL, spin, isospin), newL, spin, isospin
                )
            )
        lst.append(val)
        new_inter[i] = lst
    return new_inter


def get_smeared_dens(
    lattice, myL, sL, sNL, op1b=None, spin=2, isospin=2, verbose=False, sites=None
):
    """
    Gets the smeared density

    :param lattice: list of lattice sites returned by lattice.get_lattice
    :type lattice:  list[(int, int, int)]
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :param sNL:     non-local smearing strength
    :type sNL:      float
    :param op1b:    Optional; one body operator used to generate rho in the form
                    a^dagger [op1b] a. If None, then the identity operator is used
    :type op1b:     scipy.sparse.csr_array()
    :param spin:    Optional; number of spin degrees of freedom
    :type spin:     int
    :param isospin: Optional; number of isospin degrees of freedom
    :type isospin:  int
    :param verbose: Optional; whether or not to print progress during calculation
    :type verbose:  bool
    :param sites:   Optional; Give default value or None in order to compute the interaction at
                    all sites, or give a list of sites in the format [i, j, k] to only compute it
                    at the given sites
    :type sites:    list[int,int,int]
    :returns:       smeared density as list of densities at a site
    :rtype:         list[scipy.sparse.coo_array()]
    """
    rho_n = []
    if verbose:
        print("Generating Densities...", end="")
    for site1 in lattice:
        rho_n.append(rho_op(site1, myL, op1b=op1b, sNL=sNL, spin=spin, isospin=isospin))
    if verbose:
        print("Done")
    dim = myL**3 * spin * isospin
    if sites is not None:
        site1Loop = sites
        if sL == 0:
            rho_ns = []
            for loc in sites:
                pos = loc[0] * myL**2 + loc[1] * myL + loc[2]
                rho_ns.append(rho_n[pos])
            rho_n = rho_ns
    else:
        site1Loop = lattice
    if sL != 0:
        if verbose:
            print("Performing Local Smearing...", end="")
        rho_smeared = []
        for site1 in site1Loop:
            tmp = sparse.csr_array(np.zeros([dim, dim]))
            for site2 in lattice:
                pos = site2[0] * myL**2 + site2[1] * myL + site2[2]
                scale = smear_local(site1, site2, myL, sL)
                if scale != 0:
                    tmp += rho_n[pos] * scale
            rho_smeared.append(tmp.tocoo())
        if verbose:
            print("Done")
    else:
        rho_smeared = rho_n
    return rho_smeared


def smear_local(site1, site2, myL, sL):
    """
    Calculates the local smearing function at two points with strength sL

    :param site1:   first site on the lattice
    :type site1:    (int, int, int)
    :param site2:   second site on the lattice
    :type site2:    (int, int, int)
    :param myL:     number of lattice sites in each direction
    :type myL:      int
    :param sL:      local smearing strength
    :type sL:       float
    :return:        value of the local smearing function
    :rtype:         float
    """
    if site1 == site2:
        return 1
    i1, j1, k1 = site1
    i2, j2, k2 = site2
    dist_sq = (
        ((i1 - i2 + myL // 2) % myL - myL // 2) ** 2
        + ((j1 - j2 + myL // 2) % myL - myL // 2) ** 2
        + ((k1 - k2 + myL // 2) % myL - myL // 2) ** 2
    )
    if dist_sq == 1:
        return sL
    return 0
