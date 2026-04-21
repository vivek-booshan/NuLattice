import math
import numpy as np

import NuLattice.utils.constants as consts
from NuLattice.utils._types import OneBodyOperator

def _get_sp_basis(L: int, spin: int = 2, isospin: int = 2) -> np.ndarray:
    lattice_sites = np.mgrid[0:L, 0:L, 0:L, 0:isospin, 0:spin]
    return lattice_sites.reshape(5, -1).T

def Tkin(
    lattice_sites, L: int, Nk: int, a_lat: float, spin: int = 2, isospin: int = 2
):
    k_stride = isospin * spin
    j_stride = L * k_stride
    i_stride = L * j_stride

    basis = _get_sp_basis(L, spin, isospin)
    nstat = len(basis)
    indices = np.arange(nstat, dtype=np.int64)

    h = -1.0 / (2.0 * consts.MASS * a_lat)
    cf0 = 0.0

    all_indices = []
    all_values = []

    for k in range(1, Nk + 1):
        weight = ((-1)**(k + 1) * 2.0 *
                  (math.factorial(Nk) / math.factorial(Nk - k)) /
                  (math.factorial(Nk + k) / math.factorial(Nk)) / (k**2) * h)
        
        val = weight / a_lat
        cf0 -= 2 * weight

        for dim in range(3):
            neighbor_right = basis.copy()
            neighbor_right[:, dim] = (neighbor_right[:, dim] + k) % L
            
            idx_right = (
                neighbor_right[:, 0] * i_stride
                + neighbor_right[:, 1] * j_stride
                + neighbor_right[:, 2] * k_stride
                + neighbor_right[:, 3] * spin
                + neighbor_right[:, 4]
            )
            
            all_indices.append(np.column_stack([indices, idx_right]))
            all_values.append(np.full(nstat, val, dtype=np.float64))

            neighbor_left = basis.copy()
            neighbor_left[:, dim] = (neighbor_left[:, dim] - k) % L
            
            idx_left = (
                neighbor_left[:, 0] * i_stride
                + neighbor_left[:, 1] * j_stride
                + neighbor_left[:, 2] * k_stride
                + neighbor_left[:, 3] * spin
                + neighbor_left[:, 4]
            )
            
            all_indices.append(np.column_stack([indices, idx_left]))
            all_values.append(np.full(nstat, val, dtype=np.float64))

    cf0 *= 3
    diag_val = cf0 / a_lat
    
    all_indices.append(np.column_stack([indices, indices]))
    all_values.append(np.full(nstat, diag_val, dtype=np.float64))

    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    return OneBodyOperator(final_indices, final_values, nstat)

def tau_x(L: int, spin: int = 2, isospin: int = 2):
    assert isospin == 2, "Pauli operators require isospin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    # Extract current isospin state (0 or 1)
    tz = (indices // spin) % isospin
    
    # Flip the state: if tz=0 -> +spin, if tz=1 -> -spin
    flipped_indices = indices + (1 - 2 * tz) * spin

    # Original code returns [flipped_index, original_index, value]
    final_indices = np.column_stack([flipped_indices, indices])
    final_values = np.full(nstat, 0.5, dtype=np.float64)

    return OneBodyOperator(final_indices, final_values, nstat)


def tau_y(L: int, spin: int = 2, isospin: int = 2):
    assert isospin == 2, "Pauli operators require isospin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    tz = (indices // spin) % isospin
    flipped_indices = indices + (1 - 2 * tz) * spin

    # tz=0 -> -0.5j, tz=1 -> 0.5j
    final_values = np.sign(tz - 0.5) * 0.5j
    final_indices = np.column_stack([flipped_indices, indices])

    return OneBodyOperator(final_indices, final_values, nstat)


def tau_z(L: int, spin: int = 2, isospin: int = 2):
    assert isospin == 2, "Pauli operators require isospin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    tz = (indices // spin) % isospin
    
    # Diagonal operator: no flipping required
    final_values = (tz - 0.5).astype(np.float64)
    final_indices = np.column_stack([indices, indices])

    return OneBodyOperator(final_indices, final_values, nstat)


def spin_x(L: int, spin: int = 2, isospin: int = 2):
    assert spin == 2, "Pauli operators require spin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    # Extract current spin state
    sz = indices % spin
    
    # Flip the state: if sz=0 -> +1, if sz=1 -> -1
    flipped_indices = indices + (1 - 2 * sz)

    final_indices = np.column_stack([flipped_indices, indices])
    final_values = np.full(nstat, 0.5, dtype=np.float64)

    return OneBodyOperator(final_indices, final_values, nstat)


def spin_y(L: int, spin: int = 2, isospin: int = 2):
    assert spin == 2, "Pauli operators require spin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    sz = indices % spin
    flipped_indices = indices + (1 - 2 * sz)

    # sz=0 -> -0.5j, sz=1 -> 0.5j
    final_values = np.sign(sz - 0.5) * 0.5j
    final_indices = np.column_stack([flipped_indices, indices])

    return OneBodyOperator(final_indices, final_values, nstat)


def spin_z(L: int, spin: int = 2, isospin: int = 2):
    """Vectorized 1-body spin-z operator."""
    assert spin == 2, "Pauli operators require spin=2"
    nstat = (L**3) * spin * isospin
    indices = np.arange(nstat, dtype=np.int64)

    sz = indices % spin
    
    final_values = (sz - 0.5).astype(np.float64)
    final_indices = np.column_stack([indices, indices])

    return OneBodyOperator(final_indices, final_values, nstat)
