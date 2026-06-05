# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module defining normal ordering utilities for the IMSRG."""

__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

from NuLattice.utils._jax_types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator

from functools import partial
# import numpy as np
from typing import Optional
import jax
import jax.numpy as jnp

@jax.jit
def create_occupations_nstates(n_states, ref_indices):
    """
    Creates occupation array from a list of occupied state indices.

    :param n_stat: Total number of single-particle states (int)
    :param ref_indices: Array of indices that are occupied (jnp.ndarray)
    """
    occs = jnp.zeros(n_states, dtype=jnp.float64)
    for idx in ref_indices:
        occs[idx] = 1.0
    return occs


def create_occupations(basis, ref_indices):
    """
    Creates occupation array from a list of occupied state indices.

    :param n_stat: Total number of single-particle states (int)
    :param ref_indices: Array of indices that are occupied (jnp.ndarray)
    """
    occs = jnp.zeros(shape=len(basis))
    for idx in ref_indices:
        i = basis.index(idx)
        occs = occs.at[i].set(1.0)
    return occs



@partial(jax.jit, static_argnames=("dim", ))
def _compute_op1_kernel(h1, dim, occs, e0_arr, f):
    for p in range(dim):
        for q in range(dim):
            val = h1[p, q]
            f = f.at[p, q].add(val)
            if p == q:
                e0_arr = e0_arr.at[0].add(occs[p] * val)
    return e0_arr, f


def _accumulate_2b_contributions(p, q, r, s, val, occs, e0_arr, f, gamma):
    """
    Helper to accumulate a single permuted 2-body element into normal ordered operators.
    """
    # 2-Body (Gamma) Contribution
    gamma = gamma.at[p, q, r, s].add(val)

    # 1-Body (Fock) Contribution: Contraction over q=s
    q_s_mask = (q == s)
    term = jnp.where(q_s_mask, occs[q] * val, 0.0)
    f = f.at[p, r].add(term)

    p_r_mask = q_s_mask & (p == r)
    e0_term = jnp.where(p_r_mask, 0.5 * occs[p] * term, 0.0)
    e0_arr = e0_arr.at[0].add(jnp.sum(e0_term))

    return e0_arr, f, gamma


@jax.jit
def _compute_op2_kernel(indices, values, occs, e0_arr, f, gamma):
    """
    Numba kernel to process 2-body interactions in SoA format.
    Applies 4 antisymmetrization permutations on-the-fly.
    """
    p = indices[:, 0]
    q = indices[:, 1]
    r = indices[:, 2]
    s = indices[:, 3]


    # Apply antisymmetrization: <pq|rs>
    e0_arr, f, gamma = _accumulate_2b_contributions(p, q, r, s, values, occs, e0_arr, f, gamma)
    e0_arr, f, gamma = _accumulate_2b_contributions(q, p, r, s, -values, occs, e0_arr, f, gamma)
    e0_arr, f, gamma = _accumulate_2b_contributions(p, q, s, r, -values, occs, e0_arr, f, gamma)
    e0_arr, f, gamma = _accumulate_2b_contributions(q, p, s, r, values, occs, e0_arr, f, gamma)

    return e0_arr, f, gamma


@jax.jit
def _compute_op3_kernel(indices, values, occs, e0_arr, f, gamma):
    """
    Vectorizes over the data length and unrolls the 36 antisymmetrization
    permutations using vectorized masking.
    """
    perm_map = [
        [0, 1, 2], [2, 0, 1], [1, 2, 0],  # Even (+1)
        [1, 0, 2], [0, 2, 1], [2, 1, 0],  # Odd (-1)
    ]
    perm_signs = [1.0, 1.0, 1.0, -1.0, -1.0, -1.0]

    p_raw = indices[:, 0]
    q_raw = indices[:, 1]
    r_raw = indices[:, 2]
    s_raw = indices[:, 3]
    t_raw = indices[:, 4]
    u_raw = indices[:, 5]

    bra_raw = [p_raw, q_raw, r_raw]
    ket_raw = [s_raw, t_raw, u_raw]

    for bi in range(6):
        p = bra_raw[perm_map[bi][0]]
        q = bra_raw[perm_map[bi][1]]
        r = bra_raw[perm_map[bi][2]]
        sign_bra = perm_signs[bi]

        for ki in range(6):
            s = ket_raw[perm_map[ki][0]]
            t = ket_raw[perm_map[ki][1]]
            u = ket_raw[perm_map[ki][2]]
            sign_ket = perm_signs[ki]

            val = values * (sign_bra * sign_ket)


            # Effective 2-Body
            mask_gamma = (r == u)
            gamma_increment = jnp.where(mask_gamma, val, 0.0)
            gamma = gamma.at[p, q, s, t].add(gamma_increment)

            # Effective 1-Body
            mask_f = mask_gamma & (q == t)
            term = 0.5 * occs[q] * occs[r] * val
            f_increment = jnp.where(mask_f, term, 0.0)
            f = f.at[p, s].add(f_increment)

            # Scalar Energy
            mask_e0 = mask_f & (p == s)
            e0_increment = jnp.where(mask_e0, (1.0 / 3.0) * occs[p] * term, 0.0)
            
            e0_arr = e0_arr.at[0].add(jnp.sum(e0_increment))

    return e0_arr, f, gamma


def compute_normal_ordered_hamiltonian_no2b(
    occs: jnp.ndarray,
    op1: OneBodyOperator | jnp.ndarray,
    op2: TwoBodyOperator,
    op3: Optional[ThreeBodyOperator] = None,
):
    """
    Computes the normal-ordered Hamiltonian with respect to a reference state

    Transforms the Hamiltonian to normal-ordered form by summing over occupied states,
    yielding the reference-state energy, effective one-body (Fock) operator, and effective two-body interactions

    :param occs:    Occupation numbers (dim, )
    :type occs:     numpy bool array
    :return:        Reference state energy, normal-ordered one-body operator (Fock matrix), normal-ordered two-body operator
    :rtype:         float, numpy array, numpy array
    """
    dim = len(occs)
    e0 = jnp.zeros(1, dtype=jnp.float64)
    f = jnp.zeros((dim, dim), dtype=jnp.float64)
    gamma = jnp.zeros((dim, dim, dim, dim), dtype=jnp.float64)

    h1 = op1.to_dense() if hasattr(op1, "to_dense") else op1
    e0, f = _compute_op1_kernel(h1, dim, occs, e0, f)

    e0, f, gamma = _compute_op2_kernel(op2.indices, op2.values, occs, e0, f, gamma)

    if op3 is not None:
        e0, f, gamma = _compute_op3_kernel(op3.indices, op3.values, occs, e0, f, gamma)

    return e0[0], f, gamma
