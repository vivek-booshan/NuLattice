from functools import partial
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from NuLattice.utils._jax_types import ThreeBodyOperator, TwoBodyOperator

# TODO: avoid jnp.zeros eager allocation

def get_3NF(
    part: List[int],
    hole: List[int],
    op3: ThreeBodyOperator,
) -> Tuple[ThreeBodyOperator, ...]:
    """
    Sorts raw three-body matrix elements into 9 blocks using JAX vectorized ops.
    Note: This function returns dynamically sized arrays based on the data,
    so it cannot be JIT-compiled as a whole, but it will run very fast via JAX dispatch.
    """
    nstat = len(part) + len(hole)

    indices, values = op3.indices, op3.values

    max_idx = int(jnp.max(indices))

    # 0=hole, 1=particle, -1=invalid
    type_map = jnp.full((max_idx + 1,), -1, dtype=jnp.int32)
    local_map = jnp.full((max_idx + 1,), -1, dtype=jnp.int32)

    h_tens = jnp.array(hole, dtype=jnp.int32)
    p_tens = jnp.array(part, dtype=jnp.int32)

    type_map = type_map.at[h_tens].set(0)
    type_map = type_map.at[p_tens].set(1)

    local_map = local_map.at[h_tens].set(jnp.arange(len(hole), dtype=jnp.int32))
    local_map = local_map.at[p_tens].set(jnp.arange(len(part), dtype=jnp.int32))

    types = type_map[indices]  # (N, 6)

    ket_types = types[:, :3]
    bra_types = types[:, 3:]

    ket_indices = indices[:, :3]
    bra_indices = indices[:, 3:]

    @jax.jit
    def vectorize_order(current_types, current_indices):
        """Reorders indices to (p...p h...h) format and computes sign flips.
        Rewritten to use jnp.where to avoid dynamic boolean masking."""
        sums = jnp.sum(current_types, axis=1)
        new_indices = current_indices
        signs = jnp.ones(current_indices.shape[0], dtype=jnp.float64)

        # Case: pph (sum=2)
        is_php = (sums == 2) & (current_types[:, 1] == 0)
        new_indices = jnp.where(
            is_php[:, None], current_indices[:, [0, 2, 1]], new_indices
        )
        signs = jnp.where(is_php, -1.0, signs)

        is_hpp = (sums == 2) & (current_types[:, 0] == 0)
        new_indices = jnp.where(
            is_hpp[:, None], current_indices[:, [1, 2, 0]], new_indices
        )
        signs = jnp.where(is_hpp, -1.0, signs)

        # Case: phh (sum=1)
        is_hph = (sums == 1) & (current_types[:, 1] == 1)
        new_indices = jnp.where(
            is_hph[:, None], current_indices[:, [1, 0, 2]], new_indices
        )
        signs = jnp.where(is_hph, -1.0, signs)

        is_hhp = (sums == 1) & (current_types[:, 2] == 1)
        new_indices = jnp.where(
            is_hhp[:, None], current_indices[:, [2, 0, 1]], new_indices
        )
        signs = jnp.where(is_hhp, -1.0, signs)

        return new_indices, signs, sums

    ket_canon, ket_signs, ket_sums = vectorize_order(ket_types, ket_indices)
    bra_canon, bra_signs, bra_sums = vectorize_order(bra_types, bra_indices)

    bucket_defs = [
        (3, 2),
        (3, 1),
        (2, 2),
        (3, 0),
        (2, 1),
        (2, 0),
        (1, 1),
        (1, 0),
        (0, 0),
    ]

    perms_lookup = {
        3: (
            jnp.array(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]]
            ),
            jnp.array([1, -1, -1, -1, 1, 1], dtype=jnp.float64),
        ),
        0: (
            jnp.array(
                [[0, 1, 2], [1, 0, 2], [2, 1, 0], [0, 2, 1], [1, 2, 0], [2, 0, 1]]
            ),
            jnp.array([1, -1, -1, -1, 1, 1], dtype=jnp.float64),
        ),
        2: (jnp.array([[0, 1, 2], [1, 0, 2]]), jnp.array([1, -1], dtype=jnp.float64)),
        1: (jnp.array([[0, 1, 2], [0, 2, 1]]), jnp.array([1, -1], dtype=jnp.float64)),
    }

    results = []
    for k_s, b_s in bucket_defs:
        mask = (ket_sums == k_s) & (bra_sums == b_s)

        if not jnp.any(mask):
            empty_idx = jnp.empty((0, 6), dtype=jnp.int32)
            empty_val = jnp.empty((0,), dtype=jnp.float64)
            results.append(ThreeBodyOperator(empty_idx, empty_val, nstat))
            continue

        base_ket = ket_canon[mask]
        base_bra = bra_canon[mask]
        base_vals = values[mask] * ket_signs[mask] * bra_signs[mask]

        kp, ks = perms_lookup[k_s]
        bp, bs = perms_lookup[b_s]

        final_ket, final_bra, final_vals = expand_permutations_kernel(base_ket, base_bra, base_vals, kp, ks, bp, bs)

        local_ket = local_map[final_ket]
        local_bra = local_map[final_bra]

        final_indices = jnp.concatenate([local_ket, local_bra], axis=1)
        results.append(ThreeBodyOperator(final_indices, final_vals, nstat))

    return tuple(results)


@jax.jit
def _eref_kernel(indices, values):
    """Internal JIT-compiled kernel for Eref"""
    mask = (
        (indices[:, 0] == indices[:, 3])
        & (indices[:, 1] == indices[:, 4])
        & (indices[:, 2] == indices[:, 5])
    )
    # Jnp.where allows JAX to keep a static shape while summing only valid entries
    return jnp.sum(jnp.where(mask, values, 0.0)) / 6.0


def get_3NF_Eref(w_hhh_hhh: ThreeBodyOperator) -> float:
    if len(w_hhh_hhh.values) == 0:
        return 0.0
    return float(_eref_kernel(w_hhh_hhh.indices, w_hhh_hhh.values))


# @partial(jax.jit, static_argnums=(3,))
def _fock_accumulator(target, indices, values, row_col_map):
    """Internal JIT-compiled accumulator for 1-body normal ordering"""
    if values.size == 0:
        return target

    mask = (indices[:, 1] == indices[:, 4]) & (indices[:, 2] == indices[:, 5])

    row_idx = indices[:, row_col_map[0]]
    col_idx = indices[:, row_col_map[1]]

    # JAX `.at` automatically handles repeated indices, removing the need for flat strides
    masked_vals = np.where(mask, 0.5 * values, 0.0)
    np.add.at(target, (row_idx, col_idx), masked_vals)
    return target
    # return target.at[row_idx, col_idx].add(masked_vals)


def get_3NF_fock(
    hnum: int,
    pnum: int,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    f_pp = np.zeros((pnum, pnum), dtype=np.float64)
    f_ph = np.zeros((pnum, hnum), dtype=np.float64)
    f_hh = np.zeros((hnum, hnum), dtype=np.float64)

    f_pp = _fock_accumulator(f_pp, w_phh_phh.indices, w_phh_phh.values, (0, 3))
    f_ph = _fock_accumulator(f_ph, w_phh_hhh.indices, w_phh_hhh.values, (0, 3))
    f_hh = _fock_accumulator(f_hh, w_hhh_hhh.indices, w_hhh_hhh.values, (0, 3))

    return f_pp, f_ph, f_hh


# @partial(jax.jit, static_argnums=(3,))
def _dense_tbme_accumulator(target, indices, values, dim_map):
    """Internal JIT-compiled accumulator for 2-body dense tensor construction"""
    if values.size == 0:
        return target

    mask = indices[:, 2] == indices[:, 5]
    masked_vals = np.where(mask, values, 0.0)

    idx_0 = indices[:, dim_map[0]]
    idx_1 = indices[:, dim_map[1]]
    idx_2 = indices[:, dim_map[2]]
    idx_3 = indices[:, dim_map[3]]

    # 4D scatter add without any stride calculations!
    np.add.at(target, (idx_0, idx_1, idx_2, idx_3), masked_vals)
    return target
    # return target.at[idx_0, idx_1, idx_2, idx_3].add(masked_vals)


def get_3NF_tbme(
    w_pph_pph: ThreeBodyOperator,
    w_pph_phh: ThreeBodyOperator,
    w_pph_hhh: ThreeBodyOperator,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
    pnum: int,
    hnum: int,
) -> Tuple[Union[TwoBodyOperator, np.ndarray], ...]:
    nstat = pnum + hnum

    v_pphh = np.zeros((pnum, pnum, hnum, hnum), dtype=np.float64)
    v_phph = np.zeros((pnum, hnum, pnum, hnum), dtype=np.float64)
    v_phhh = np.zeros((pnum, hnum, hnum, hnum), dtype=np.float64)
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum), dtype=np.float64)

    def get_sparse(op: ThreeBodyOperator, dim_map) -> TwoBodyOperator:
        # We cannot JIT this because boolean masking changes the shape
        indices = op.indices
        values = op.values
        if len(values) == 0:
            return TwoBodyOperator(
                jnp.empty((0, 4), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.float64),
                nstat,
            )

        mask = indices[:, 2] == indices[:, 5]
        valid_idx = indices[mask]
        valid_val = values[mask]

        new_indices = valid_idx[:, list(dim_map)]
        return TwoBodyOperator(new_indices, valid_val, nstat)

    v_pppp = get_sparse(w_pph_pph, (0, 1, 3, 4))

    v_ppph = get_sparse(w_pph_phh, (0, 1, 3, 4))

    v_pphh = _dense_tbme_accumulator(
        v_pphh, np.asarray(w_pph_hhh.indices), np.asarray(w_pph_hhh.values), (0, 1, 3, 4)
    )
    v_phph = _dense_tbme_accumulator(
        v_phph, np.asarray(w_phh_phh.indices), np.asarray(w_phh_phh.values), (0, 1, 3, 4)
    )
    v_phhh = _dense_tbme_accumulator(
        v_phhh, np.asarray(w_phh_hhh.indices), np.asarray(w_phh_hhh.values), (0, 1, 3, 4)
    )
    v_hhhh = _dense_tbme_accumulator(
        v_hhhh, np.asarray(w_hhh_hhh.indices), np.asarray(w_hhh_hhh.values), (0, 1, 3, 4)
    )

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh

@jax.jit
def expand_permutations_kernel(
    base_ket: jax.Array, 
    base_bra: jax.Array, 
    base_vals: jax.Array, 
    kp: jax.Array, 
    ks: jax.Array, 
    bp: jax.Array, 
    bs: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    JIT-compiled expansion of 3-body permutations with type annotations.
    
    Args:
        base_ket: (M, 3) canonical ket indices
        base_bra: (M, 3) canonical bra indices
        base_vals: (M,) canonical values
        kp, bp: Permutation index arrays (e.g., shape (6, 3))
        ks, bs: Permutation sign arrays (e.g., shape (6,))
        
    Returns:
        A tuple of (final_ket, final_bra, final_vals)
    """
    n_perms_k: int = ks.shape[0]
    n_perms_b: int = bs.shape[0]
    M: int = base_ket.shape[0]

    expanded_ket: jax.Array = base_ket[:, kp]  # (M, n_perms_k, 3)
    expanded_bra: jax.Array = base_bra[:, bp]  # (M, n_perms_b, 3)

    # (n_perms_k, n_perms_b)
    comb_signs: jax.Array = ks[:, None] * bs[None, :]

    # (M, n_perms_k, n_perms_b)
    expanded_vals: jax.Array = base_vals[:, None, None] * comb_signs[None, :, :]

    # (NNZ, 3)
    final_ket: jax.Array = jnp.broadcast_to(
        expanded_ket[:, :, None, :], (M, n_perms_k, n_perms_b, 3)
    ).reshape(-1, 3)
    
    final_bra: jax.Array = jnp.broadcast_to(
        expanded_bra[:, None, :, :], (M, n_perms_k, n_perms_b, 3)
    ).reshape(-1, 3)
    
    final_vals: jax.Array = expanded_vals.reshape(-1)

    return final_ket, final_bra, final_vals
