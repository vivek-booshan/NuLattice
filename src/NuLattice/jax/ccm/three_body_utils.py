import jax
import jax.numpy as jnp
from typing import List, Tuple, Union
from NuLattice.utils._jax_types import ThreeBodyOperator, TwoBodyOperator

ThreeBodyList = List[List[Union[int, float]]]


def get_3NF(
    part: List[int],
    hole: List[int],
    my3body: ThreeBodyList,
) -> Tuple[ThreeBodyOperator, ...]:
    """
    Sorts raw three-body matrix elements into 9 blocks using JAX vectorized ops.
    Note: This function returns dynamically sized arrays based on the data,
    so it cannot be JIT-compiled as a whole, but it will run very fast via JAX dispatch.
    """
    nstat = len(part) + len(hole)

    if not my3body:
        empty_idx = jnp.empty((0, 6), dtype=jnp.int32)
        empty_val = jnp.empty((0,), dtype=jnp.float64)
        return tuple(ThreeBodyOperator(empty_idx, empty_val, nstat) for _ in range(9))

    data_tensor = jnp.array(my3body, dtype=jnp.float64)
    indices = data_tensor[:, :6].astype(jnp.int32)  # (N, 6)
    values = data_tensor[:, 6]  # (N,)

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

        n_perms_k = len(ks)
        n_perms_b = len(bs)

        expanded_ket = base_ket[:, kp]
        expanded_bra = base_bra[:, bp]

        comb_signs = ks[:, None] * bs[None, :]
        expanded_vals = base_vals[:, None, None] * comb_signs[None, :, :]

        M = base_ket.shape[0]
        final_ket = jnp.broadcast_to(
            jnp.expand_dims(expanded_ket, 2), (M, n_perms_k, n_perms_b, 3)
        ).reshape(-1, 3)
        final_bra = jnp.broadcast_to(
            jnp.expand_dims(expanded_bra, 1), (M, n_perms_k, n_perms_b, 3)
        ).reshape(-1, 3)
        final_vals = expanded_vals.reshape(-1)

        local_ket = local_map[final_ket]
        local_bra = local_map[final_bra]

        final_indices = jnp.concatenate([local_ket, local_bra], axis=1)
        results.append(ThreeBodyOperator(final_indices, final_vals, nstat))

    return tuple(results)


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


def _fock_accumulator(target, indices, values, row_col_map):
    """Internal JIT-compiled accumulator for 1-body normal ordering"""
    if values.size == 0:
        return target

    mask = (indices[:, 1] == indices[:, 4]) & (indices[:, 2] == indices[:, 5])

    row_idx = indices[:, row_col_map[0]]
    col_idx = indices[:, row_col_map[1]]

    # JAX `.at` automatically handles repeated indices, removing the need for flat strides
    masked_vals = jnp.where(mask, 0.5 * values, 0.0)
    return target.at[row_idx, col_idx].add(masked_vals)


def get_3NF_fock(
    hnum: int,
    pnum: int,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    f_pp = jnp.zeros((pnum, pnum), dtype=jnp.float64)
    f_ph = jnp.zeros((pnum, hnum), dtype=jnp.float64)
    f_hh = jnp.zeros((hnum, hnum), dtype=jnp.float64)

    f_pp = _fock_accumulator(f_pp, w_phh_phh.indices, w_phh_phh.values, (0, 3))
    f_ph = _fock_accumulator(f_ph, w_phh_hhh.indices, w_phh_hhh.values, (0, 3))
    f_hh = _fock_accumulator(f_hh, w_hhh_hhh.indices, w_hhh_hhh.values, (0, 3))

    return f_pp, f_ph, f_hh


def _dense_tbme_accumulator(target, indices, values, dim_map):
    """Internal JIT-compiled accumulator for 2-body dense tensor construction"""
    if values.size == 0:
        return target

    mask = indices[:, 2] == indices[:, 5]
    masked_vals = jnp.where(mask, values, 0.0)

    idx_0 = indices[:, dim_map[0]]
    idx_1 = indices[:, dim_map[1]]
    idx_2 = indices[:, dim_map[2]]
    idx_3 = indices[:, dim_map[3]]

    # 4D scatter add without any stride calculations!
    return target.at[idx_0, idx_1, idx_2, idx_3].add(masked_vals)


def get_3NF_tbme(
    w_pph_pph: ThreeBodyOperator,
    w_pph_phh: ThreeBodyOperator,
    w_pph_hhh: ThreeBodyOperator,
    w_phh_phh: ThreeBodyOperator,
    w_phh_hhh: ThreeBodyOperator,
    w_hhh_hhh: ThreeBodyOperator,
    pnum: int,
    hnum: int,
    sparse_pppp: bool = True,
    sparse_ppph: bool = True,
) -> Tuple[Union[TwoBodyOperator, jnp.ndarray], ...]:
    nstat = pnum + hnum

    v_pphh = jnp.zeros((pnum, pnum, hnum, hnum), dtype=jnp.float64)
    v_phph = jnp.zeros((pnum, hnum, pnum, hnum), dtype=jnp.float64)
    v_phhh = jnp.zeros((pnum, hnum, hnum, hnum), dtype=jnp.float64)
    v_hhhh = jnp.zeros((hnum, hnum, hnum, hnum), dtype=jnp.float64)

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

    if sparse_pppp:
        v_pppp = get_sparse(w_pph_pph, (0, 1, 3, 4))
    else:
        v_pppp = jnp.zeros((pnum, pnum, pnum, pnum), dtype=jnp.float64)
        v_pppp = _dense_tbme_accumulator(
            v_pppp, w_pph_pph.indices, w_pph_pph.values, (0, 1, 3, 4)
        )

    if sparse_ppph:
        v_ppph = get_sparse(w_pph_phh, (0, 1, 3, 4))
    else:
        v_ppph = jnp.zeros((pnum, pnum, pnum, hnum), dtype=jnp.float64)
        v_ppph = _dense_tbme_accumulator(
            v_ppph, w_pph_phh.indices, w_pph_phh.values, (0, 1, 3, 4)
        )

    v_pphh = _dense_tbme_accumulator(
        v_pphh, w_pph_hhh.indices, w_pph_hhh.values, (0, 1, 3, 4)
    )
    v_phph = _dense_tbme_accumulator(
        v_phph, w_phh_phh.indices, w_phh_phh.values, (0, 1, 3, 4)
    )
    v_phhh = _dense_tbme_accumulator(
        v_phhh, w_phh_hhh.indices, w_phh_hhh.values, (0, 1, 3, 4)
    )
    v_hhhh = _dense_tbme_accumulator(
        v_hhhh, w_hhh_hhh.indices, w_hhh_hhh.values, (0, 1, 3, 4)
    )

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh
