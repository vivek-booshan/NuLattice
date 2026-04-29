import jax.numpy as jnp
import numpy as np

from NuLattice.utils._jax_types import OneBodyOperator, TwoBodyOperator, ThreeBodyOperator

from . import three_body_utils as tbu

# duplicated cuz lazy
def get_ref_energy(no_1b_hh, no_2b_hhhh, w_hhh_hhh=None):
    en = 0.0
    hnum = len(no_1b_hh)
    for i in range(hnum):
        en += no_1b_hh[(i, i)]
        for j in range(hnum):
            en -= 0.5 * no_2b_hhhh[(i, j, i, j)]

    if w_hhh_hhh is not None:
        if not isinstance(w_hhh_hhh, ThreeBodyOperator):
            raise ValueError(
                f"w_hhh_hhh must be ThreeBodyOperator, but got {type(w_hhh_hhh)}"
            )
        en += tbu.get_3NF_Eref(w_hhh_hhh)
    return float(en)

def normal_order_tensors_from_stamper(
    stamper,
    ref_state: list,
    NO2B=True,
    dtype=np.float64,
):
    L = stamper.L
    spin, isospin = stamper.spin, stamper.isospin
    stamp_1b, stamp_2b, stamp_3b = stamper.one_body, stamper.two_body, stamper.three_body
    num_local_states = spin * isospin
    nstat = (L**3) * num_local_states

    if stamper.pmask is None or stamper.hmask is None:
        stamper.normal_order_masks(ref_state)
    mask_P_jnp, mask_H_jnp = stamper.pmask, stamper.hmask

    vacEn_pure = stamper.get_reference_energy()

    # NOTE: need to clean up jax/numpy usage
    mask_P, mask_H = np.array(mask_P_jnp), np.array(mask_H_jnp)
    part_idx = np.where(mask_P == 1.0)[0]
    hole_idx = np.where(mask_H == 1.0)[0]
    pnum, hnum = len(part_idx), len(hole_idx)

    local_map = np.zeros(nstat, dtype=np.int64)
    local_map[part_idx] = np.arange(pnum)
    local_map[hole_idx] = np.arange(hnum)

    H_dense = np.zeros((nstat, nstat), dtype=dtype)
    for d, W in zip(*stamp_1b):
        nz = np.where(W != 0)
        for a, b in zip(*nz):
            idx = stamper.get_global_indices(d, [a, b])
            np.add.at(H_dense, (idx[:, 0], idx[:, 1]), W[a, b])

    f_pp = H_dense[np.ix_(part_idx, part_idx)]
    f_ph = H_dense[np.ix_(part_idx, hole_idx)]
    f_hh = H_dense[np.ix_(hole_idx, hole_idx)]

    v_pphh = np.zeros((pnum, pnum, hnum, hnum), dtype=dtype)
    v_phph = np.zeros((pnum, hnum, pnum, hnum), dtype=dtype)
    v_phhh = np.zeros((pnum, hnum, hnum, hnum), dtype=dtype)
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum), dtype=dtype)
    idx_pppp, val_pppp, idx_ppph, val_ppph = [], [], [], []

    for d, W in zip(*stamp_2b):
        nz = np.where(~np.isnan(W))
        for p, q, r, s in zip(*nz):
            val = W[p, q, r, s]
            idx = stamper.get_global_indices(d, [p, q, r, s])

            # Binary scoring based on particle/hole status
            P = mask_P[idx].astype(np.int64)
            scores = P[:, 0] * 8 + P[:, 1] * 4 + P[:, 2] * 2 + P[:, 3] * 1

            # Map global indices to local p/h indices
            mp, mq, mr, ms = (
                local_map[idx[:, 0]],
                local_map[idx[:, 1]],
                local_map[idx[:, 2]],
                local_map[idx[:, 3]],
            )

            for target, arr, mode in [
                (12, v_pphh, "both"),
                (10, v_phph, "none"),
                (8, v_phhh, "bra"),
                (0, v_hhhh, "both"),
            ]:
                m = scores == target
                if not np.any(m):
                    continue
                _mp, _mq, _mr, _ms = mp[m], mq[m], mr[m], ms[m]

                np.add.at(arr, (_mp, _mq, _mr, _ms), val)
                if mode == "bra":
                    np.add.at(arr, (_mp, _mq, _ms, _mr), -val)
                elif mode == "both":
                    np.add.at(arr, (_mq, _mp, _mr, _ms), -val)
                    np.add.at(arr, (_mp, _mq, _ms, _mr), -val)
                    np.add.at(arr, (_mq, _mp, _ms, _mr), val)

            # Handle PPPP (15)
            m15 = scores == 15
            if np.any(m15):
                base = np.column_stack([mp[m15], mq[m15], mr[m15], ms[m15]])
                idx_pppp.extend(
                    [
                        base,
                        base[:, [1, 0, 2, 3]],
                        base[:, [0, 1, 3, 2]],
                        base[:, [1, 0, 3, 2]],
                    ]
                )
                val_pppp.extend(
                    [
                        np.full(len(base), val),
                        np.full(len(base), -val),
                        np.full(len(base), -val),
                        np.full(len(base), val),
                    ]
                )

            # Handle PPPH (14)
            m14 = scores == 14
            if np.any(m14):
                base = np.column_stack([mp[m14], mq[m14], mr[m14], ms[m14]])
                idx_ppph.extend([base, base[:, [1, 0, 2, 3]]])
                val_ppph.extend([np.full(len(base), val), np.full(len(base), -val)])

    def build_sparse(ilist, vlist):
        if not ilist:
            return TwoBodyOperator(np.empty((0, 4), dtype=np.int64), np.empty(0), nstat)
        return TwoBodyOperator(np.vstack(ilist), np.concatenate(vlist), nstat)

    v_pppp = build_sparse(idx_pppp, val_pppp)
    v_ppph = build_sparse(idx_ppph, val_ppph)

    f_pp += np.einsum("aibi->ab", v_phph)
    f_ph += np.einsum("aibi->ab", v_phhh)
    f_hh += np.einsum("aibi->ab", v_hhhh)

    if stamp_3b is not None:
        from NuLattice.jax.ccm.stamp_operator_utils import stamp_to_operator
        op3 = stamp_to_operator(stamp_3b.deltas, stamp_3b.weights, L, spin, isospin)

        w_res = tbu.get_3NF(part_idx, hole_idx, op3)
        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_res[6], w_res[7], w_res[8])
        f_pp += dum_fock[0]
        f_ph += dum_fock[1]
        f_hh += dum_fock[2]

        dum_2b = tbu.get_3NF_tbme(
            w_res[2], w_res[4], w_res[5], w_res[6], w_res[7], w_res[8], pnum, hnum
        )

        def merge_ops(op1, op2):
            if len(op2) == 0:
                return op1
            if len(op1) == 0:
                return op2
            return TwoBodyOperator(
                np.concatenate([op1.indices, op2.indices]),
                np.concatenate([op1.values, op2.values]),
                nstat,
            )

        v_pppp = merge_ops(v_pppp, dum_2b[0])
        v_ppph = merge_ops(v_ppph, dum_2b[1])
        v_pphh += dum_2b[2]
        v_phph += dum_2b[3]
        v_phhh += dum_2b[4]
        v_hhhh += dum_2b[5]

        vacEn = get_ref_energy(f_hh, v_hhhh, w_res[8])
    else:
        vacEn = vacEn_pure

    two_body = [
        v_pppp,
        v_ppph,
        jnp.array(v_pphh),
        jnp.array(v_phph),
        jnp.array(v_phhh),
        jnp.array(v_hhhh),
    ]
    fock = [jnp.array(f_pp), jnp.array(f_ph), jnp.array(f_hh)]

    NO2B_stuff = vacEn, fock, two_body
    return NO2B_stuff if (NO2B or stamp_3b is None) else (NO2B_stuff, w_res)

def stamp_to_operator(
    deltas: np.ndarray, 
    weights: np.ndarray, 
    L: int, 
    spin: int = 2, 
    isospin: int = 2
):
    """
    Generic converter that transforms topological stamps into sparse Operator objects.
    Automatically detects 1-body, 2-body, or 3-body interactions by tensor rank.
    """
    num_local_states = spin * isospin
    basis_size = (L**3) * num_local_states
    n_sites = L**3

    n_legs = weights.ndim - 1
    n_body = n_legs // 2
    
    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = np.array([L**2, L, 1]) * num_local_states

    all_indices = []
    all_values = []

    for delta_matrix, W in zip(deltas, weights):
        d_matrix = np.atleast_2d(delta_matrix)
        
        full_deltas = np.vstack([np.zeros((1, 3), dtype=int), d_matrix])
        
        shifted_sites = (spatial_coords[:, np.newaxis, :] + full_deltas) % L
        bases = np.sum(shifted_sites * strides, axis=2)
        
        valid_mask = (W != 0) & (~np.isnan(W))
        internal_combos = np.array(np.where(valid_mask)).T # Shape: (N_nonzero, n_legs)
        
        for combo in internal_combos:
            val = W[tuple(combo)]
            
            global_indices = bases + combo
            
            all_indices.append(global_indices)
            all_values.append(np.full(n_sites, val))


    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    sort_keys = tuple(final_indices[:, i] for i in range(n_legs - 1, -1, -1))
    sort_idx = np.lexsort(sort_keys)

    operator_kwargs = {
        "indices": final_indices[sort_idx],
        "values": final_values[sort_idx],
        "nstat": basis_size
    }

    if n_body == 1:
        return OneBodyOperator(**operator_kwargs)
    elif n_body == 2:
        return TwoBodyOperator(**operator_kwargs)
    else:
        return ThreeBodyOperator(**operator_kwargs)
