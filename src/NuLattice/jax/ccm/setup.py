from typing import Tuple, List, Optional
import jax.numpy as jnp
import numpy as np

import NuLattice.jax.lattice as lat
from NuLattice.utils._jax_types import (
    OneBodyOperator,
    TwoBodyOperator,
    ThreeBodyOperator,
)

from . import three_body_utils as tbu


def get_fock_matrices(
    part: int,
    hole: int,
    myTkin: OneBodyOperator,
    v_phph: jnp.array,
    v_phhh: jnp.array,
    v_hhhh: jnp.array,
) -> Tuple[jnp.array, jnp.array, jnp.array]:
    pnum = len(part)
    hnum = len(hole)
    n_states = pnum + hnum
    dtype = v_phph.dtype

    h_dense = jnp.zeros((n_states, n_states), dtype=dtype)
    p = myTkin.indices[:, 0]
    q = myTkin.indices[:, 1]
    tkin_values = myTkin.values

    # JAX scatter add
    h_dense = h_dense.at[p, q].add(tkin_values)

    p_idx = jnp.array(part, dtype=jnp.int32)
    h_idx = jnp.array(hole, dtype=jnp.int32)

    f_pp = h_dense[p_idx[:, None], p_idx]
    f_ph = h_dense[p_idx[:, None], h_idx]
    f_hh = h_dense[h_idx[:, None], h_idx]

    f_pp += jnp.einsum("aibi->ab", v_phph)
    f_ph += jnp.einsum("aibi->ab", v_phhh)
    f_hh += jnp.einsum("aibi->ab", v_hhhh)

    return f_pp, f_ph, f_hh


def get_ref_energy(no_1b_hh, no_2b_hhhh, w_hhh_hhh=None):
    en = 0.0
    hnum = len(no_1b_hh)
    for i in range(hnum):
        en += no_1b_hh[(i, i)]
        for j in range(hnum):
            en -= 0.5 * no_2b_hhhh[(i, j, i, j)]

    if w_hhh_hhh is not None:
        if not isinstance(w_hhh_hhh, ThreeBodyOperator):
            raise ValueError(f"w_hhh_hhh must be ThreeBodyOperator, but got {type(w_hhh_hhh)}")
        en += tbu.get_3NF_Eref(w_hhh_hhh)
    return float(en)


def get_all_interactions(part, hole, mycontact, dtype=np.float64):
    part, hole = np.array(part), np.array(hole)
    pnum, hnum = len(part), len(hole)
    nstat = pnum + hnum
    
    all_indices = np.array(mycontact.indices, dtype=np.int32)
    max_idx = np.max(all_indices)
    
    is_p = np.zeros(max_idx + 1, dtype=bool)
    is_p[part] = True
    
    local_map = np.zeros(max_idx + 1, dtype=np.int32)
    local_map[part] = np.arange(pnum)
    local_map[hole] = np.arange(hnum)

    # Particle = 1, Hole = 0
    # e.g., PPHH -> 1100 = 12
    p_bits = is_p[all_indices].astype(np.int32)
    sector_scores = p_bits @ np.array([8, 4, 2, 1])
    
    local_idx = local_map[all_indices]
    vals = np.array(mycontact.values, dtype=dtype)

    # NOTE: will eventually get big for large enough L / atom
    v_pphh = np.zeros((pnum, pnum, hnum, hnum), dtype=dtype)
    v_phph = np.zeros((pnum, hnum, pnum, hnum), dtype=dtype)
    v_phhh = np.zeros((pnum, hnum, hnum, hnum), dtype=dtype)
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum), dtype=dtype)
    
    def add_at_sector(target, score, sign_flip=None):
        mask = (sector_scores == score)
        if not np.any(mask):
            return
        
        m_idx = local_idx[mask]
        m_val = vals[mask]
        
        a, b, c, d = m_idx[:, 0], m_idx[:, 1], m_idx[:, 2], m_idx[:, 3]
        
        np.add.at(target, (a, b, c, d), m_val)
        if sign_flip == "ket":
            np.add.at(target, (b, a, c, d), -m_val)
        elif sign_flip == "bra":
            np.add.at(target, (a, b, d, c), -m_val)
        elif sign_flip == "both":
            np.add.at(target, (b, a, c, d), -m_val)
            np.add.at(target, (a, b, d, c), -m_val)
            np.add.at(target, (b, a, d, c), m_val)

    add_at_sector(v_pphh, 12,"both") # 1100 (PPHH)
    add_at_sector(v_phph, 10)        # 1010 (PHPH)
    add_at_sector(v_phhh, 8, "bra")  # 1000 (PHHH)
    add_at_sector(v_hhhh, 0, "both") # 0000 (HHHH)

    def get_sparse_soa(score, flip_ket=False):
        mask = (sector_scores == score)
        if not np.any(mask):
            return TwoBodyOperator(np.empty((0, 4), dtype=np.int32), np.empty(0), nstat)
        
        m_idx = local_idx[mask]
        m_val = vals[mask]
        
        if flip_ket:
            # Concat (a,b,c,d) with (b,a,c,d) and flip sign
            idx = np.concatenate([m_idx, m_idx[:, [1, 0, 2, 3]]])
            v = np.concatenate([m_val, -m_val])
        else: # PPPP Full anti-sym
            idx = np.concatenate([
                m_idx,                  # abcd
                m_idx[:, [1, 0, 2, 3]], # bacd (-)
                m_idx[:, [0, 1, 3, 2]], # abdc (-)
                m_idx[:, [1, 0, 3, 2]]  # badc (+)
            ])
            v = np.concatenate([m_val, -m_val, -m_val, m_val])
            
        return TwoBodyOperator(idx, v, nstat)

    v_pppp = get_sparse_soa(15) # 1111
    v_ppph = get_sparse_soa(14, flip_ket=True) # 1110

    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh

def get_norm_ordered_ham(
    L: int,
    ref_state: List[List[int]],
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: Optional[ThreeBodyOperator] = None,
    NO2B: bool = True,
    dtype=jnp.float64,
):
    hole, part = lat.states2PHSpace(ref_state, L)
    hnum, pnum = len(hole), len(part)
    nstat = pnum + hnum

    # np except pppp, ppph
    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = get_all_interactions(
        part, hole, op2, dtype=dtype
    )

    # jnp
    f_pp, f_ph, f_hh = get_fock_matrices(part, hole, op1, v_phph, v_phhh, v_hhhh)

    if op3 is not None:
        w_res = tbu.get_3NF(part, hole, op3)

        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_res[6], w_res[7], w_res[8])
        f_pp += dum_fock[0]
        f_ph += dum_fock[1]
        f_hh += dum_fock[2]

        # returns pppp, ppph, pphh, phph, phhh, hhhh
        dum_2b = tbu.get_3NF_tbme(
            w_res[2], # dense
            w_res[4], # dense
            w_res[5], # dense
            w_res[6], # dense
            w_res[7], # dense
            w_res[8], # dense
            pnum,
            hnum,
        )

        def merge_ops(op1, op2):
            if len(op2) == 0:
                return op1
            if len(op1) == 0:
                return op2
            new_idx = jnp.concatenate([op1.indices, op2.indices], axis=0)
            new_vals = jnp.concatenate([op1.values, op2.values], axis=0)
            return TwoBodyOperator(new_idx, new_vals, nstat)

        v_pppp = merge_ops(v_pppp, dum_2b[0])
        v_ppph = merge_ops(v_ppph, dum_2b[1])

        v_pphh += dum_2b[2]
        v_phph += dum_2b[3]
        v_phhh += dum_2b[4]
        v_hhhh += dum_2b[5]

        vacEn = get_ref_energy(f_hh, v_hhhh, w_res[8])
    else:
        vacEn = get_ref_energy(f_hh, v_hhhh, None)

    two_body = [v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh]
    fock = [f_pp, f_ph, f_hh]

    NO2B_stuff = vacEn, fock, two_body
    return NO2B_stuff if (NO2B or op3 is None) else (NO2B_stuff, w_res)


def get_norm_ord_int(
    thisL: int,
    holes: int,
    vT1: float,
    vS1: float,
    str_3NF: float = 0,
    dtype=jnp.float64,
):
    lattice = lat.get_lattice(thisL)
    myTkin = lat.Tkin(lattice, thisL)
    mycontact = lat.contacts(vT1, vS1, lattice, thisL)
    hole, part = lat.states2PHSpace(holes, thisL)

    hnum, pnum = len(hole), len(part)
    nstat = hnum + pnum

    raw_2b = list(
        get_all_interactions(part, hole, mycontact, dtype=dtype)
    )

    fock_mats = list(
        get_fock_matrices(part, hole, myTkin, raw_2b[3], raw_2b[4], raw_2b[5])
    )

    if str_3NF != 0:
        my3body = lat.NNNcontact(str_3NF, lattice, thisL)
        w_ops = tbu.get_3NF(part, hole, my3body.to_list())

        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_ops[6], w_ops[7], w_ops[8])
        for i in range(3):
            fock_mats[i] += dum_fock[i]

        dum_two_body = tbu.get_3NF_tbme(
            w_ops[2],
            w_ops[4],
            w_ops[5],
            w_ops[6],
            w_ops[7],
            w_ops[8],
            pnum,
            hnum,
        )

        def merge_soa(op1, op2):
            if len(op2) == 0:
                return op1
            if len(op1) == 0:
                return op2
            new_idx = jnp.concatenate([op1.indices, op2.indices], axis=0)
            new_vals = jnp.concatenate([op1.values, op2.values], axis=0)
            return TwoBodyOperator(new_idx, new_vals, nstat)

        raw_2b[0] = merge_soa(raw_2b[0], dum_two_body[0])
        raw_2b[1] = merge_soa(raw_2b[1], dum_two_body[1])

        for i in range(2, 6):
            raw_2b[i] += dum_two_body[i]

        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], w_ops[8])
    else:
        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], None)

    return vacEn, fock_mats, raw_2b
