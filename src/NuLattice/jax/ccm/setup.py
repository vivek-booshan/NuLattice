from typing import Tuple
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
        if isinstance(w_hhh_hhh, ThreeBodyOperator):
            en += tbu.get_3NF_Eref(w_hhh_hhh)
        else:
            for ele in w_hhh_hhh:
                [m, i, j, n, k, l, val] = ele
                if (m, i, j) == (n, k, l):
                    en += val / 6.0
    return float(en)


def get_all_interactions(part, hole, mycontact, sparse=False, dtype=jnp.float64):
    pnum, hnum = len(part), len(hole)
    nstat = pnum + hnum

    lookup_h = {idx: i for i, idx in enumerate(hole)}
    lookup_p = {idx: i for i, idx in enumerate(part)}

    # PREVENT JAX GRAPH EXPLOSION: Initialize as pure NumPy arrays.
    # Mutating a jnp array inside a native loop destroys compilation time and RAM.
    if sparse:
        v_pppp_list, v_ppph_list = [], []
    else:
        v_pppp = np.zeros((pnum, pnum, pnum, pnum), dtype=dtype)
        v_ppph = np.zeros((pnum, pnum, pnum, hnum), dtype=dtype)

    v_pphh = np.zeros((pnum, pnum, hnum, hnum), dtype=dtype)
    v_phph = np.zeros((pnum, hnum, pnum, hnum), dtype=dtype)
    v_phhh = np.zeros((pnum, hnum, hnum, hnum), dtype=dtype)
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum), dtype=dtype)

    def get_indices_and_signs(a, b, c, d, sector):
        if sector in [("p", "p", "p", "p"), ("p", "p", "h", "h"), ("h", "h", "h", "h")]:
            return ((a, b, c, d), (b, a, c, d), (a, b, d, c), (b, a, d, c)), (
                1.0,
                -1.0,
                -1.0,
                1.0,
            )
        if sector == ("p", "p", "p", "h"):
            return ((a, b, c, d), (b, a, c, d)), (1.0, -1.0)
        if sector == ("p", "h", "h", "h"):
            return ((a, b, c, d), (a, b, d, c)), (1.0, -1.0)
        if sector == ("p", "h", "p", "h"):
            return ((a, b, c, d),), (1.0,)
        return None, None

    indices = np.array(mycontact.indices)
    values = np.array(mycontact.values)

    for position, val in zip(indices, values):
        i1, i2, i3, i4 = position
        k_t = [("h" if i in hole else "p") for i in [i1, i2]]
        b_t = [("h" if i in hole else "p") for i in [i3, i4]]

        s_k, s_b = 1.0, 1.0
        if k_t == ["h", "p"]:
            i1, i2, k_t, s_k = i2, i1, ["p", "h"], -1.0
        if b_t == ["h", "p"]:
            i3, i4, b_t, s_b = i4, i3, ["p", "h"], -1.0

        sector = tuple(k_t + b_t)
        mapped = [
            lookup_p[i] if t == "p" else lookup_h[i]
            for i, t in zip([i1, i2, i3, i4], sector)
        ]

        target = {
            ("p", "p", "p", "p"): (v_pppp_list if sparse else v_pppp, True),
            ("p", "p", "p", "h"): (v_ppph_list if sparse else v_ppph, True),
            ("p", "p", "h", "h"): (v_pphh, False),
            ("p", "h", "p", "h"): (v_phph, False),
            ("p", "h", "h", "h"): (v_phhh, False),
            ("h", "h", "h", "h"): (v_hhhh, False),
        }.get(sector)

        if target:
            buf, is_sparse_candidate = target
            perms, signs = get_indices_and_signs(*mapped, sector)
            base_val = float(val) * s_k * s_b

            for p, s in zip(perms, signs):
                term = base_val * s
                if is_sparse_candidate and sparse:
                    buf.append([p[0], p[1], p[2], p[3], term])
                else:
                    buf[p] = term

    if sparse:
        v_pppp = TwoBodyOperator.from_list(v_pppp_list, nstat)
        v_ppph = TwoBodyOperator.from_list(v_ppph_list, nstat)
    else:
        v_pppp = jnp.array(v_pppp, dtype=dtype)
        v_ppph = jnp.array(v_ppph, dtype=dtype)


    return v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh


def get_norm_ordered_ham(
    thisL: int,
    holes: int,
    myTkin: OneBodyOperator,
    mycontact: TwoBodyOperator,
    my3body: ThreeBodyOperator = None,
    sparse: bool = True,
    NO2B: bool = True,
    dtype=jnp.float64,
):
    hole, part = lat.states2PHSpace(holes, thisL)
    hnum, pnum = len(hole), len(part)
    nstat = pnum + hnum

    v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh = get_all_interactions(
        part, hole, mycontact, sparse=sparse, dtype=dtype
    )

    f_pp, f_ph, f_hh = get_fock_matrices(part, hole, myTkin, v_phph, v_phhh, v_hhhh)

    if my3body is not None:
        w_res = tbu.get_3NF(part, hole, my3body.to_list())

        dum_fock = tbu.get_3NF_fock(hnum, pnum, w_res[6], w_res[7], w_res[8])
        f_pp += dum_fock[0]
        f_ph += dum_fock[1]
        f_hh += dum_fock[2]

        dum_2b = tbu.get_3NF_tbme(
            w_res[2],
            w_res[4],
            w_res[5],
            w_res[6],
            w_res[7],
            w_res[8],
            pnum,
            hnum,
            sparse_pppp=sparse,
            sparse_ppph=sparse,
        )

        if sparse:

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
        else:
            v_pppp += dum_2b[0]
            v_ppph += dum_2b[1]

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
    return NO2B_stuff if (NO2B or my3body is None) else (NO2B_stuff, w_res)


def get_norm_ord_int(
    thisL: int,
    holes: int,
    vT1: float,
    vS1: float,
    str_3NF: float = 0,
    sparse: bool = True,
    dtype=jnp.float64,
):
    lattice = lat.get_lattice(thisL)
    myTkin = lat.Tkin(lattice, thisL)
    mycontact = lat.contacts(vT1, vS1, lattice, thisL)
    hole, part = lat.states2PHSpace(holes, thisL)

    hnum, pnum = len(hole), len(part)
    nstat = hnum + pnum

    raw_2b = list(
        get_all_interactions(part, hole, mycontact, sparse=sparse, dtype=dtype)
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
            sparse_pppp=sparse,
            sparse_ppph=sparse,
        )

        if sparse:

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
        else:
            raw_2b[0] += dum_two_body[0]
            raw_2b[1] += dum_two_body[1]

        for i in range(2, 6):
            raw_2b[i] += dum_two_body[i]

        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], w_ops[8])
    else:
        vacEn = get_ref_energy(fock_mats[2], raw_2b[5], None)

    return vacEn, fock_mats, raw_2b
