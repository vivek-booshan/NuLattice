from collections import deque

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from NuLattice.utils._jax_types import Chef

from . import ccDgrams as dgrams


@jax.jit
def ccsd_energy(f_ph, v_pphh, t2, t1):
    e_1 = jnp.einsum("ai,ai->", f_ph, t1)
    e_2 = 0.25 * jnp.einsum("abij,abij->", v_pphh, t2)
    e_3 = 0.5 * jnp.einsum("abij,ai,bj->", v_pphh, t1, t1)
    return e_1 + e_2 + e_3


@jax.jit
def t1Init(f_ph, f_pp, f_hh, delta):
    diag_h = jnp.diag(f_hh)
    diag_p = -jnp.diag(f_pp)
    denom = (diag_p[:, None] + diag_h[None, :]) + delta
    return f_ph / denom


@jax.jit
def t2Init(f_pp, f_hh, v_pphh, delta):
    diag_h = jnp.diag(f_hh)
    diag_p = -jnp.diag(f_pp)

    denom_hh = diag_h[None, :] + diag_h[:, None]  # j, i -> ij
    denom_pp = diag_p[None, :] + diag_p[:, None]  # b, a -> ab

    denom = (denom_pp[:, :, None, None] + denom_hh[None, None, :, :]) + delta
    return v_pphh / denom


@jax.jit
def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph_soa):
    indices, values = v_ppph_soa
    idx_c, idx_d, idx_a, idx_k = indices

    H1 = f_ph + dgrams.dgram_akci_ck(v_phph, t1)
    H1 += dgrams.dgram_ck_acik(f_ph, t2)
    H1 += dgrams.dgram_cikl_cakl(v_phhh, t2)
    H1 += dgrams.dgram_cdkl_ck_dali(v_pphh, t1, t2)

    # v_ppph dgram
    # Diagram h1[a, i] -= 0.5 * sum_{cdk} V[c,d,a,k] * T2[c,d,k,i]
    H1.at[idx_a, :].add(-0.5 * values[:, None] * t2[idx_c, idx_d, idx_k, :])

    X_hh = -f_hh + dgrams.dgram_ck_ci(f_ph, t1)
    X_hh += dgrams.dgram_bijk_bj(v_phhh, t1)
    X_hh += dgrams.dgram_cdlk_cdli(v_pphh, t2)
    X_hh += dgrams.dgram_cdlk_cl_di(v_pphh, t1)

    X_pp = f_pp + dgrams.dgram_ck_ak(f_ph, t1)
    X_pp += dgrams.dgram_dckl_dakl(v_pphh, t2)
    X_pp += dgrams.dgram_cdkl_dk_al(v_pphh, t1)

    # v_ppph dgram
    X_pp.at[idx_a, idx_d].add(-(values * t1[idx_c, idx_k]))

    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    diag_h = jnp.diag(X_hh)
    diag_p = jnp.diag(X_pp)
    denom = -(diag_p[:, None] + diag_h[None, :])

    return t1 + (H1 / denom)


@jax.jit
def t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh):
    X_hh = -f_hh + dgrams.dgram_cdkl_cdjl(v_pphh, t2)
    X_hh += dgrams.dgram_ck_cj(f_ph, t1)
    X_hh += dgrams.dgram_cdlk_cl_dj(v_pphh, t1)

    X_pp = f_pp + dgrams.dgram_cdkl_bdkl(v_pphh, t2)
    X_pp += dgrams.dgram_ck_bk(f_ph, t1)
    X_pp += dgrams.dgram_cdlk_dk_bl(v_pphh, t1)
    return X_hh, X_pp


@jax.jit
def t2_H2(t1, t2, v_pppp, v_ppph_soa, v_pphh, v_phph, v_phhh, v_hhhh):
    indices, values = v_ppph_soa
    pnum, hnum = t1.shape
    idx_c, idx_d, idx_a, idx_k = indices

    H2 = v_pphh + dgrams.dgram_klij_abkl(v_hhhh, t2)
    H2 += dgrams.dgram_bkcj_acik(v_phph, t2)
    H2 += dgrams.dgram_bkij_ak(v_phhh, t1)
    H2 += dgrams.dgram_cdkl_acik_dblj(v_pphh, t2, t2)
    H2 += dgrams.dgram_cdkl_cdij_abkl(v_pphh, t2, t2)
    H2 += dgrams.dgram_klij_ak_bl(v_hhhh, t1)
    H2 += dgrams.dgram_bkci_ak_cj(v_phph, t1)
    H2 += dgrams.dgram_cikl_ck_ablj(v_phhh, t1, t2)
    H2 += dgrams.dgram_cikl_al_bcjk(v_phhh, t1, t2)
    H2 += dgrams.dgram_cjkl_ci_abkl(v_phhh, t1, t2)
    H2 += dgrams.dgram_cjkl_ci_ak_bl(v_phhh, t1)
    H2 += dgrams.dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2)
    H2 += dgrams.dgram_cdkl_ci_ak_dj_bl(v_pphh, t1)

    ## v_ppph dgrams
    # H2 += dgrams.pIJ(v_ppph[2])
    # H2 += dgrams.dgram_da_dbij(v_ppph[3], t2)
    # H2 += dgrams.dgram_acik_bcjk(v_ppph[4], t2)
    # H2 += dgrams.dgram_bijk_ak1(v_ppph[5], t1)
    # H2 += dgrams.dgram_bijk_ak2(v_ppph[6], t1)

    # H2 += dgrams.pIJ(v_ppph_res[2])  <-- OLD ret2 (p,p,h,h)
    # ret2[c, d, j, k] += V[c,d,a,k] * T1[a, j]
    term_2 = values[:, None] * t1[idx_a, :]  # (nnz, hnum)
    H2 = H2.at[idx_c, idx_d, :, idx_k].add(term_2)
    H2 = H2.at[idx_c, idx_d, idx_k, :].add(-term_2)  # pIJ

    # H2 += dgrams.dgram_da_dbij(v_ppph_res[3], t2) <-- OLD ret3 (p,p)
    # ret3[d, a] += V[c,d,a,k] * T1[c, k]
    H2 += dgrams.dgram_da_dbij(
        jnp.zeros((pnum, pnum)).at[idx_d, idx_a].add(values * t1[idx_c, idx_k]), t2
    )
    # Diagram 4: ret4[a, d, j, k] += V[c,d,a,k] * T1[c, j]
    H2 += dgrams.dgram_acik_bcjk(
        jnp.zeros((pnum, pnum, hnum, hnum))
        .at[idx_a, idx_d, :, idx_k]
        .add(values[:, None] * t1[idx_c, :]),
        t2,
    )

    # H2 += dgrams.dgram_bijk_ak1(v_ppph_res[5], t1)
    # ret5[a, i, j, k] += V[c,d,a,k] * T2[c,d,i,j]
    H2 += dgrams.dgram_bijk_ak1(
        jnp.zeros((pnum, hnum, hnum, hnum))
        .at[idx_a, :, :, idx_k]
        .add(values[:, None, None] * t2[idx_c, idx_d, :, :]),
        t1,
    )

    # H2 += dgrams.dgram_bijk_ak2(v_ppph_res[6], t1)
    # ret6[a, i, j, k] += V[c,d,a,k] * (T1[c,i]*T1[d,j])
    t1_c = t1[idx_c, :]
    t1_d = t1[idx_d, :]
    H2 += dgrams.dgram_bijk_ak2(
        jnp.zeros((pnum, hnum, hnum, hnum))
        .at[idx_a, :, :, idx_k]
        .add(values[:, None, None] * (t1_c[:, :, None] * t1_d[:, None, :])),
        t1,
    )

    ## v_pppp dgrams
    p_idx_a, p_idx_b, p_idx_c, p_idx_d = v_pppp[0]
    p_values = v_pppp[1]

    # 1. Diagram 1: H2 += 0.5 * V[a,b,c,d] * T2[c,d,i,j]
    # Avoids the 'ret1' 8GB buffer entirely
    term_1 = p_values[:, None, None] * t2[p_idx_c, p_idx_d, :, :]
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(0.5 * term_1)

    # 2. Diagram 2: H2 += 0.5 * pIJ( V[a,b,c,d] * T1[c,i] * T1[d,j] )
    # Avoids the 'ret2' 8GB buffer AND the transpose ghost in pIJ
    t1_c = t1[p_idx_c, :]
    t1_d = t1[p_idx_d, :]
    term_2 = p_values[:, None, None] * (t1_c[:, :, None] * t1_d[:, None, :])

    # Add anti-symmetrically in-place to avoid pIJ(ghost)
    H2_contrib = 0.5 * term_2
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(H2_contrib)
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(-H2_contrib.transpose(0, 2, 1))
    # ret1, ret2 = dgrams.v_pppp_dgrams(v_pppp, t1, t2)
    # H2 += 0.5 * ret1
    # H2 += 0.5 * dgrams.pIJ(ret2)

    return H2


@jax.jit
def t2_update(t2, X_hh, X_pp, H2):
    # H2 += dgrams.pAB(jnp.einsum("bc, acij -> abij", X_pp, t2))
    # H2 += dgrams.pIJ(jnp.einsum("kj, abik -> abij", X_hh, t2))
    term_pp = jnp.einsum("bc, acij -> abij", X_pp, t2)
    H2 += term_pp
    H2 -= term_pp.transpose(1, 0, 2, 3) 
    del term_pp

    term_hh = jnp.einsum("kj, abik -> abij", X_hh, t2)
    H2 += term_hh
    H2 -= term_hh.transpose(0, 1, 3, 2)
    del term_hh

    diag_h = jnp.diag(X_hh)
    diag_p = jnp.diag(X_pp)
    denom_hh = diag_h[None, :] + diag_h[:, None]
    denom_pp = diag_p[None, :] + diag_p[:, None]

    return t2 + (H2 / -(denom_pp[:, :, None, None] + denom_hh[None, None, :, :]))


def ccsd_solver(
    fock_mats,
    two_body_int,
    t1initial=None,
    eps=1e-8,
    maxSteps=1000,
    max_diis=10,
    delta=0,
    mixing=0.5,
    verbose=False,
    ccs=False,
    dtype=jnp.float64,
    chef: Chef = None,
):

    f_pp, f_ph, f_hh = fock_mats
    v_pppp_sparse, v_ppph_sparse, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int

    v_pppp = (v_pppp_sparse.indices.T, v_pppp_sparse.values)
    v_ppph = (v_ppph_sparse.indices.T, v_ppph_sparse.values)

    if chef is not None:
        mesh = chef.mesh

        f_pp = chef.prepare(f_pp)
        f_ph = chef.prepare(f_ph)
        f_hh = chef.prepare(f_hh, rank=0)  # replicate

        v_pphh = chef.prepare(v_pphh)
        v_phph = chef.prepare(v_phph)
        v_phhh = chef.prepare(v_phhh)
        v_hhhh = chef.prepare(v_hhhh, rank=0)  # replicate

        idx_sharding = NamedSharding(mesh, P(None, "data"))
        val_sharding = NamedSharding(mesh, P("data"))

        # NOTE(vivek): to_soa_sparse may emit tensor of size zero, avoid sharding that
        # NOTE(vivek): but does size = 0 ever happen? if so, shouldn't we raise error?
        if v_pppp[0].size > 0:
            v_pppp = (
                jax.device_put(v_pppp[0], idx_sharding),
                jax.device_put(v_pppp[1], val_sharding),
            )

        v_ppph = (
            jax.device_put(v_ppph[0], idx_sharding),
            jax.device_put(v_ppph[1], val_sharding),
        )

    t1 = (
        t1Init(f_ph, f_pp, f_hh, delta)
        if t1initial is None
        else jnp.array(t1initial, dtype) # possible source of memory issue
    )
    t2 = (
        jnp.zeros_like(v_pphh) # possible source of memory issue
        if (ccs or t1initial is not None)
        else t2Init(f_pp, f_hh, v_pphh, delta)
    )

    if max_diis > 0:
        diis_t1 = deque(maxlen=max_diis)
        diis_t2 = deque(maxlen=max_diis)
        diis_errors = deque(maxlen=max_diis)
        diis_t1.append(t1)
        diis_t2.append(t2)

    prevEnergy = ccsd_energy(f_ph, v_pphh, t2, t1)
    if verbose:
        print(f"Step 0: {prevEnergy}")

    for step in range(maxSteps):
        oldT1, oldT2 = t1, t2

        t1_new = t1Iter(
            t1,
            t2,
            f_ph,
            f_pp,
            f_hh,
            v_phph,
            v_phhh,
            v_pphh,
            v_ppph,
        )
        t1 = t1 + mixing * (t1_new - t1)

        if not ccs:
            X_hh, X_pp = t2_X(oldT1, t2, f_pp, f_ph, f_hh, v_pphh)
            X_pp.block_until_ready()  # force intermediate dealloc

            H2 = t2_H2(
                oldT1,
                t2,
                v_pppp,
                v_ppph,
                v_pphh,
                v_phph,
                v_phhh,
                v_hhhh,
            )
            H2.block_until_ready()  # force intermediate dealloc

            t2_new = t2_update(t2, X_hh, X_pp, H2)
            del X_hh, X_pp, H2
            t2 = t2 + mixing * (t2_new - t2)

        energy = ccsd_energy(f_ph, v_pphh, t2, t1)
        diff = abs(energy - prevEnergy) / max(1.0, abs(energy))

        if verbose:
            print(f"Step {step + 1}: {energy} difference = {diff}")

        if diff < eps:
            return float(energy), t1, t2

        if max_diis > 0:
            diis_t1.append(t1)
            diis_t2.append(t2)

            # STORE AS NATIVE TUPLES. Do not use flatten/reshape(-1)!
            # Flattening breaks GSPMD layout and causes Out-Of-Memory.
            diis_errors.append((t1 - oldT1, t2 - oldT2))

            if len(diis_errors) == max_diis:
                size = len(diis_errors)
                B = jnp.zeros((size, size), dtype=dtype)

                for x in range(size):
                    for y in range(x, size):
                        e1x, e2x = diis_errors[x]
                        e1y, e2y = diis_errors[y]

                        # Local element-wise mult + global reduction sum.
                        # Communicates a single scalar rather than gigabytes of data.
                        val = jnp.sum(e1x * e1y) + jnp.sum(e2x * e2y)

                        B = B.at[x, y].set(val)
                        if x != y:
                            B = B.at[y, x].set(val)

                B = B / (jnp.max(jnp.abs(B)) + 1e-16)

                A = -jnp.ones((size + 1, size + 1), dtype=dtype)
                A = A.at[:size, :size].set(B)
                A = A.at[size, size].set(0.0)

                rhs = jnp.zeros(size + 1, dtype=dtype)
                rhs = rhs.at[size].set(-1.0)

                try:
                    c = jnp.linalg.solve(A, rhs)[:size]
                    t1_new_diis = jnp.zeros_like(t1)
                    t2_new_diis = jnp.zeros_like(t2)

                    for k in range(size):
                        t1_new_diis += c[k] * diis_t1[k]
                        if not ccs:
                            t2_new_diis += c[k] * diis_t2[k]

                    t1, t2 = t1_new_diis, t2_new_diis
                except Exception:
                    pass

                diis_t1.clear()
                diis_t2.clear()
                diis_errors.clear()

        if abs(energy) > 1e10 or jnp.isnan(energy):
            print("Diverged.")
            break

        prevEnergy = energy

    print("Max iterations reached.")
    return float(energy), t1, t2
