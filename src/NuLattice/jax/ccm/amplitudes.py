# for solving t1 and t2
import jax
import jax.numpy as jnp

from . import ccDgrams as dgrams

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


