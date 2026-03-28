import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit)
def pAB(val):
    """Permutator for ab indices (0, 1). Returns val^{ab}_{ij} - val^{ba}_{ij}"""
    return val - jnp.transpose(val, (1, 0, 2, 3))


@partial(jax.jit)
def pIJ(val):
    """Permutator for ij indices (2, 3). Returns val^{ab}_{ij} - val^{ab}_{ji}"""
    return val - jnp.transpose(val, (0, 1, 3, 2))


@partial(jax.jit)
def v_ppph_dgrams(v_ppph_soa, t1, t2):
    indices, values = v_ppph_soa
    pnum, hnum = t1.shape

    # Initialize return buffers
    ret0 = jnp.zeros((pnum, hnum))
    ret1 = jnp.zeros((pnum, pnum))
    ret2 = jnp.zeros((pnum, pnum, hnum, hnum))
    ret3 = jnp.zeros((pnum, pnum))
    ret4 = jnp.zeros((pnum, pnum, hnum, hnum))
    ret5 = jnp.zeros((pnum, hnum, hnum, hnum))
    ret6 = jnp.zeros((pnum, hnum, hnum, hnum))

    if values.size == 0:
        return ret0, ret1, ret2, ret3, ret4, ret5, ret6

    # V indices: c, d, a, k
    idx_c, idx_d, idx_a, idx_k = indices

    # Diagram 0: ret0[a, i] -= 0.5 * sum_{cdk} V[c,d,a,k] * T2[c,d,k,i]
    term_0 = -0.5 * values[:, None] * t2[idx_c, idx_d, idx_k, :]
    ret0 = ret0.at[idx_a, :].add(term_0)

    # Diagrams 1 & 3: Scatter to ret1 and ret3
    term_1_3 = values * t1[idx_c, idx_k]
    ret1 = ret1.at[idx_a, idx_d].add(-term_1_3)
    ret3 = ret3.at[idx_d, idx_a].add(term_1_3)

    # Pre-broadcast for 2D/3D expansions
    # JAX handles broadcasting efficiently with at[...].add()

    # Diagram 2: ret2[c, d, j, k] += V[c,d,a,k] * T1[a, j]
    # We use None to align j-indices
    term_2 = values[:, None] * t1[idx_a, :]  # (nnz, hnum)
    ret2 = ret2.at[idx_c, idx_d, :, idx_k].add(term_2)

    # Diagram 4: ret4[a, d, j, k] += V[c,d,a,k] * T1[c, j]
    term_4 = values[:, None] * t1[idx_c, :]
    ret4 = ret4.at[idx_a, idx_d, :, idx_k].add(term_4)

    # Diagram 5: ret5[a, i, j, k] += V[c,d,a,k] * T2[c,d,i,j]
    term_5 = values[:, None, None] * t2[idx_c, idx_d, :, :]  # (nnz, hnum, hnum)
    ret5 = ret5.at[idx_a, :, :, idx_k].add(term_5)

    # Diagram 6: ret6[a, i, j, k] += V[c,d,a,k] * (T1[c,i]*T1[d,j])
    # Memory optimization: compute sliced doubleT1 on the fly
    t1_c = t1[idx_c, :]  # (nnz, hnum)
    t1_d = t1[idx_d, :]  # (nnz, hnum)
    # Outer product for each nnz entry: (nnz, hnum, 1) * (nnz, 1, hnum)
    term_6 = values[:, None, None] * (t1_c[:, :, None] * t1_d[:, None, :])
    ret6 = ret6.at[idx_a, :, :, idx_k].add(term_6)

    return ret0, ret1, ret2, ret3, ret4, ret5, ret6


@partial(jax.jit)
def v_pppp_dgrams(v_pppp_soa, t1, t2):
    indices, values = v_pppp_soa
    idx_a, idx_b, idx_c, idx_d = indices
    pnum, hnum = t1.shape

    # Precompute sliced intermediate to avoid large N^4 intermediate
    t1_c = t1[idx_c, :]
    t1_d = t1[idx_d, :]
    doubleT1_sliced = t1_c[:, :, None] * t1_d[:, None, :]  # (nnz, hnum, hnum)

    # ret1[a, b, :, :] += v * t2[c, d, :, :]
    term_1 = values[:, None, None] * t2[idx_c, idx_d, :, :]
    ret1 = jnp.zeros((pnum, pnum, hnum, hnum)).at[idx_a, idx_b, :, :].add(term_1)

    # ret2[a, b, :, :] += v * doubleT1[c, d, :, :]
    term_2 = values[:, None, None] * doubleT1_sliced
    ret2 = jnp.zeros((pnum, pnum, hnum, hnum)).at[idx_a, idx_b, :, :].add(term_2)

    return ret1, ret2


@partial(jax.jit)
def dgram_akci_ck(v_phph, t1):
    return -jnp.einsum("akci, ck -> ai", v_phph, t1)


@partial(jax.jit)
def dgram_ck_acik(f_ph, t2):
    return jnp.einsum("ck, acik -> ai", f_ph, t2)


@partial(jax.jit)
def dgram_cikl_cakl(v_phhh, t2):
    return -0.5 * jnp.einsum("cikl, cakl -> ai", v_phhh, t2)


@partial(jax.jit)
def dgram_cdkl_ck_dali(v_pphh, t1, t2):
    return jnp.einsum("cdkl, ck, dali -> ai", v_pphh, t1, t2)


@partial(jax.jit)
def dgram_ck_ci(f_ph, t1):
    return -0.5 * jnp.einsum("ck, ci -> ki", f_ph, t1)


@partial(jax.jit)
def dgram_ck_ak(f_ph, t1):
    return -0.5 * jnp.einsum("ck, ak -> ac", f_ph, t1)


@partial(jax.jit)
def dgram_bijk_bj(v_phhh, t1):
    return -jnp.einsum("bijk, bj -> ki", v_phhh, t1)


@partial(jax.jit)
def dgram_cdlk_cdli(v_pphh, t2):
    return -0.5 * jnp.einsum("cdlk, cdli -> ki", v_pphh, t2)


@partial(jax.jit)
def dgram_dckl_dakl(v_pphh, t2):
    return -0.5 * jnp.einsum("dckl, dakl -> ac", v_pphh, t2)


@partial(jax.jit)
def dgram_cdlk_cl_di(v_pphh, t1):
    return -0.5 * jnp.einsum("cdlk, cl, di -> ki", v_pphh, t1, t1)


@partial(jax.jit)
def dgram_cdkl_dk_al(v_pphh, t1):
    return 0.5 * jnp.einsum("cdkl, dk, al -> ac", v_pphh, t1, t1)


@partial(jax.jit)
def dgram_klij_abkl(v_hhhh, t2):
    return 0.5 * jnp.einsum("klij, abkl -> abij", v_hhhh, t2)


@partial(jax.jit)
def dgram_bkcj_acik(v_phph, t2):
    contracted = jnp.einsum("bkcj, acik -> abij", v_phph, t2)
    return -pIJ(pAB(contracted))


@partial(jax.jit)
def dgram_bkij_ak(v_phhh, t1):
    contracted = jnp.einsum("bkij, ak -> abij", v_phhh, t1)
    return pAB(contracted)


@partial(jax.jit)
def dgram_cdkl_acik_dblj(v_pphh, t2, t2_alt):
    # Pass t2 twice as requested in original logic
    return 0.5 * pIJ(pAB(jnp.einsum("cdkl, acik, dblj -> abij", v_pphh, t2, t2_alt)))


@partial(jax.jit)
def dgram_cdkl_cdij_abkl(v_pphh, t2_a, t2_b):
    return 0.25 * jnp.einsum("cdkl, cdij, abkl -> abij", v_pphh, t2_a, t2_b)


@partial(jax.jit)
def dgram_klij_ak_bl(v_hhhh, t1):
    return 0.5 * pAB(jnp.einsum("klij, ak, bl -> abij", v_hhhh, t1, t1))


@partial(jax.jit)
def dgram_bkci_ak_cj(v_phph, t1):
    return -pAB(pIJ(jnp.einsum("bkci, ak, cj -> abij", v_phph, t1, t1)))


@partial(jax.jit)
def dgram_cikl_ck_ablj(v_phhh, t1, t2):
    return -pIJ(jnp.einsum("cikl, ck, ablj -> abij", v_phhh, t1, t2))


@partial(jax.jit)
def dgram_da_dbij(v_ppph_res, t2):
    contracted = jnp.einsum("da, dbij -> abij", v_ppph_res, t2)
    return -pAB(contracted)


@partial(jax.jit)
def dgram_acik_bcjk(v_ppph_res, t2):
    contracted = jnp.einsum("acik, bcjk -> abij", v_ppph_res, t2)
    return pIJ(pAB(contracted))


@partial(jax.jit)
def dgram_cikl_al_bcjk(v_phhh, t1, t2):
    return -pIJ(pAB(jnp.einsum("cikl, al, bcjk -> abij", v_phhh, t1, t2)))


@partial(jax.jit)
def dgram_cjkl_ci_abkl(v_phhh, t1, t2):
    return 0.5 * pIJ(jnp.einsum("cjkl, ci, abkl -> abij", v_phhh, t1, t2))


@partial(jax.jit)
def dgram_bijk_ak1(v_ppph_res, t1):
    return 0.5 * pAB(jnp.einsum("bijk, ak -> abij", v_ppph_res, t1))


@partial(jax.jit)
def dgram_bijk_ak2(v_ppph_res, t1):
    return 0.5 * pIJ(pAB(jnp.einsum("bijk, ak -> abij", v_ppph_res, t1)))


@partial(jax.jit)
def dgram_cjkl_ci_ak_bl(v_phhh, t1):
    return 0.5 * pIJ(pAB(jnp.einsum("cjkl, ci, ak, bl -> abij", v_phhh, t1, t1, t1)))


@partial(jax.jit)
def dgram_cdkl_ci_dj_abkl(v_pphh, t1, t2):
    return 0.25 * pIJ(jnp.einsum("cdkl, ci, dj, abkl -> abij", v_pphh, t1, t1, t2))


@partial(jax.jit)
def dgram_cdkl_ak_bl_cdij(v_pphh, t1, t2):
    return 0.25 * pAB(jnp.einsum("cdkl, ak, bl, cdij -> abij", v_pphh, t1, t1, t2))


@partial(jax.jit)
def dgram_cdkl_ci_bl_adkj(v_pphh, t1, t2):
    return pIJ(pAB(jnp.einsum("cdkl, ci, bl, adkj -> abij", v_pphh, t1, t1, t2)))


@partial(jax.jit)
def dgram_cdkl_ci_ak_dj_bl(v_pphh, t1):
    return 0.25 * pIJ(
        pAB(jnp.einsum("cdkl, ci, ak, dj, bl -> abij", v_pphh, t1, t1, t1, t1))
    )


@partial(jax.jit)
def dgram_cdkl_bdkl(v_pphh, t2):
    return -0.5 * jnp.einsum("cdkl, bdkl -> bc", v_pphh, t2)


@partial(jax.jit)
def dgram_cdkl_cdjl(v_pphh, t2):
    return -0.5 * jnp.einsum("cdkl, cdjl -> kj", v_pphh, t2)


@partial(jax.jit)
def dgram_ck_bk(f_ph, t1):
    return -jnp.einsum("ck, bk -> bc", f_ph, t1)


@partial(jax.jit)
def dgram_ck_cj(f_ph, t1):
    return -jnp.einsum("ck, cj -> kj", f_ph, t1)


@partial(jax.jit)
def dgram_cdlk_cl_dj(v_pphh, t1):
    return -jnp.einsum("cdlk, cl, dj -> kj", v_pphh, t1, t1)


@partial(jax.jit)
def dgram_cdlk_dk_bl(v_pphh, t1):
    return -jnp.einsum("cdlk, dk, bl -> bc", v_pphh, t1, t1)
