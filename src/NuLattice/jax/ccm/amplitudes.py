# for solving t1 and t2
import jax
import jax.numpy as jnp

from . import ccDgrams as dgrams
from .ccDgrams import (
    add_AB,
    add_IJ,
    add_AB_IJ,
)


@jax.jit
def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph):
    indices, values = v_ppph
    idx_c, idx_d, idx_a, idx_k = indices

    H1 = f_ph
    H1 -= jnp.einsum("akci, ck -> ai", v_phph, t1)
    H1 += jnp.einsum("ck, acik -> ai", f_ph, t2)
    H1 -= 0.5 * jnp.einsum("cikl, cakl -> ai", v_phhh, t2)
    H1 += jnp.einsum("cdkl, ck, dali -> ai", v_pphh, t1, t2)

    # v_ppph dgram
    # Diagram h1[a, i] -= 0.5 * sum_{cdk} V[c,d,a,k] * T2[c,d,k,i]
    H1.at[idx_a, :].add(-0.5 * values[:, None] * t2[idx_c, idx_d, idx_k, :])

    X_hh = -f_hh
    X_hh -= 0.5 * jnp.einsum("ck, ci -> ki", f_ph, t1)
    X_hh -= jnp.einsum("bijk, bj -> ki", v_phhh, t1)
    X_hh -= jnp.einsum("cdlk, cdli -> ki", v_pphh, t2)
    X_hh -= 0.5 * jnp.einsum("cdlk, cl, di -> ki", v_pphh, t1, t1)

    X_pp = f_pp
    X_pp -= 0.5 * jnp.einsum("ck, ak -> ac", f_ph, t1)
    X_pp -= 0.5 * jnp.einsum("dckl, dakl -> ac", v_pphh, t2)
    X_pp += 0.5 * jnp.einsum("cdkl, dk, al -> ac", v_pphh, t1, t1)

    # v_ppph dgram
    X_pp.at[idx_a, idx_d].add(-values * t1[idx_c, idx_k])

    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    return t1 - (H1 / (jnp.diag(X_pp)[:, None] + jnp.diag(X_hh)[None, :]))


@jax.jit
def t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh):
    X_hh = -f_hh
    X_hh -= 0.5 * jnp.einsum("cdkl, cdjl -> kj", v_pphh, t2)
    X_hh -= jnp.einsum("ck, cj -> kj", f_ph, t1)
    X_hh -= jnp.einsum("cdlk, cl, dj -> kj", v_pphh, t1, t1)

    X_pp = f_pp
    X_pp -= 0.5 * jnp.einsum("cdkl, bdkl -> bc", v_pphh, t2)
    X_pp -= jnp.einsum("ck, bk -> bc", f_ph, t1)
    X_pp -= jnp.einsum("cdlk, dk, bl -> bc", v_pphh, t1, t1)
    return X_hh, X_pp


# NOTE: no jax.jit to force sequential execution and avoid intermediate storage
def t2_H2(t1, t2, v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh):
    H2 = t2_H2_dense(t1, t2, v_pphh, v_phph, v_phhh, v_hhhh)
    H2 = t2_H2_ppph(H2, t1, t2, v_ppph)
    H2 = t2_H2_pppp(H2, t1, t2, v_pppp)
    return H2


# NOTE: somewhere in this code causes all-gather, which stops scaling
@jax.jit
def t2_H2_dense(t1, t2, v_pphh, v_phph, v_phhh, v_hhhh):
    H2 = v_pphh
    H2 = H2.at[:].add(0.5 * jnp.einsum("klij, abkl -> abij", v_hhhh, t2))
    H2 = add_AB_IJ(H2, -jnp.einsum("bkcj, acik -> abij", v_phph, t2))
    H2 = add_AB(H2, jnp.einsum("bkij, ak -> abij", v_phhh, t1))
    H2 = add_AB_IJ(H2, 0.5 * jnp.einsum("cdkl, acik, dblj -> abij", v_pphh, t2, t2))
    H2 = H2.at[:].add(0.25 * jnp.einsum("cdkl, cdij, abkl -> abij", v_pphh, t2, t2))
    H2 = add_AB(H2, 0.5 * jnp.einsum("klij, ak, bl -> abij", v_hhhh, t1, t1))
    H2 = add_AB_IJ(H2, -jnp.einsum("bkci, ak, cj -> abij", v_phph, t1, t1))
    H2 = add_IJ(H2, -jnp.einsum("cikl, ck, ablj -> abij", v_phhh, t1, t2))
    H2 = add_AB_IJ(H2, -jnp.einsum("cikl, al, bcjk -> abij", v_phhh, t1, t2))
    H2 = add_IJ(H2, 0.5 * jnp.einsum("cjkl, ci, abkl -> abij", v_phhh, t1, t2))
    H2 = add_AB_IJ(H2, 0.5 * jnp.einsum("cjkl, ci, ak, bl -> abij", v_phhh, t1, t1, t1))
    H2 = add_IJ(H2, 0.25 * jnp.einsum("cdkl, ci, dj, abkl -> abij", v_pphh, t1, t1, t2))
    H2 = add_AB(H2, 0.25 * jnp.einsum("cdkl, ak, bl, cdij -> abij", v_pphh, t1, t1, t2))
    H2 = add_AB_IJ(H2, jnp.einsum("cdkl, ci, bl, adkj -> abij", v_pphh, t1, t1, t2))
    H2 = add_AB_IJ(
        H2, 0.25 * jnp.einsum("cdkl, ci, ak, dj, bl -> abij", v_pphh, t1, t1, t1, t1)
    )
    return H2


@jax.jit
def t2_H2_ppph(H2, t1, t2, v_ppph):
    indices, values = v_ppph
    idx_c, idx_d, idx_a, idx_k = indices
    pnum, hnum = t1.shape
    ## v_ppph
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
        # jnp.zeros((pnum, pnum, hnum, hnum))
        jnp.zeros_like(H2)
        .at[idx_a, idx_d, :, idx_k]
        .add(values[:, None] * t1[idx_c, :]),
        t2,
    )

    # H2 += dgrams.dgram_bijk_ak1(v_ppph_res[5], t1)
    # ret5[a, i, j, k] += V[c,d,a,k] * T2[c,d,i,j]
    H2 += dgrams.dgram_bijk_ak1(
        jnp.zeros((pnum, hnum, hnum, hnum))
        # jnp.zeros_like(H2)
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
    return H2


@jax.jit
def t2_H2_pppp(H2, t1, t2, v_pppp):
    ## v_pppp dgrams
    p_idx_a, p_idx_b, p_idx_c, p_idx_d = v_pppp[0]
    values = v_pppp[1]

    # 1. Diagram 1: H2 += 0.5 * V[a,b,c,d] * T2[c,d,i,j]
    # Avoids the 'ret1' 8GB buffer entirely
    term_1 = values[:, None, None] * t2[p_idx_c, p_idx_d, :, :]
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(0.5 * term_1)

    # 2. Diagram 2: H2 += 0.5 * pIJ( V[a,b,c,d] * T1[c,i] * T1[d,j] )
    # Avoids the 'ret2' 8GB buffer AND the transpose ghost in pIJ
    t1_c = t1[p_idx_c, :]
    t1_d = t1[p_idx_d, :]
    term_2 = 0.5 * values[:, None, None] * (t1_c[:, :, None] * t1_d[:, None, :])
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(term_2)
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(-term_2.transpose(0, 2, 1))
    return H2


@jax.jit
def t2_update(t2, X_hh, X_pp, H2):
    H2 = add_AB(H2, jnp.einsum("bc, acij -> abij", X_pp, t2))
    H2 = add_IJ(H2, jnp.einsum("kj, abik -> abij", X_hh, t2))

    diag_h = jnp.diag(X_hh)
    diag_p = jnp.diag(X_pp)
    return t2 - (
        H2
        / (
            diag_p[:, None, None, None]
            + diag_p[None, :, None, None]
            + diag_h[None, None, :, None]
            + diag_h[None, None, None, :]
        )
    )
