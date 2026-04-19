from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint

@jax.jit
def add_AB(target, val):
    """target inplace mutation of AB permutation of val"""
    return target.at[:].add(val - val.transpose(1, 0, 2, 3))

@jax.jit
def add_IJ(target, val):
    """target inplace mutation of IJ permutation of val"""
    return target.at[:].add(val - val.transpose(0, 1, 3, 2))

@jax.jit
def add_AB_IJ(target, val):
    """
    target assignment of AB(IJ()) permutation of val

    val - val(ji) - val(ba) + val(ba, ji)
    """
    return target.at[:].add(
        (val - val.transpose(0, 1, 3, 2)) -
        (val.transpose(1, 0, 2, 3) - val.transpose(1, 0, 3, 2))
    )


def cond_sharding_constraint(tensor, shard):
    if shard is not None:
        return with_sharding_constraint(tensor, shard)
    return tensor

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


@partial(jax.jit, static_argnames=("shard_pphh",))
def t2_H2_ppph(H2, t1, t2, v_ppph, shard_pphh):
    indices, values = v_ppph
    idx_c, idx_d, idx_a, idx_k = indices
    pnum, hnum = t1.shape

    # Diagram 2
    term_2 = values[:, None] * t1[idx_a, :]  # (nnz, hnum)
    H2 = H2.at[idx_c, idx_d, :, idx_k].add(term_2)
    H2 = H2.at[idx_c, idx_d, idx_k, :].add(-term_2)  

    # Diagram 3
    d3_v = jnp.zeros((pnum, pnum)).at[idx_d, idx_a].add(values * t1[idx_c, idx_k])
    d3_int = jnp.einsum("da, dbij -> abij", d3_v, t2)
    d3_int = cond_sharding_constraint(d3_int, shard_pphh)
    H2 = add_AB(H2, -d3_int)

    # Diagram 4
    d4_v = (
        jnp.zeros_like(H2)
        .at[idx_a, idx_d, :, idx_k]
        .add(values[:, None] * t1[idx_c, :])
    )
    d4_int = jnp.einsum("acik, bcjk -> abij", d4_v, t2)
    d4_int = cond_sharding_constraint(d4_int, shard_pphh)
    H2 = add_AB_IJ(H2, d4_int)

    # Diagram 5
    d5_v = (
        jnp.zeros((pnum, hnum, hnum, hnum))
        .at[idx_a, :, :, idx_k]
        .add(values[:, None, None] * t2[idx_c, idx_d, :, :])
    )
    d5_int = jnp.einsum("bijk, ak -> abij", d5_v, t1)
    d5_int = cond_sharding_constraint(d5_int, shard_pphh)
    H2 = add_AB(H2, 0.5 * d5_int)

    # Diagram 6
    t1_c = t1[idx_c, :]
    t1_d = t1[idx_d, :]
    d6_v = (
        jnp.zeros((pnum, hnum, hnum, hnum))
        .at[idx_a, :, :, idx_k]
        .add(values[:, None, None] * (t1_c[:, :, None] * t1_d[:, None, :]))
    )
    d6_int = jnp.einsum("bijk, ak -> abij", d6_v, t1)
    d6_int = cond_sharding_constraint(d6_int, shard_pphh)
    H2 = add_AB_IJ(H2, 0.5 * d6_int)

    return H2

@jax.jit
def t2_H2_pppp(H2, t1, t2, v_pppp):
    ## v_pppp dgrams
    p_idx_a, p_idx_b, p_idx_c, p_idx_d = v_pppp[0]
    values = v_pppp[1]

    # Diagram 1: H2 += 0.5 * V[a,b,c,d] * T2[c,d,i,j]
    term_1 = values[:, None, None] * t2[p_idx_c, p_idx_d, :, :]
    H2 = H2.at[p_idx_a, p_idx_b, :, :].add(0.5 * term_1)

    # Diagram 2: H2 += 0.5 * pIJ( V[a,b,c,d] * T1[c,i] * T1[d,j] )
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

@partial(jax.jit, static_argnames=("shard_pphh",))
def t2_H2_dense_part1(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh):
    """Diagrams 1 to 5"""
    d1 = jnp.einsum("klij, abkl -> abij", v_hhhh, t2)
    d1 = cond_sharding_constraint(d1, shard_pphh)
    H2 = H2.at[:].add(0.5 * d1)
    
    d2 = jnp.einsum('bkcj, acik -> abij', v_phph, t2)
    d2 = cond_sharding_constraint(d2, shard_pphh)
    H2 = add_AB_IJ(H2, d2)
    
    d3 = jnp.einsum("bkij, ak -> abij", v_phhh, t1)
    d3 = cond_sharding_constraint(d3, shard_pphh)
    H2 = add_AB(H2, d3)
    
    d4_int = jnp.einsum("acik, cdkl -> adil", t2, v_pphh)
    d4_int = cond_sharding_constraint(d4_int, shard_pphh)
    d4 = jnp.einsum("adil, dblj -> abij", d4_int, t2)
    d4 = cond_sharding_constraint(d4, shard_pphh)
    H2 = add_AB_IJ(H2, 0.5 * d4)
    
    d5_int = jnp.einsum("cdij, cdkl -> ijkl", t2, v_pphh, optimize="optimal")
    d5 = jnp.einsum("ijkl, abkl -> abij", d5_int, t2)
    d5 = cond_sharding_constraint(d5, shard_pphh)
    H2 = H2.at[:].add(0.25 * d5)
    
    return H2

@partial(jax.jit, static_argnames=("shard_pphh", "shard_phph"))
def t2_H2_dense_part2(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph):
    """Diagrams 6 to 10"""
    d6_int = jnp.einsum("ak, klij -> alij", t1, v_hhhh)
    d6 = jnp.einsum("alij, bl -> abij", d6_int, t1)
    d6 = cond_sharding_constraint(d6, shard_pphh)
    H2 = add_AB(H2, 0.5 * d6)
    
    d7_int = jnp.einsum("cj, bkci -> bkji", t1, v_phph)
    d7 = jnp.einsum("ak, bkji -> abij", t1, d7_int)
    d7 = cond_sharding_constraint(d7, shard_pphh)
    H2 = add_AB_IJ(H2, -d7)
    
    d8_int = jnp.einsum("cikl, ck -> il", v_phhh, t1)
    d8 = jnp.einsum("il, ablj -> abij", d8_int, t2)
    d8 = cond_sharding_constraint(d8, shard_pphh)
    H2 = add_IJ(H2, -d8)
    
    d9_int = jnp.einsum("cikl, al -> ciak", v_phhh, t1)
    d9_int = cond_sharding_constraint(d9_int, shard_phph)
    d9 = jnp.einsum("ciak, bcjk -> abij", d9_int, t2)
    d9 = cond_sharding_constraint(d9, shard_pphh)
    H2 = add_AB_IJ(H2, -d9)
    
    d10_int = jnp.einsum("cjkl, ci -> jkli", v_phhh, t1) 
    d10 = jnp.einsum("jkli, abkl -> abij", d10_int, t2)
    d10 = cond_sharding_constraint(d10, shard_pphh)
    H2 = add_IJ(H2, 0.5 * d10)
    
    return H2

@partial(jax.jit, static_argnames=("shard_pphh", "shard_phph"))
def t2_H2_dense_part3(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph):
    """Diagrams 11 to 15"""
    d11_int1 = jnp.einsum("cjkl, ci -> jkli", v_phhh, t1) 
    d11_int2 = jnp.einsum("jkli, ak -> alij", d11_int1, t1) 
    d11 = jnp.einsum("alij, bl -> abij", d11_int2, t1)
    d11 = cond_sharding_constraint(d11, shard_pphh)
    H2 = add_AB_IJ(H2, 0.5 * d11)

    d12_int1 = jnp.einsum("cdkl, ci -> dkli", v_pphh, t1) 
    d12_int2 = jnp.einsum("dkli, dj -> klij", d12_int1, t1) 
    d12 = jnp.einsum("klij, abkl -> abij", d12_int2, t2)
    d12 = cond_sharding_constraint(d12, shard_pphh)
    H2 = add_IJ(H2, 0.25 * d12)

    d13_int1 = jnp.einsum("cdij, cdkl -> ijkl", t2, v_pphh, optimize="optimal") 
    d13_int2 = jnp.einsum("ijkl, ak -> ijal", d13_int1, t1) 
    d13 = jnp.einsum("ijal, bl -> abij", d13_int2, t1)
    d13 = cond_sharding_constraint(d13, shard_pphh)
    H2 = add_AB(H2, 0.25 * d13)

    d14_A = jnp.einsum("cdkl, ci -> dkli", v_pphh, t1)
    d14_B = jnp.einsum("adkj, dkli -> alij", t2, d14_A)
    d14 = jnp.einsum("alij, bl -> abij", d14_B, t1)
    d14 = cond_sharding_constraint(d14, shard_pphh)
    H2 = add_AB_IJ(H2, d14)

    d15_int1 = jnp.einsum("cdkl, ci -> dkli", v_pphh, t1) 
    d15_int2 = jnp.einsum("dkli, dj -> klij", d15_int1, t1) 
    d15_int3 = jnp.einsum("klij, ak -> alij", d15_int2, t1) 
    d15 = jnp.einsum("alij, bl -> abij", d15_int3, t1)
    d15 = cond_sharding_constraint(d15, shard_pphh)
    H2 = add_AB_IJ(H2, 0.25 * d15)

    return H2


def t2Iter(t1, t2, f_pp, f_ph, f_hh, v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph):
    H2 = v_pphh

    H2 = t2_H2_dense_part1(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh)
    H2 = t2_H2_dense_part2(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph)
    H2 = t2_H2_dense_part3(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph)
    
    H2 = t2_H2_ppph(H2, t1, t2, v_ppph, shard_pphh)
    H2 = t2_H2_pppp(H2, t1, t2, v_pppp)

    X_hh, X_pp = t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh)
    t2_new = t2_update(t2, X_hh, X_pp, H2)
    
    return t2_new
