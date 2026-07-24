# TODO: align mesh and contraction indices optimally
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
    # NOTE: must be (a - b) - (c - d)!!! otherwise memory err holding all 4
    return target.at[:].add(
        (val - val.transpose(0, 1, 3, 2))
        - (val.transpose(1, 0, 2, 3) - val.transpose(1, 0, 3, 2))
    )


def cond_sharding_constraint(tensor, shard):
    if shard is not None:
        return with_sharding_constraint(tensor, shard)
    return tensor


@jax.jit
def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph):
    """
    Perform a single iteration update for the T1 (singles) amplitudes.

    This function constructs the T1 residual (H1) and the effective Fock
    intermediates (X_pp, X_hh) to solve the T1 amplitude equation using
    a Jacobi-like update.

    Parameters
    ----------
    t1 : jax.Array
        Singles amplitudes. Shape: (p,h).
    t2 : jax.Array
        Doubles amplitudes. Shape: (p,p,h,h).
    f_ph, f_pp, f_hh : jax.Array
        Slices of the Fock matrix (PH, PP, and HH).
    v_phph, v_phhh, v_pphh : jax.Array
        Dense interaction potential slices.
    v_ppph : tuple of (indices, values)
        Sparse representation of the PPPH interaction.
        `indices` is (c, d, a, k), where 'a' is the target particle index.

    Returns
    -------
    jax.Array
        The updated T1 amplitudes.

    Notes
    -----
    - The function constructs effective 1-body intermediates $X_{pp}$ and $X_{hh}$
      which include correlations from the 2-body interaction.
    - Sparse contribution: $H1_{ai} -= 0.5 \\sum_{cdk} V_{cdak} T2_{cdki}$.
    - The final step performs an energy denominator division for convergence.
    """
    indices, values = v_ppph
    idx_c, idx_d, idx_a, idx_k = indices

    H1 = f_ph
    H1 -= jnp.einsum("akci, ck -> ai", v_phph, t1)
    H1 += jnp.einsum("ck, acik -> ai", f_ph, t2)
    H1 -= 0.5 * jnp.einsum("cikl, cakl -> ai", v_phhh, t2)
    H1 += jnp.einsum("cdkl, ck, dali -> ai", v_pphh, t1, t2)

    # v_ppph dgram
    # Diagram h1[a, i] -= 0.5 * sum_{cdk} V[c,d,a,k] * T2[c,d,k,i]
    H1 = H1.at[idx_a, :].add(-0.5 * values[:, None] * t2[idx_c, idx_d, idx_k, :])

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
    X_pp = X_pp.at[idx_a, idx_d].add(-values * t1[idx_c, idx_k])

    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    return t1 - (H1 / (jnp.diag(X_pp)[:, None] + jnp.diag(X_hh)[None, :]))


@jax.jit
def t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh):
    """
    Construct the effective 1-body intermediates (X_hh, X_pp) for the T2 update.

    These intermediates (sometimes called "chi" or "Fock-like" intermediates)
    renormalize the occupied and virtual orbital energies with information
    from the T-amplitudes and the 2-body potential.

    Parameters
    ----------
    t1 : jax.Array
        Singles amplitudes.
    t2 : jax.Array
        Doubles amplitudes.
    f_pp, f_ph, f_hh : jax.Array
        Fock matrix slices.
    v_pphh : jax.Array
        The Particle-Particle-Hole-Hole potential slice.

    Returns
    -------
    X_hh : jax.Array
        Effective hole-hole intermediate. Shape: (h, h).
    X_pp : jax.Array
        Effective particle-particle intermediate. Shape: (p, p).
    """
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
    """
    Compute T2 residual contributions from the sparse PPPH interaction.

    This function implements specific diagrammatic contributions where
    three particles and one hole are involved in the interaction vertex.

    Parameters
    ----------
    H2 : jax.Array
        Current T2 residual tensor.
    t1 : jax.Array
        Singles amplitudes.
    t2 : jax.Array
        Doubles amplitudes.
    v_ppph : tuple of (indices, values)
        Sparse PPPH potential. Indices: (c, d, a, k).
    shard_pphh : bool
        Static sharding flag.

    Returns
    -------
    jax.Array
        Updated T2 residual.

    Notes
    -----
    - Diagram 2: Direct coupling of V and T1 into the ppph sector.
    - Diagram 3: Modification of T2 particle lines by a V*T1 intermediate.
    - Diagram 4: Ring-like coupling between a sparse vertex and T2.
    - Diagram 5 & 6: Triple-excitation mimics where T1 and T2 are
      coordinated by the sparse vertex.
    """
    indices, values = v_ppph
    idx_c, idx_d, idx_a, idx_k = indices
    pnum, hnum = t1.shape

    # Diagram 2
    # NOTE: cannot do permutation cuz term_2.ndim == 2
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
    """
    Compute T2 residual contributions from the sparse PPPP interaction.

    Handles the "all-particle" scattering sector, involving particle-particle
    ladders and non-linear T1 contributions.

    Parameters
    ----------
    H2 : jax.Array
        Current T2 residual tensor.
    t1, t2 : jax.Array
        Cluster amplitudes.
    v_pppp : tuple of (indices, values)
        Sparse PPPP potential. Indices: (a, b, c, d).

    Returns
    -------
    jax.Array
        Updated T2 residual.

    Notes
    -----
    - Diagram 1: $H2_{abij} += 0.5 \\sum_{cd} V_{abcd} T2_{cdij}$ (PP-Ladder).
      Describes pairs of excited electrons scattering into different virtual states.
    - Diagram 2: $H2_{abij} += 0.5 P(ij) \\sum_{cd} V_{abcd} T1_{ci} T1_{dj}$.
      Describes two independent single excitations interacting to mimic a double.
    """
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
def t2_final_step(t2, X_hh, X_pp, H2):
    """
    Finalize the T2 amplitude update using the Jacobi method with intermediates.

    This function applies the effective 1-body terms to the residual and
    performs the multi-dimensional energy denominator division.

    Parameters
    ----------
    t2 : jax.Array
        Current T2 amplitudes.
    X_hh : jax.Array
        Effective hole-hole intermediate (X_kj).
    X_pp : jax.Array
        Effective particle-particle intermediate (X_bc).
    H2 : jax.Array
        The fully constructed T2 residual (sum of all diagrams).

    Returns
    -------
    jax.Array
        The updated T2 amplitudes for the next iteration.

    Notes
    -----
    The update uses the 4-index denominator:
    $\\Delta_{abij} = \\epsilon_a + \\epsilon_b - \\epsilon_i - \\epsilon_j$
    where $\\epsilon$ are the diagonal elements of the X-intermediates.
    """
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
    """
    Compute Diagrams 1 through 5 of the CCSD T2 amplitude residual.

    This function handles the primary two-body contractions where the
    interaction potential couples directly to T2 or via a single T1
    bridge. It includes the hole-hole and particle-particle ladder
    terms.

    Parameters
    ----------
    H2 : jax.Array
        The T2 intermediate tensor (residual) to be updated.
        Shape: (pphh).
    t1 : jax.Array
        Cluster amplitudes for single excitations.
        Shape: (ph).
    t2 : jax.Array
        Cluster amplitudes for double excitations.
        Shape: (pphh).
    v_pphh, v_phph, v_phhh, v_hhhh : jax.Array
        Two body interactions for pphh, phph, phhh, and hhhh, respectively.
    shard_pphh : NamedSharding
        sharding configuration to apply sharding constraints
        to the pphh-sector tensors during JIT compilation.

    Returns
    -------
    jax.Array
        The updated H2 intermediate with contributions from diagrams 1-5.

    Notes
    -----
    The diagrams implemented are:

    - D1: 0.5 * v_{klij} * t_{abkl} (Hole-Hole Ladder)
      Scattering of two holes into new hole states; describes the
      interaction of the "voids" left in the Fermi sea.
    - D2: P(ab)P(ij) [ v_{bkcj} * t_{acik} ] (Particle-Hole Ring)
      A "bubble" diagram where a particle and a hole exchange state;
      responsible for screen-like polarization effects.
    - D3: P(ab) [ v_{bkij} * t_{ak} ] (Single-to-Double Coupling)
      A three-hole, one-particle potential interaction colliding with
      a single excitation (T1) to create a double excitation.
    - D4: 0.5 * P(ab)P(ij) [ t_{acik} * v_{cdkl} * t_{dblj} ] (PP-Ladder)
      Two double excitations linked by a particle-particle interaction;
      describes pairs of excited electrons scattering off each other.
    - D5: 0.25 * t_{cdij} * v_{cdkl} * t_{abkl} (HH-Ladder Intermediate)
      Two double-excitations "knitted" together via the interaction
      of their respective occupied-space (hole) components.
    """
    d1 = jnp.einsum("klij, abkl -> abij", v_hhhh, t2)
    d1 = cond_sharding_constraint(d1, shard_pphh)
    H2 = H2.at[:].add(0.5 * d1)

    d2 = jnp.einsum("bkcj, acik -> abij", v_phph, t2)
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

    d5_int = jnp.einsum("cdij, cdkl -> ijkl", t2, v_pphh)
    d5 = jnp.einsum("ijkl, abkl -> abij", d5_int, t2)
    d5 = cond_sharding_constraint(d5, shard_pphh)
    H2 = H2.at[:].add(0.25 * d5)

    return H2


@partial(jax.jit, static_argnames=("shard_pphh", "shard_phph"))
def t2_H2_dense_part2(
    H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph
):
    """
    Compute Diagrams 6 through 10 of the CCSD T2 amplitude residual.

    This function focuses on mixed T1-T2 contributions and terms where the
    T1 amplitude modifies the hole lines of the two-body interaction.

    Parameters
    ----------
    H2 : jax.Array
        The T2 intermediate tensor. Shape: (pphh).
    t1 : jax.Array
        Cluster amplitudes for single excitations.
    t2 : jax.Array
        Cluster amplitudes for double excitations.
    v_pphh, v_phph, v_phhh, v_hhhh : jax.Array
    shard_pphh : NamedSharding
        Sharding constraints on the pphh sector.
    shard_phph : NamedSharding
        Sharding constraints on the phph sector.

    Returns
    -------
    jax.Array
        The updated H2 intermediate with contributions from diagrams 6-10.

    Notes
    -----
    The diagrams implemented are:
    - D6: 0.5 * P(ab) [ t_{ak} * v_{klij} * t_{bl} ] (Double Single-Hole Scattering)
      Two separate T1 excitations "finding each other" via a
      hole-hole scattering event in the potential.
    - D7: -P(ab)P(ij) [ t_{ak} * v_{bkci} * t_{cj} ] (Single-Excitation Ring)
      Two T1 amplitudes coupled by a particle-hole interaction,
      creating a loop that contributes to the double-excitation.
    - D8: -P(ij) [ v_{cikl} * t_{ck} * t_{ablj} ] (Occupied Line Modification)
      T1 modifies a hole line before it participates in a T2
      double excitation; a form of orbital renormalization.
    - D9: -P(ab)P(ij) [ v_{cikl} * t_{al} * t_{bcjk} ] (Particle-Hole Exchange)
      A single and double excitation cross-linked by a potential
      term that swaps a particle and a hole index.
    - D10: 0.5 * P(ij) [ v_{cjkl} * t_{ci} * t_{abkl} ] (Hole Line Renormalization)
      A T1 excitation "plugging" one of the hole lines of a
      double excitation through a three-hole interaction vertex.
    """
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
def t2_H2_dense_part3(
    H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph
):
    """
    Compute Diagrams 11 through 15 of the CCSD T2 amplitude residual.

    This function handles the highest-order non-linear contributions,
    specifically terms that are cubic (T1^3) and quartic (T1^4) in
    the single-excitation amplitudes.

    Parameters
    ----------
    H2 : jax.Array
        The T2 intermediate tensor. Shape: (n_virt, n_virt, n_occ, n_occ).
    t1 : jax.Array
        Cluster amplitudes for single excitations.
    t2 : jax.Array
        Cluster amplitudes for double excitations.
    v_pphh, v_phph, v_phhh, v_hhhh : jax.Array
    shard_pphh : NamedSharding
        Sharding constraints on the pphh sector.
    shard_phph : NamedSharding
        Sharding constraints on the phph sector.

    Returns
    -------
    jax.Array
        The updated H2 intermediate with contributions from diagrams 11-15.

    Notes
    -----
    These diagrams are computed via factorization into intermediates to
    avoid XLA gather calls.

    - D11: 0.5 * P(ab)P(ij) [ v_{cjkl} * t_{ci} * t_{ak} * t_{bl} ] (Triple Singles)
      Three separate T1 amplitudes coordinated by a potential interacting
      with one particle and three hole states.
    - D12: 0.25 * P(ij) [ v_{cdkl} * t_{ci} * t_{dj} * t_{abkl} ] (Double-Singles Hole Coupling)
      Two T1 excitations modifying the hole lines of a T2 amplitude
      via a particle-particle interaction.
    - D13: 0.25 * P(ab) [ t_{cdij} * v_{cdkl} * t_{ak} * t_{bl} ] (Double-Singles Particle Coupling)
      Two T1 amplitudes modifying the particle lines of a T2
      double excitation via particle-particle scattering.
    - D14: P(ab)P(ij) [ t_{adkj} * v_{cdkl} * t_{ci} * t_{bl} ] (Braid Interaction)
      A double excitation and two single excitations "braided"
      together by the particle-particle interaction potential.
    - D15: 0.25 * P(ab)P(ij) [ v_{cdkl} * t_{ci} * t_{dj} * t_{ak} * t_{bl} ] (Quartic Singles)
      The highest order term; four T1 amplitudes coordinated by one
      two-body interaction to produce a double-excitation effect.
    """
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

    d13_int1 = jnp.einsum("cdij, cdkl -> ijkl", t2, v_pphh)
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


# NOTE: must not jit to avoid bad XLA --> big mallocs
# split kernels to keep it tight and clean
def t2Iter(
    t1,
    t2,
    f_pp,
    f_ph,
    f_hh,
    v_pppp,
    v_ppph,
    v_pphh,
    v_phph,
    v_phhh,
    v_hhhh,
    shard_pphh,
    shard_phph,
):
    H2 = v_pphh

    H2 = t2_H2_dense_part1(H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh)
    H2 = t2_H2_dense_part2(
        H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph
    )
    H2 = t2_H2_dense_part3(
        H2, t1, t2, v_pphh, v_phph, v_phhh, v_hhhh, shard_pphh, shard_phph
    )

    H2 = t2_H2_ppph(H2, t1, t2, v_ppph, shard_pphh)
    H2 = t2_H2_pppp(H2, t1, t2, v_pppp)

    X_hh, X_pp = t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh)
    t2_new = t2_final_step(t2, X_hh, X_pp, H2)

    return t2_new
