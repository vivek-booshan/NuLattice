import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial

from NuLattice.utils._jax_types import TwoBodyOperator, Chef

from . import ccDgrams as dgrams


def to_tensor(arr, dtype=jnp.float64):
    """Helper to convert numpy arrays/lists/Operators to JAX arrays."""
    if isinstance(arr, jnp.ndarray):
        return arr.astype(dtype)
        # return jnp.array(arr, dtype=dtype)
    # FIXME(vivek): base class operator has to_dense so this will always run
    if hasattr(arr, "to_dense"):  # Handle OneBodyOperator
        return jnp.array(arr.to_dense(), dtype=dtype)
    return jnp.array(arr, dtype=dtype)


def to_soa_sparse(sparse_input, dtype=jnp.float64):
    """
    Extracts SoA tensors from a TwoBodyOperator for diagrammatic contractions.
    :return: (indices, values) formatted for ccDgrams
    """
    if isinstance(sparse_input, TwoBodyOperator):
        op = sparse_input
    else:
        op = TwoBodyOperator.from_list(sparse_input, nstat=0)

    if len(op) == 0:
        return (
            jnp.empty((4, 0), dtype=jnp.int32),
            jnp.empty((0,), dtype=dtype),
        )

    # ccDgrams kernels expect (4, N) indices. Operator stores (N, 4).
    indices = op.indices.T.astype(jnp.int32)
    values = op.values.astype(dtype)

    return indices, values


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


@partial(jax.jit, static_argnames=["sparse"])
def t1Iter(t1, t2, f_ph, f_pp, f_hh, v_phph, v_phhh, v_pphh, v_ppph, sparse=True):
    H1 = f_ph + dgrams.dgram_akci_ck(v_phph, t1)
    H1 += dgrams.dgram_ck_acik(f_ph, t2)
    H1 += dgrams.dgram_cikl_cakl(v_phhh, t2)
    H1 += dgrams.dgram_cdkl_ck_dali(v_pphh, t1, t2)

    X_hh = -f_hh + dgrams.dgram_ck_ci(f_ph, t1)
    X_pp = f_pp + dgrams.dgram_ck_ak(f_ph, t1)

    X_hh += dgrams.dgram_bijk_bj(v_phhh, t1)
    X_hh += dgrams.dgram_cdlk_cdli(v_pphh, t2)
    X_pp += dgrams.dgram_dckl_dakl(v_pphh, t2)
    X_hh += dgrams.dgram_cdlk_cl_di(v_pphh, t1)
    X_pp += dgrams.dgram_cdkl_dk_al(v_pphh, t1)

    if sparse:
        H1 += v_ppph[0]
        X_pp += v_ppph[1]
    else:
        H1 += -0.5 * jnp.einsum("cdak, cdki -> ai", v_ppph, t2)
        X_pp -= jnp.einsum("cdak, ck -> ad", v_ppph, t1)

    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    diag_h = jnp.diag(X_hh)
    diag_p = jnp.diag(X_pp)
    denom = -(diag_p[:, None] + diag_h[None, :])

    return t1 + (H1 / denom)


@partial(jax.jit, static_argnames=["sparse"])
def t2Iter(
    t1,
    t2,
    f_ph,
    f_hh,
    f_pp,
    v_pppp,
    v_phph,
    v_phhh,
    v_pphh,
    v_ppph,
    v_hhhh,
    sparse=True,
):
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

    X_hh = -f_hh + dgrams.dgram_cdkl_cdjl(v_pphh, t2)
    X_pp = f_pp + dgrams.dgram_cdkl_bdkl(v_pphh, t2)

    X_pp += dgrams.dgram_ck_bk(f_ph, t1)
    X_hh += dgrams.dgram_ck_cj(f_ph, t1)
    X_hh += dgrams.dgram_cdlk_cl_dj(v_pphh, t1)
    X_pp += dgrams.dgram_cdlk_dk_bl(v_pphh, t1)

    if sparse:
        H2 += dgrams.pIJ(v_ppph[2])
        H2 += dgrams.dgram_da_dbij(v_ppph[3], t2)
        H2 += dgrams.dgram_acik_bcjk(v_ppph[4], t2)
        H2 += dgrams.dgram_bijk_ak1(v_ppph[5], t1)
        H2 += dgrams.dgram_bijk_ak2(v_ppph[6], t1)

        ret1, ret2 = dgrams.v_pppp_dgrams(v_pppp, t1, t2)
        H2 += 0.5 * ret1
        H2 += 0.5 * dgrams.pIJ(ret2)
    else:
        H2 += dgrams.pIJ(jnp.einsum("abcj, ci -> abij", v_ppph, t1))
        H2 += -dgrams.pAB(jnp.einsum("cdak, ck, dbij -> abij", v_ppph, t1, t2))
        H2 += dgrams.pIJ(
            dgrams.pAB(jnp.einsum("dcak, di, bcjk -> abij", v_ppph, t1, t2))
        )
        H2 += 0.5 * dgrams.pAB(jnp.einsum("cdbk, ak, cdij -> abij", v_ppph, t1, t2))
        H2 += 0.5 * dgrams.pIJ(
            dgrams.pAB(jnp.einsum("cdbk, ci, ak, dj -> abij", v_ppph, t1, t1, t1))
        )

        H2 += 0.5 * jnp.einsum("abcd, cdij -> abij", v_pppp, t2)
        H2 += 0.5 * dgrams.pIJ(jnp.einsum("abcd, ci, dj -> abij", v_pppp, t1, t1))

    H2 += dgrams.pAB(jnp.einsum("bc, acij -> abij", X_pp, t2))
    H2 += dgrams.pIJ(jnp.einsum("kj, abik -> abij", X_hh, t2))

    diag_h = jnp.diag(X_hh)
    diag_p = jnp.diag(X_pp)
    denom_hh = diag_h[None, :] + diag_h[:, None]
    denom_pp = diag_p[None, :] + diag_p[:, None]

    return t2 + (
        H2
        / -(
            denom_pp[:, :, jnp.newaxis, jnp.newaxis]
            + denom_hh[jnp.newaxis, jnp.newaxis, :, :]
        )
    )


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
    sparse=True,
    ccs=False,
    dtype=jnp.float64,
    chef: Chef = None,
):
    f_pp, f_ph, f_hh = [to_tensor(f, dtype) for f in fock_mats]

    v_pppp_in, v_ppph_in, v_pphh, v_phph, v_phhh, v_hhhh = [
        to_tensor(x, dtype) if not hasattr(x, "indices") else x for x in two_body_int
    ]

    if sparse:
        v_pppp = to_soa_sparse(v_pppp_in, dtype)
        v_ppph = to_soa_sparse(v_ppph_in, dtype)
    else:
        v_pppp, v_ppph = v_pppp_in, v_ppph_in

    if chef is not None:
        mesh = chef.mesh

        f_pp = chef.prepare(f_pp)
        f_ph = chef.prepare(f_ph)
        f_hh = chef.prepare(f_hh, rank=0)  # replicate

        v_pphh = chef.prepare(v_pphh)
        v_phph = chef.prepare(v_phph)
        v_phhh = chef.prepare(v_phhh)
        v_hhhh = chef.prepare(v_hhhh, rank=0)  # replicate

        if not sparse:
            v_pppp = chef.prepare(v_pppp)
            v_ppph = chef.prepare(v_ppph)
        else:
            idx_sharding = NamedSharding(mesh, P(None, "data"))
            val_sharding = NamedSharding(mesh, P("data"))

            # NOTE(vivek): to_soa_sparse may emit tensor of size zero, avoid sharding that
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
        else to_tensor(t1initial, dtype)
    )
    t2 = (
        jnp.zeros_like(v_pphh)
        if (ccs or t1initial is not None)
        else t2Init(f_pp, f_hh, v_pphh, delta)
    )

    if max_diis > 0:
        diis_t1 = [t1]
        diis_t2 = [t2]
        diis_errors = []

    prevEnergy = ccsd_energy(f_ph, v_pphh, t2, t1)
    if verbose:
        print(f"Step 0: {prevEnergy}")

    for i in range(maxSteps):
        oldT1, oldT2 = t1, t2

        v_ppph_results = dgrams.v_ppph_dgrams(v_ppph, t1, t2) if sparse else v_ppph

        t1_new = t1Iter(
            t1,
            t2,
            f_ph,
            f_pp,
            f_hh,
            v_phph,
            v_phhh,
            v_pphh,
            v_ppph_results,
            sparse=sparse,
        )
        t1 = t1 + mixing * (t1_new - t1)

        if not ccs:
            t2_new = t2Iter(
                oldT1,
                t2,
                f_ph,
                f_hh,
                f_pp,
                v_pppp,
                v_phph,
                v_phhh,
                v_pphh,
                v_ppph_results,
                v_hhhh,
                sparse=sparse,
            )
            t2 = t2 + mixing * (t2_new - t2)

        energy = ccsd_energy(f_ph, v_pphh, t2, t1)
        diff = abs(energy - prevEnergy) / max(1.0, abs(energy))

        if verbose:
            print(f"Step {i + 1}: {energy} difference = {diff}")

        if diff < eps:
            return float(energy), t1, t2

        if max_diis > 0:
            diis_t1.append(t1)
            diis_t2.append(t2)

            # STORE AS NATIVE TUPLES. Do not use flatten/reshape(-1)!
            # Flattening breaks GSPMD layout and causes Out-Of-Memory.
            diis_errors.append((t1 - oldT1, t2 - oldT2))

            if len(diis_errors) > max_diis:
                diis_t1.pop(0)
                diis_t2.pop(0)
                diis_errors.pop(0)

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
                        t1_new_diis += c[k] * diis_t1[k + 1]
                        if not ccs:
                            t2_new_diis += c[k] * diis_t2[k + 1]

                    t1, t2 = t1_new_diis, t2_new_diis
                except Exception:
                    pass

                diis_t1 = [t1]
                diis_t2 = [t2]
                diis_errors = []

        if abs(energy) > 1e10 or jnp.isnan(energy):
            print("Diverged.")
            break

        prevEnergy = energy

    print("Max iterations reached.")
    return float(energy), t1, t2
