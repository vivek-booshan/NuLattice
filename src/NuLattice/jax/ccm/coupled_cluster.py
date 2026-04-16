from typing import Optional
from collections import deque

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import Chef

from .amplitudes import t1Iter, t2_X, t2_H2, t2_update

# TODO: shard/handle t1 and t2 initialization

@jax.jit
def ccsd_energy(f_ph, v_pphh, t2, t1):
    e_1 = jnp.einsum("ai,ai->", f_ph, t1)
    e_2 = 0.25 * jnp.einsum("abij,abij->", v_pphh, t2)
    e_3 = 0.5 * jnp.einsum("abij,ai,bj->", v_pphh, t1, t1)
    return e_1 + e_2 + e_3


@jax.jit
def t1Init(f_ph, f_pp, f_hh, delta):
    return f_ph / (delta + (
        - jnp.diag(f_pp)[:, None]
        + jnp.diag(f_hh)[None, :]
    ))


@jax.jit
def t2Init(f_pp, f_hh, v_pphh, delta):
    diag_h = jnp.diag(f_hh)
    diag_p = -jnp.diag(f_pp)

    return v_pphh / (delta + (
        diag_p[:, None, None, None] + 
        diag_p[None, :, None, None] + 
        diag_h[None, None,: , None] + 
        diag_h[None, None, None, :]
    ))

@jax.jit
def error_dot(t1_x_next, t1_x, t2_x_next, t2_x, t1_y_next, t1_y, t2_y_next, t2_y):
    e1x = t1_x_next - t1_x
    e2x = t2_x_next - t2_x
    
    e1y = t1_y_next - t1_y
    e2y = t2_y_next - t2_y
    
    return jnp.sum(e1x * e1y) + jnp.sum(e2x * e2y)

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
    chef: Optional[Chef] = None,
):

    f_pp, f_ph, f_hh = fock_mats
    v_pppp_sparse, v_ppph_sparse, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int

    v_pppp = (v_pppp_sparse.indices.T, v_pppp_sparse.values)
    v_ppph = (v_ppph_sparse.indices.T, v_ppph_sparse.values)

    if chef is not None:
        f_pp = chef.prepare(f_pp)
        f_ph = chef.prepare(f_ph)
        f_hh = chef.prepare(f_hh, rank=0)  # replicate

        v_pphh = chef.prepare(v_pphh)
        v_phph = chef.prepare(v_phph)
        v_phhh = chef.prepare(v_phhh)
        v_hhhh = chef.prepare(v_hhhh, rank=0)  # replicate

        v_pppp = (
            chef.prepare(v_pppp[0], rank=0),
            chef.prepare(v_pppp[1], rank=0),
        )

        v_ppph = (
            chef.prepare(v_ppph[0], rank=0),
            chef.prepare(v_ppph[1], rank=0),
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
        diis_t1 = deque(maxlen=max_diis + 1)
        diis_t2 = deque(maxlen=max_diis + 1)
        diis_t1.append(t1)
        diis_t2.append(t2)

    prevEnergy = ccsd_energy(f_ph, v_pphh, t2, t1)
    if verbose:
        print(f"Step 0: {prevEnergy}")

    for step in range(maxSteps):
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

        if not ccs:
            X_hh, X_pp = t2_X(t1, t2, f_pp, f_ph, f_hh, v_pphh)
            X_pp.block_until_ready()  # force intermediate dealloc

            H2 = t2_H2(
                t1,
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

        # NOTE: update t1 AFTER t2 updates 
        t1 = t1 + mixing * (t1_new - t1)

        energy = ccsd_energy(f_ph, v_pphh, t2, t1)
        diff = abs(energy - prevEnergy) / max(1.0, abs(energy))

        if verbose:
            print(f"Step {step + 1}: {energy} difference = {diff}")

        if diff < eps:
            return float(energy), t1, t2

        # NOTE: end of physics step
        # below is DIIS logic

        if max_diis > 0:
            diis_t1.append(t1)
            diis_t2.append(t2)

            if len(diis_t1) == max_diis + 1:
                size = max_diis
                B = jnp.zeros((size, size), dtype=dtype)

                for x in range(size):
                    for y in range(x, size):
                        val = error_dot(
                            diis_t1[x+1], diis_t1[x],
                            diis_t2[x+1], diis_t2[x],
                            diis_t1[y+1], diis_t1[y],
                            diis_t2[y+1], diis_t2[y],
                        )

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

                diis_t1.clear()
                diis_t2.clear()
                diis_t1.append(t1)
                diis_t2.append(t2)

        if abs(energy) > 1e10 or jnp.isnan(energy):
            print("Diverged.")
            break

        prevEnergy = energy

    print("Max iterations reached.")
    return float(energy), t1, t2
