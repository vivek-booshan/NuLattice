# TODO: shard or cast diis logic to cpu
from typing import Optional
from collections import deque

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

from NuLattice.utils._jax_types import Chef

from .amplitudes import t1Iter, t2Iter

@jax.jit
def ccsd_energy(f_ph, v_pphh, t2, t1):
    """
    Calculate the CCSD correlation energy.

    This function computes the electronic correlation energy contribution 
    from the singles (T1) and doubles (T2) amplitudes using the standard 
    Coupled Cluster energy expression.

    Parameters
    ----------
    f_ph : jax.Array
        The particle-hole block of the Fock matrix. Shape: (v, o).
    v_pphh : jax.Array
        The particle-particle-hole-hole block of the interaction potential. 
        Shape: (v, v, o, o).
    t2 : jax.Array
        The current doubles amplitudes. Shape: (v, v, o, o).
    t1 : jax.Array
        The current singles amplitudes. Shape: (v, o).

    Returns
    -------
    jax.Array
        The scalar correlation energy value.

    Notes
    -----
    The energy is calculated as:
    - e1: f_{ai} * t_{ai} (Singles contribution)
    - e2: 0.25 * v_{abij} * t_{abij} (Doubles contribution)
    - e3: 0.5 * v_{abij} * t_{ai} * t_{bj} (Singles-coupling contribution)
    """
    e_1 = jnp.einsum("ai,ai->", f_ph, t1)
    e_2 = 0.25 * jnp.einsum("abij,abij->", v_pphh, t2)
    e_3 = 0.5 * jnp.einsum("abij,ai,bj->", v_pphh, t1, t1)
    return e_1 + e_2 + e_3


@jax.jit
def t1Init(f_ph, f_pp, f_hh, delta):
    """
    Initialize the T1 amplitudes using the Moller-Plesset (MP2) guess.

    Parameters
    ----------
    f_ph, f_pp, f_hh : jax.Array
        Fock matrix slices.
    delta : float
        Energy shift parameter to avoid division by zero or regularize convergence.

    Returns
    -------
    jax.Array
        Initial guess for T1 amplitudes.
    """
    return f_ph / (delta + (-jnp.diag(f_pp)[:, None] + jnp.diag(f_hh)[None, :]))


@jax.jit
def t2Init(f_pp, f_hh, v_pphh, delta):
    """
    Initialize the T2 amplitudes using the Moller-Plesset (MP2) guess.

    Parameters
    ----------
    f_pp, f_hh : jax.Array
        Fock matrix slices used to build the energy denominator.
    v_pphh : jax.Array
        Interaction potential used as the numerator for the guess.
    delta : float
        Energy shift parameter.

    Returns
    -------
    jax.Array
        Initial guess for T2 amplitudes (MP2-like).
    """
    diag_h = jnp.diag(f_hh)
    diag_p = -jnp.diag(f_pp)

    return v_pphh / (
        delta
        + (
            diag_p[:, None, None, None]
            + diag_p[None, :, None, None]
            + diag_h[None, None, :, None]
            + diag_h[None, None, None, :]
        )
    )


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
    """
    Solver for the Coupled Cluster Singles and Doubles (CCSD) equations.

    This solver manages the iterative process, including amplitude updates, 
    energy evaluation, DIIS convergence acceleration, and distributed 
    data sharding across nodes/GPUs.

    Parameters
    ----------
    fock_mats : tuple of jax.Array
        Fock matrix blocks (f_pp, f_ph, f_hh).
    two_body_int : tuple of jax.Array
        Two-body interaction blocks (v_pppp, v_ppph, v_pphh, v_phph, v_phhh, v_hhhh).
    t1initial : jax.Array, optional
        Starting guess for T1 amplitudes. If None, uses T1Init.
    eps : float
        Convergence threshold for the energy difference between iterations.
    maxSteps : int
        Maximum number of iterations allowed.
    max_diis : int
        Number of previous amplitudes to store for DIIS extrapolation. 
        Set to 0 to disable DIIS.
    delta : float
        Energy shift added to the denominators.
    mixing : float
        Damping parameter for amplitude updates (0.5 = 50% new amplitude).
    verbose : bool
        If True, prints iteration information and energy to stdout.
    ccs : bool
        If True, restricts the calculation to Singles only (CCS).
    dtype : jnp.dtype
        Floating point precision (default: float64).
    chef : Chef, optional
        A distributed orchestration object used to shard and prepare 
        data across multiple devices.

    Returns
    -------
    energy : float
        Final correlation energy (scaled by physical units).
    t1 : jax.Array
        Converged singles amplitudes.
    t2 : jax.Array
        Converged doubles amplitudes.

    Notes
    -----
    The solver uses Direct Inversion in the Iterative Subspace (DIIS) to 
    accelerate convergence by finding a linear combination of previous 
    amplitudes that minimizes the norm of the residual vector.
    """
    f_pp, f_ph, f_hh = fock_mats
    v_pppp_sparse, v_ppph_sparse, v_pphh, v_phph, v_phhh, v_hhhh = two_body_int

    v_pppp = (v_pppp_sparse.indices.T, v_pppp_sparse.values)
    v_ppph = (v_ppph_sparse.indices.T, v_ppph_sparse.values)

    shard_pphh = None
    shard_phph = None
    if chef is not None:
        f_pp = chef.prepare(f_pp, rank=0)
        f_ph = chef.prepare(f_ph, rank=0)
        f_hh = chef.prepare(f_hh, rank=0)  # replicate

        v_pphh = chef.prepare(v_pphh)
        v_phph = chef.prepare(v_phph, spec=P("nodes", None, "gpus", None))
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

        shard_pphh = NamedSharding(chef.mesh, P("nodes", "gpus", None, None))
        shard_phph = NamedSharding(chef.mesh, P("nodes", None, "gpus", None))

    t1 = (
        t1Init(f_ph, f_pp, f_hh, delta)
        if t1initial is None
        else jnp.zeros_like(t1initial, dtype)  # possible source of memory issue
    )
    t2 = (
        jnp.zeros_like(v_pphh)  # zeros_like should shard like pphh
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
            t2_new = t2Iter(
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
            )
            t2 = t2 + mixing * (t2_new - t2)

        # NOTE: update t1 AFTER t2 updates
        t1 = t1 + mixing * (t1_new - t1)
        del t1_new, t2_new

        energy = ccsd_energy(f_ph, v_pphh, t2, t1)
        diff = abs(energy - prevEnergy) / max(1.0, abs(energy))

        if verbose and jax.process_index() == 0:
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
                            diis_t1[x + 1], diis_t1[x],
                            diis_t2[x + 1], diis_t2[x],
                            diis_t1[y + 1], diis_t1[y],
                            diis_t2[y + 1], diis_t2[y],
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
                diis_t1.append(t1_new_diis)
                diis_t2.append(t2_new_diis)

                # NOTE: if mem issue; move to np; jax.experimental.multihost_utils.process_allgather to append cpu data back to gpu

        if abs(energy) > 1e10 or jnp.isnan(energy):
            print("Diverged.")
            break

        prevEnergy = energy

    print("Max iterations reached.")
    return float(energy), t1, t2
