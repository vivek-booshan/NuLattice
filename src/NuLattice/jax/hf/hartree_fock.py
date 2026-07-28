from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import ShardingManager

from .subspace_solver import _occupied_orbitals

Array = jax.Array

def _adjoint(x):
    return jnp.swapaxes(jnp.conj(x), -1, -2)

def hermitianize(x):
    return 0.5 * (x + _adjoint(x))

@jax.jit
def contract_2nf_fused(indices: Array, values: Array, dens: Array) -> Array:
    """Contract the sparse two-body interaction with a one-body density."""
    p, q, r, s = (indices[:, i] for i in range(4))
    n = dens.shape[0]
    dtype = jnp.result_type(values.dtype, dens.dtype)
    res = jnp.zeros((n, n), dtype=dtype)
    res = res.at[p, r].add(+values * dens[q, s])
    res = res.at[q, r].add(-values * dens[p, s])
    res = res.at[p, s].add(-values * dens[q, r])
    res = res.at[q, s].add(+values * dens[p, r])
    return res


@jax.jit
def contract_3nf_fused(indices: Array, values: Array, dens: Array) -> Array:
    """Contract the sparse three-body interaction with two densities."""
    a, b, c, d, e, f = (indices[:, i] for i in range(6))
    n = dens.shape[0]
    dtype = jnp.result_type(values.dtype, dens.dtype)
    v2 = values * 2.0
    res = jnp.zeros((n, n), dtype=dtype)

    res = res.at[a, d].add(v2 * (dens[b, e] * dens[c, f] - dens[c, e] * dens[b, f]))
    res = res.at[b, d].add(v2 * (dens[c, e] * dens[a, f] - dens[a, e] * dens[c, f]))
    res = res.at[c, d].add(v2 * (dens[a, e] * dens[b, f] - dens[b, e] * dens[a, f]))

    res = res.at[a, e].add(v2 * (dens[b, f] * dens[c, d] - dens[c, f] * dens[b, d]))
    res = res.at[b, e].add(v2 * (dens[c, f] * dens[a, d] - dens[a, f] * dens[c, d]))
    res = res.at[c, e].add(v2 * (dens[a, f] * dens[b, d] - dens[b, f] * dens[a, d]))

    res = res.at[a, f].add(v2 * (dens[b, d] * dens[c, e] - dens[c, d] * dens[b, e]))
    res = res.at[b, f].add(v2 * (dens[c, d] * dens[a, e] - dens[a, d] * dens[c, e]))
    res = res.at[c, f].add(v2 * (dens[a, d] * dens[b, e] - dens[b, d] * dens[a, e]))
    return res

def build_mean_fields(
    dens: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array = None,
    w3_val: Array = None,
) -> tuple[Array, Array]:
    gamma = hermitianize(contract_2nf_fused(v2_idx, v2_val, dens))
    omega = None
    if (w3_idx is not None) and (w3_val is not None):
        omega = hermitianize(contract_3nf_fused(w3_idx, w3_val, dens))
    return gamma, omega


def build_fock(
    h1: Array,
    gamma: Array,
    omega: Array = None,
) -> Array:
    if omega is None:
        return hermitianize(h1 + gamma)
    return hermitianize(h1 + gamma + 0.5 * omega)

def hf_energy(
    dens: Array,
    h1: Array,
    gamma: Array,
    omega: Array = None,
) -> Array:

    e_h1 = jnp.einsum("ij,ji->", h1, dens)
    e_gamma = jnp.einsum("ij,ji->", gamma, dens)
    e_omega = jnp.asarray(0, dtype=jnp.real(dens[0]).dtype)
    if omega is not None:
        e_omega = jnp.einsum("ij,ji->", omega, dens)
    return jnp.real(e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega)

def init_density(nstat: int, hole: Tuple[int], dtype=None):
    dens = jnp.zeros((nstat, nstat), dtype=dtype)
    hole_indices = jnp.array(hole)
    dens = dens.at[hole_indices, hole_indices].set(1.0)
    return dens

@partial(jax.jit, static_argnames=("npart", "diagonalizer"))
def _scf_step(
    dens, h1, v2_idx, v2_val, w3_idx, w3_val, npart, mix, prev_vecs,
    diagonalizer,
):
    gamma, omega = build_mean_fields(dens, v2_idx, v2_val, w3_idx, w3_val)
    fock = build_fock(h1, gamma, omega)
    energy = hf_energy(dens, h1, gamma, omega)

    if diagonalizer == "dense":
        _, orbitals = jnp.linalg.eigh(fock)
        occ = orbitals[:, :npart]
    else:
        _, occ = _occupied_orbitals(fock, npart, prev_vecs)

    new_density = occ @ _adjoint(occ)

    residual_density = jnp.sum(jnp.abs(new_density - dens))
    mixed_density = (1.0 - mix) * dens + mix * new_density

    return occ, energy, mixed_density, residual_density

def prepare_inputs(op1, op2, op3, dens, sm: ShardingManager, dtype=jnp.float64):
    has_three_body = op3 is not None and len(op3) > 0

    if sm is not None:
        assert sm.num_nodes == 1 or sm.num_gpus == 1, "HF expects 1D mesh, ensure sm.num_nodes or sm.num_gpus is 1"
        h1 = sm.prepare(op1.to_dense(), rank=0)
        dens = sm.prepare(dens, rank=0)
        v2_idx = sm.prepare(op2.indices)
        v2_val = sm.prepare(op2.values)
        if has_three_body:
            w3_idx = sm.prepare(op3.indices)
            w3_val = sm.prepare(op3.values)
        else:
            w3_idx = None
            w3_val = None
    else:
        h1 = jnp.asarray(op1.to_dense())
        v2_idx = jnp.asarray(op2.indices)
        v2_val = jnp.asarray(op2.values)
        if has_three_body:
            w3_idx = jnp.asarray(op3.indices)
            w3_val = jnp.asarray(op3.values)
        else:
            w3_idx = None
            w3_val = None
        dens = jnp.asarray(dens)

    return h1, v2_idx, v2_val, w3_idx, w3_val, dens

def solve_HF(
    L,
    a_lat,
    op1,
    op2,
    op3,
    dens,
    mix=0.5,
    eps=1e-8,
    max_iter=100,
    verbose=False,
    sm: ShardingManager = None,
    diagonalizer="davidson",
):

    if diagonalizer not in {"davidson", "dense"}:
        raise ValueError("diagonalizer must be 'davidson' or 'dense'")

    h1_dense, v2_idx, v2_val, w3_idx, w3_val, _dens = prepare_inputs(
        op1, op2, op3, dens, sm
    )

    prev_energy = 0.0
    converged = False
    npart = int(jnp.real(jnp.trace(_dens)).round())

    occ = occupied_orbitals_from_diagonal_density(_dens, npart)

    for i in range(max_iter):
        occ, energy, _dens, diff_dens = _scf_step(
            _dens, h1_dense, v2_idx, v2_val, w3_idx, w3_val, npart, mix, occ,
            diagonalizer,
        )

        dE = jnp.abs(energy - prev_energy)

        if verbose:
            # convert to jax debug logging
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.6e}, dRho={diff_dens:.6e}")

        if (diff_dens < eps or dE < eps) and i > 1:
            converged = True
            break

        prev_energy = energy

    return energy, occ, converged

def occupied_orbitals_from_diagonal_density(dens: jax.Array, npart: int) -> jax.Array:
    indices = jnp.argsort(jnp.real(jnp.diag(dens)))[-npart:]
    return dens[:, indices]
