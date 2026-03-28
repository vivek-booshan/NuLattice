from typing import Tuple

import jax
import jax.numpy as jnp
from NuLattice.utils._jax_types import (
    Chef,
    OneBodyOperator,
    TwoBodyOperator,
    ThreeBodyOperator,
)
from functools import partial


def init_density(nstat: int, hole: Tuple[int]) -> jnp.array:
    dens = jnp.zeros((nstat, nstat))
    hole_indices = jnp.array(hole)
    dens = dens.at[hole_indices, hole_indices].set(1.0)
    return dens


@jax.jit
def contract_2nf(indices, values, dens):
    p, q, r, s = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
    n = dens.shape[0]
    res = jnp.zeros((n, n), dtype=dens.dtype)

    # Accumulate all 4 permutations (Direct and Exchange)
    res = res.at[p, r].add(values * dens[q, s])
    res = res.at[q, r].add(-values * dens[p, s])
    res = res.at[p, s].add(-values * dens[q, r])
    res = res.at[q, s].add(values * dens[p, r])
    return res


@jax.jit
def contract_3nf(indices, values, dens):
    a, b, c, d, e, f = [indices[:, i] for i in range(6)]
    n = dens.shape[0]
    res = jnp.zeros((n, n), dtype=dens.dtype)

    v2 = values * 2.0
    # Vectorized extraction of density elements
    rbe, rcf = dens[b, e], dens[c, f]
    rce, rbf = dens[c, e], dens[b, f]
    rae, raf = dens[a, e], dens[a, f]
    rbd, rcd, rad = dens[b, d], dens[c, d], dens[a, d]

    # JAX will fuse these additions into a single kernel
    res = res.at[a, d].add(v2 * (rbe * rcf - rce * rbf))
    res = res.at[b, d].add(v2 * (rce * raf - rae * rcf))
    res = res.at[c, d].add(v2 * (rae * rbf - rbe * raf))

    res = res.at[a, e].add(v2 * (rbf * rcd - rcf * rbd))
    res = res.at[b, e].add(v2 * (rcf * rad - raf * rcd))
    res = res.at[c, e].add(v2 * (raf * rbd - rbf * rad))

    res = res.at[a, f].add(v2 * (rbd * rce - rcd * rbe))
    res = res.at[b, f].add(v2 * (rce * raf - rae * rcf))
    res = res.at[c, f].add(v2 * (rad * rbe - rbd * rae))
    return res


@partial(jax.jit, static_argnames=("npart",))
def _hf_step(dens, h1, v2_idx, v2_val, w3_idx, w3_val, npart, mix):
    gamma = contract_2nf(v2_idx, v2_val, dens)
    omega = contract_3nf(w3_idx, w3_val, dens)

    hf_ham = h1 + gamma + 0.5 * omega

    e_h1 = jnp.sum(h1 * dens)
    e_gamma = jnp.sum(gamma * dens)
    e_omega = jnp.sum(omega * dens)
    energy = e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega

    vals, vecs = jnp.linalg.eigh(hf_ham)
    occ = vecs[:, :npart]
    new_dens = occ @ occ.T

    diff_dens = jnp.sum(jnp.abs(new_dens - dens))

    updated_dens = (1.0 - mix) * dens + mix * new_dens

    return updated_dens, energy, diff_dens, vecs


def solve_HF(
    op1: OneBodyOperator,
    op2: TwoBodyOperator,
    op3: ThreeBodyOperator,
    dens: jnp.array,
    mix: float = 0.5,
    eps: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = False,
    chef: Chef = None,
):
    if chef is not None:
        h1_dense = chef.prepare_op_dense(op1)
        op2 = chef.prepare_operator(op2)
        op3 = chef.prepare_operator(op3)
        _dens = chef.shard_array(dens)
        v2_idx, v2_val = op2.indices, op2.data
        w3_idx, w3_val = op3.indices, op3.data
    else:
        h1_dense = op1.to_dense()
        op2 = op2.to_bcoo()
        op3 = op3.to_bcoo()
        _dens = dens
        v2_idx, v2_val = op2.indices, op2.data
        w3_idx, w3_val = op3.indices, op3.data


    npart = int(jnp.trace(_dens).round())
    prev_energy = 0.0
    converged = False

    for i in range(max_iter):
        _dens, energy, diff_dens, vecs = _hf_step(
            _dens, h1_dense, v2_idx, v2_val, w3_idx, w3_val, npart, mix
        )

        dE = jnp.abs(energy - prev_energy)
        if verbose:
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.4e}, dRho={diff_dens:.4e}")

        if (diff_dens < eps or dE < 1e-12) and i > 1:
            converged = True
            break

        prev_energy = energy

    return float(energy), vecs, converged
