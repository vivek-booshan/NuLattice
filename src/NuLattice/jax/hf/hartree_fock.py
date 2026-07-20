from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import ShardingManager

@jax.jit
def _local_orthonormalize(V):
    # Compute the small overlap matrix (2k x 2k). 
    S = jnp.dot(V.T, V)  # calls (cheap) AllReduce on mesh
    S += 1e-11 * jnp.eye(S.shape[0], dtype=S.dtype)
    L = jnp.linalg.cholesky(S)
    L_inv = jnp.linalg.inv(L)
    return jnp.dot(V, L_inv.T)

@partial(jax.jit, static_argnames=("npart", ))
def davidson_eigh(H, npart, guess_vecs, max_iter=10):
    """
    Finds the lowest `npart` eigenvalues/eigenvectors of a sharded dense Hamiltonian H.
    
    Args:
        H: Sharded Hamiltonian matrix of shape (nstat, nstat)
        npart: Number of occupied states (lowest roots needed)
        guess_vecs: Initial guess vectors of shape (nstat, npart) from previous SCF step
        max_iter: Number of subspace expansion steps (try 3-5 for warm starts)

    Frankensteined from https://joshuagoings.com/2013/08/23/davidsons-method/
    """
    nstat = H.shape[0]
    D = jnp.diag(H) # Extract diagonal for the preconditioner
    
    # Initialize a static subspace V of size (nstat, 2 * npart)
    V = jnp.zeros((nstat, 2 * npart), dtype=H.dtype)
    V = V.at[:, :npart].set(guess_vecs)
    V = _local_orthonormalize(V)
    
    def body_fun(i, state):
        V_sub, _ = state
        
        # Project into subspace: M = VT H V -> (2k, 2k)
        HV = jnp.dot(H, V_sub)
        M = jnp.dot(V_sub.T, HV)
        
        # local eigen solution
        vals, evecs = jnp.linalg.eigh(M)
        
        best_vals = vals[:npart]
        best_evecs = evecs[:, :npart]
        
        X = jnp.dot(V_sub, best_evecs)
        HX = jnp.dot(H, X)
        R = HX - X * best_vals[None, :]
        
        # preconditioner: (D - energy)^{-1} * R
        DIVISION_BY_ZERO_THRESHOLD = 1e-5
        denom = D[:, None] - best_vals[None, :]
        denom = jnp.where(jnp.abs(denom) < DIVISION_BY_ZERO_THRESHOLD, DIVISION_BY_ZERO_THRESHOLD, denom)
        Y = R / denom
        
        # Collapse and expand the subspace statically
        V_next = jnp.concatenate([X, Y], axis=1)
        V_next = _local_orthonormalize(V_next)
        
        return V_next, best_vals

    # Run fixed-iteration loop to avoid dynamic compilation tracing
    initial_vals = jnp.zeros((npart,), dtype=H.dtype)
    final_V, final_vals = jax.lax.fori_loop(0, max_iter, body_fun, (V, initial_vals))
    
    # Final extraction of the converged vectors
    final_M = jnp.dot(final_V.T, jnp.dot(H, final_V))
    _, final_evecs = jnp.linalg.eigh(final_M)
    vecs_out = jnp.dot(final_V, final_evecs[:, :npart])
    
    return final_vals, vecs_out

def init_density(nstat: int, hole: Tuple[int]):
    dens = jnp.zeros((nstat, nstat))
    hole_indices = jnp.array(hole)
    dens = dens.at[hole_indices, hole_indices].set(1.0)
    return dens

@jax.jit
def contract_2nf_fused(indices, values, dens):
    """Fused 2-Body kernel: Performs exactly 1 AllReduce across GPUs."""
    p, q, r, s = indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]
    n = dens.shape[0]
    res = jnp.zeros((n, n), dtype=dens.dtype)
    res = res.at[p, r].add(+values * dens[q, s])
    res = res.at[q, r].add(-values * dens[p, s])
    res = res.at[p, s].add(-values * dens[q, r])
    res = res.at[q, s].add(+values * dens[p, r])
    return res
    

@jax.jit
def contract_3nf_fused(indices, values, dens):
    """Fused 3-Body kernel: Performs exactly 1 AllReduce across GPUs."""
    a, b, c = indices[:, 0], indices[:, 1], indices[:, 2]
    d, e, f = indices[:, 3], indices[:, 4], indices[:, 5]

    n = dens.shape[0]
    v2 = values * 2.0
    res = jnp.zeros((n, n), dtype=dens.dtype)

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

@partial(jax.jit, static_argnames=("npart",))
def _hf_step(dens, h1, v2_idx, v2_val, w3_idx, w3_val, npart, mix, prev_vecs):
    gamma = contract_2nf_fused(v2_idx, v2_val, dens)
    omega = contract_3nf_fused(w3_idx, w3_val, dens)

    gamma = 0.5 * (gamma + gamma.T)
    omega = 0.5 * (omega + omega.T)

    hf_ham = h1 + gamma + 0.5 * omega
    hf_ham = 0.5 * (hf_ham + hf_ham.T) # Double protection for eigh

    e_h1 = jnp.sum(h1 * dens)
    e_gamma = jnp.sum(gamma * dens)
    e_omega = jnp.sum(omega * dens)
    energy = e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega

    _, vecs = davidson_eigh(hf_ham, npart, prev_vecs)
    occ = vecs[:, :npart]
    new_dens = occ @ occ.T

    diff_dens = jnp.sum(jnp.abs(new_dens - dens))
    updated_dens = (1.0 - mix) * dens + mix * new_dens
    
    return updated_dens, energy, diff_dens, vecs

# TODO: split func and auto-diff
def solve_HF(L, a_lat, op1, op2, op3, dens, mix=0.5, eps=1e-8, max_iter=100, verbose=False, sm: ShardingManager = None):
    if sm is not None:
        assert sm.num_nodes == 1 or sm.num_gpus == 1, "HF expects 1D mesh, ensure sm.num_nodes or sm.num_gpus is 1"
        h1_dense = sm.prepare(op1.to_dense(), rank=0)
        _dens = sm.prepare(dens, rank=0)
        v2_idx = sm.prepare(op2.indices)
        v2_val = sm.prepare(op2.values)
        w3_idx = sm.prepare(op3.indices)
        w3_val = sm.prepare(op3.values)
    else:
        h1_dense = jnp.array(op1.to_dense())
        v2_idx, v2_val = jnp.array(op2.indices), jnp.array(op2.values)
        w3_idx, w3_val = jnp.array(op3.indices), jnp.array(op3.values)
        _dens = jnp.array(dens)

    prev_energy = 0.0
    converged = False
    npart = int(jnp.trace(_dens).round())

    # NOTE(vivek): _dens already diagonal but diag(dens) returns as 1d vector
    top_indices = jnp.argsort(jnp.diag(_dens))[-npart:]
    vecs = dens[:, top_indices]

    for i in range(max_iter): # maybe switch to fori? 
        _dens, energy, diff_dens, vecs = _hf_step(
            _dens, h1_dense, v2_idx, v2_val, w3_idx, w3_val, npart, mix, vecs
        )
        
        dE = jnp.abs(energy - prev_energy)
        if verbose:
            # convert to jax debug logging
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.4e}, dRho={diff_dens:.4e}")

        if (diff_dens < eps or dE < eps) and i > 1:
            converged = True
            break
        
        prev_energy = energy

    return float(energy), vecs, converged
