from typing import Tuple
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial

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
    
    # Compute the 4 permutation updates
    t1 = values * dens[q, s]
    t2 = -values * dens[p, s]
    t3 = -values * dens[q, r]
    t4 = values * dens[p, r]
    
    # Concatenate updates and target indices to force a single Atomic Scatter
    updates = jnp.concatenate([t1, t2, t3, t4], axis=0)
    targets = jnp.concatenate([
        jnp.stack([p, r], axis=1),
        jnp.stack([q, r], axis=1),
        jnp.stack([p, s], axis=1),
        jnp.stack([q, s], axis=1)
    ], axis=0)
    
    res = jnp.zeros((n, n), dtype=dens.dtype)
    return res.at[targets[:, 0], targets[:, 1]].add(updates)

@jax.jit
def contract_3nf_fused(indices, values, dens):
    """Fused 3-Body kernel: Performs exactly 1 AllReduce across GPUs."""
    a, b, c, d, e, f = [indices[:, i] for i in range(6)]
    n = dens.shape[0]
    v2 = values * 2.0
    
    # Compute all 9 terms sequentially (held in GPU registers)
    t1 = v2 * (dens[b, e] * dens[c, f] - dens[c, e] * dens[b, f])
    t2 = v2 * (dens[c, e] * dens[a, f] - dens[a, e] * dens[c, f])
    t3 = v2 * (dens[a, e] * dens[b, f] - dens[b, e] * dens[a, f])
    
    t4 = v2 * (dens[b, f] * dens[c, d] - dens[c, f] * dens[b, d])
    t5 = v2 * (dens[c, f] * dens[a, d] - dens[a, f] * dens[c, d])
    t6 = v2 * (dens[a, f] * dens[b, d] - dens[b, f] * dens[a, d])
    
    t7 = v2 * (dens[b, d] * dens[c, e] - dens[c, d] * dens[b, e])
    t8 = v2 * (dens[c, d] * dens[a, e] - dens[a, d] * dens[c, e])
    t9 = v2 * (dens[a, d] * dens[b, e] - dens[b, d] * dens[a, e])
    
    updates = jnp.concatenate([t1, t2, t3, t4, t5, t6, t7, t8, t9], axis=0)
    targets = jnp.concatenate([
        jnp.stack([a, d], axis=1), jnp.stack([b, d], axis=1), jnp.stack([c, d], axis=1),
        jnp.stack([a, e], axis=1), jnp.stack([b, e], axis=1), jnp.stack([c, e], axis=1),
        jnp.stack([a, f], axis=1), jnp.stack([b, f], axis=1), jnp.stack([c, f], axis=1)
    ], axis=0)

    res = jnp.zeros((n, n), dtype=dens.dtype)
    return res.at[targets[:, 0], targets[:, 1]].add(updates)

@partial(jax.jit, static_argnames=("npart",))
def _hf_step(dens, h1, v2_idx, v2_val, w3_idx, w3_val, npart, mix):
    gamma = contract_2nf_fused(v2_idx, v2_val, dens)
    omega = contract_3nf_fused(w3_idx, w3_val, dens)

    # CRITICAL MULTI-GPU FIX: Force perfect Hermiticity.
    # Distributed accumulation introduces floating point noise where H[i,j] != H[j,i]
    # jnp.linalg.eigh will fail to converge if the matrix is slightly asymmetric.
    gamma = 0.5 * (gamma + gamma.T)
    omega = 0.5 * (omega + omega.T)

    hf_ham = h1 + gamma + 0.5 * omega
    hf_ham = 0.5 * (hf_ham + hf_ham.T) # Double protection for eigh

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

def solve_HF(op1, op2, op3, dens, mix=0.5, eps=1e-8, max_iter=100, verbose=False, chef=None):
    if chef is not None:
        mesh = chef.mesh
        # 1. REPLICATED DATA (P()): Every GPU needs a full copy of the small matrices
        # Sharding these causes JAX to distribute the eigensolver, which destroys performance.
        rep_sharding = NamedSharding(mesh, P())
        h1_dense = jax.device_put(op1.to_dense(), rep_sharding)
        _dens = jax.device_put(dens, rep_sharding)

        # 2. SHARDED DATA (P("data")): Distribute the massive interaction lists across GPUs
        data_sharding = NamedSharding(mesh, P("data"))
        v2_idx = jax.device_put(op2.indices, data_sharding)
        v2_val = jax.device_put(op2.values, data_sharding)
        w3_idx = jax.device_put(op3.indices, data_sharding)
        w3_val = jax.device_put(op3.values, data_sharding)
    else:
        # Avoid jnp.asarray if they are already jax arrays, just in case
        h1_dense = jnp.array(op1.to_dense())
        v2_idx, v2_val = jnp.array(op2.indices), jnp.array(op2.values)
        w3_idx, w3_val = jnp.array(op3.indices), jnp.array(op3.values)
        _dens = jnp.array(dens)
    
    npart = int(jnp.trace(_dens).round())
    prev_energy = 0.0
    converged = False

    for i in range(max_iter):
        _dens, energy, diff_dens, vecs = _hf_step(
            _dens, h1_dense, v2_idx, v2_val, w3_idx, w3_val, npart, mix
        )
        
        # Block only on the scalar to prevent locking the whole GPU graph
        dE = jnp.abs(energy - prev_energy)
        if verbose:
            energy.block_until_ready()
            print(f"Iter {i}: E={energy:.8f}, dE={dE:.4e}, dRho={diff_dens:.4e}")

        if (diff_dens < eps or dE < 1e-12) and i > 1:
            converged = True
            break
        
        prev_energy = energy

    return float(energy), vecs, converged
