import jax
import jax.numpy as jnp

from .helpers import hermitianize

Array = jax.Array


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
    w3_idx: Array,
    w3_val: Array,
) -> tuple[Array, Array]:
    gamma = hermitianize(contract_2nf_fused(v2_idx, v2_val, dens))
    omega = hermitianize(contract_3nf_fused(w3_idx, w3_val, dens))
    return gamma, omega


def build_fock(
    dens: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array,
    w3_val: Array,
) -> Array:
    gamma, omega = build_mean_fields(dens, v2_idx, v2_val, w3_idx, w3_val)
    return hermitianize(h1 + gamma + 0.5 * omega)


def hf_energy(
    dens: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array,
    w3_val: Array,
) -> Array:
    """Evaluate the HF functional at exactly ``dens``."""
    gamma, omega = build_mean_fields(dens, v2_idx, v2_val, w3_idx, w3_val)
    e_h1 = jnp.einsum("ij,ji->", h1, dens)
    e_gamma = jnp.einsum("ij,ji->", gamma, dens)
    e_omega = jnp.einsum("ij,ji->", omega, dens)
    return jnp.real(e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega)
