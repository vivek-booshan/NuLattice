# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Module to construct IMSRG generator using PyTorch."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

from functools import partial

import jax
import jax.numpy as jnp

@jax.jit
def get_hole_spes(occs: jax.Array, f: jax.Array) -> jax.Array:
    """
    Extracts single-particle energies for hole states
    """
    return occs * jnp.diag(f)


@jax.jit
def get_particle_spes(occs: jax.Array, f: jax.Array, delta: float = 0.0) -> jax.Array:
    """
    Extracts single-particle energies for particle states
    """
    return (1 - occs) * (jnp.diag(f) + delta)


def build_1b_energy_difference(occs: jax.Array, f: jax.Array, delta: float = 0.0) -> jax.Array:
    """
    Constructs one-body energy differences (eps_i - eps_a).
    """
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    # (N, 1) - (1, N) -> (N, N)
    # (e_i - e_a) in the ia block
    f_hp = spe_h[:, None] - spe_p[None, :]

    # Antisymmetrize to get (e_a - e_i) in the ai block
    # 1e-20 prevents division by zero during generator construction
    return f_hp - f_hp.T + 1e-20


def build_2b_energy_difference(occs: jax.Array, f: jax.Array, delta: float = 0.0) -> jax.Array:
    """
    Constructs two-body energy differences (eps_i + eps_j - eps_a - eps_b).
    Uses 4D broadcasting to avoid O(N^4) contractions.
    """
    spe_h = get_hole_spes(occs, f)
    spe_p = get_particle_spes(occs, f, delta)

    # (N, N, N, N)
    # Calculates (e_i + e_j) - (e_a + e_b) efficiently
    gamma_hhpp = (
        spe_h[:, None, None, None] + spe_h[None, :, None, None] 
        - spe_p[None, None, :, None] - spe_p[None, None, None, :]
    )

    # Antisymmetrize to fill the pphh block
    return gamma_hhpp - gamma_hhpp.transpose(2, 3, 0, 1) + 1e-20


@partial(jax.jit, static_argnames=("delta", ))
def build_1b_arctan_generator(occs: jax.Array, f: jax.Array, delta: float = 0.0) -> jax.Array:
    """
    Constructs the 1-body arctan generator using boolean indexing.
    """
    e_diff = build_1b_energy_difference(occs, f, delta)

    h = (occs > 0.5)
    p = ~h

    hp_mask = (h[:, None] & p[None, :]) | (p[:, None] & h[None, :])

    eta_fill = 0.5 * jnp.arctan(2 * f / e_diff)
    
    return jnp.where(hp_mask, eta_fill, 0)


@partial(jax.jit, static_argnames=("delta", ))
def build_2b_arctan_generator(occs: jax.Array, f: jax.Array, gamma: jax.Array, delta: float = 0.0) -> jax.Array:
    """
    Constructs the 2-body arctan generator using boolean indexing.
    """
    e_diff = build_2b_energy_difference(occs, f, delta)

    h = (occs > 0.5)
    p = ~h

    hhpp_mask = (
        h[:, None, None, None] & h[None, :, None, None] & 
        p[None, None, :, None] & p[None, None, None, :]
    )

    pphh_mask = hhpp_mask.transpose(2, 3, 0, 1)

    mask = hhpp_mask | pphh_mask

    eta_fill = 0.5 * jnp.arctan(2 * gamma / e_diff)

    return jnp.where(mask, eta_fill, 0)
