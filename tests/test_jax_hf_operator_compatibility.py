import jax.numpy as jnp

from NuLattice.jax.hf.hartree_fock import (
    init_density,
    # prepare_hf_inputs,
    prepare_inputs,
    solve_HF,
)
from NuLattice.utils._jax_types import OneBodyOperator, TwoBodyOperator


def _two_level_operators(dtype=jnp.float32):
    h1 = OneBodyOperator(
        jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.int32),
        jnp.array([-1.0, 0.25, 0.25, 1.0], dtype=dtype),
        2,
    )
    v2 = TwoBodyOperator(
        jnp.empty((0, 4), dtype=jnp.int32),
        jnp.empty((0,), dtype=dtype),
        2,
    )
    return h1, v2


def test_legacy_solve_hf_preserves_dense_diagonalizer_argument():
    op1, op2 = _two_level_operators()
    dens0 = init_density(2, (0,), dtype=jnp.float32)

    energy, orbitals, converged = solve_HF(
        1,
        1.0,
        op1,
        op2,
        None,
        dens0,
        mix=0.7,
        eps=1.0e-7,
        max_iter=60,
        diagonalizer="dense",
    )
    expected = -jnp.sqrt(jnp.asarray(1.0 + 0.25**2, dtype=jnp.float32))

    assert converged
    assert orbitals.shape == (2, 1)
    assert jnp.allclose(energy, expected, rtol=2.0e-6, atol=2.0e-6)
