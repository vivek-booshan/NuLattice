import jax
import jax.numpy as jnp

from NuLattice.jax.hf.subspace_solver import (
    _initial_davidson_basis,
    _regularize_denominator,
    davidson_eigh,
    density_from_orbitals,
)


def test_initial_davidson_basis_contains_no_zero_padded_directions():
    dtype = jnp.float32
    guess = jnp.eye(8, 2, dtype=dtype)
    basis = _initial_davidson_basis(guess)
    overlap = basis.T.conj() @ basis

    assert basis.shape == (8, 4)
    assert jnp.allclose(overlap, jnp.eye(4, dtype=dtype), atol=2.0e-5)
    assert jnp.min(jnp.linalg.norm(basis, axis=0)) > 0.99


def test_small_negative_preconditioner_denominator_keeps_its_sign():
    denom = jnp.array([-1.0e-9, 0.0, 1.0e-9, -2.0e-4], dtype=jnp.float32)
    regularized = _regularize_denominator(denom, 1.0e-5)

    assert regularized[0] == -1.0e-5
    assert regularized[1] == 1.0e-5
    assert regularized[2] == 1.0e-5
    assert regularized[3] == denom[3]


def test_davidson_matches_dense_lowest_projector():
    key = jax.random.key(7)
    raw = jax.random.normal(key, (8, 8), dtype=jnp.float32)
    hamiltonian = 0.5 * (raw + raw.T)
    hamiltonian += jnp.diag(jnp.linspace(-2.0, 2.0, 8, dtype=jnp.float32))
    guess = jnp.eye(8, 2, dtype=jnp.float32)

    vals, orbitals = davidson_eigh(
        hamiltonian,
        2,
        guess,
        max_iter=30,
    )
    dense_vals, dense_vecs = jnp.linalg.eigh(hamiltonian)
    projector = density_from_orbitals(orbitals)
    dense_projector = density_from_orbitals(dense_vecs[:, :2])

    assert jnp.allclose(vals, dense_vals[:2], rtol=5.0e-5, atol=5.0e-5)
    assert jnp.allclose(projector, dense_projector, rtol=1.0e-4, atol=1.0e-4)

