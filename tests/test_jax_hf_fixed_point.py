import jax
import jax.numpy as jnp

from NuLattice.jax.hf.hartree_fock import (
    HFConfig,
    hf_energy_from_density,
    make_hf_solver,
    orbitals_from_diagonal_density,
)


def _empty_interactions(dtype):
    return (
        jnp.zeros((0, 4), dtype=jnp.int32),
        jnp.zeros((0,), dtype=dtype),
        None,
        None,
    )


def test_primal_solver_returns_energy_for_the_returned_density():
    dtype = jnp.float32
    h1 = jnp.array([[-1.0, 0.25], [0.25, 1.0]], dtype=dtype)
    v2_idx, v2_val, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 1)
    config = HFConfig(
        npart=1,
        mix=0.7,
        density_tol=1.0e-7,
        energy_tol=1.0e-7,
        scf_max_iter=60,
        eigensolver="dense",
    )

    result = jax.jit(make_hf_solver(config))(
        h1, v2_idx, v2_val, w3_idx, w3_val, dens0, guess0
    )
    recomputed = hf_energy_from_density(
        result.density, h1, v2_idx, v2_val, w3_idx, w3_val
    )
    expected = -jnp.sqrt(jnp.asarray(1.0 + 0.25**2, dtype=dtype))

    assert jnp.allclose(result.energy, recomputed, atol=1.0e-7)
    assert jnp.allclose(result.energy, expected, rtol=2.0e-6, atol=2.0e-6)


def test_density_residual_must_converge_even_when_energy_change_is_small():
    dtype = jnp.float32
    h1 = jnp.array([[-1.0, 0.4], [0.4, 1.0]], dtype=dtype)
    v2_idx, v2_val, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 1)
    config = HFConfig(
        npart=1,
        mix=0.2,
        density_tol=2.0e-6,
        energy_tol=1.0,
        scf_max_iter=200,
        eigensolver="dense",
    )

    result = jax.jit(make_hf_solver(config))(
        h1, v2_idx, v2_val, w3_idx, w3_val, dens0, guess0
    )

    assert bool(result.converged)
    assert int(result.iterations) > 1
    assert result.residual <= config.density_tol


def test_validation_checks_projector_and_eigenproblem_invariants():
    from NuLattice.jax.hf.hartree_fock import validate_hf_result

    dtype = jnp.float32
    h1 = jnp.array(
        [
            [-1.2, 0.08, 0.02, 0.00],
            [0.08, -0.7, 0.03, 0.01],
            [0.02, 0.03, 0.4, 0.04],
            [0.00, 0.01, 0.04, 0.9],
        ],
        dtype=dtype,
    )
    v2_idx = jnp.array(
        [[0, 1, 0, 1], [0, 2, 0, 2], [1, 2, 1, 2]],
        dtype=jnp.int32,
    )
    v2_val = jnp.array([0.12, -0.04, 0.06], dtype=dtype)
    dens0 = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 2)
    config = HFConfig(
        npart=2,
        mix=0.5,
        density_tol=2.0e-6,
        energy_tol=2.0e-6,
        scf_max_iter=150,
        eigensolver="dense",
    )

    result = jax.jit(make_hf_solver(config))(
        h1, v2_idx, v2_val, None, None, dens0, guess0
    )
    checks = validate_hf_result(
        result, h1, v2_idx, v2_val, None, None, config.npart
    )

    assert bool(result.converged)
    assert checks.particle_number_error < 3.0e-6
    assert checks.idempotency_residual < 5.0e-6
    assert checks.commutator_residual < 5.0e-6
    assert checks.orbital_residual < 5.0e-6
    assert checks.energy_error < 1.0e-7
 
