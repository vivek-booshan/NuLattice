import jax
import jax.numpy as jnp

from NuLattice.jax.hf.hartree_fock import (
    HFConfig,
    orbitals_from_diagonal_density,
    solve_hf_unrolled,
)


def _empty_interactions(dtype):
    return (
        jnp.zeros((0, 4), dtype=jnp.int32), # v2_idx
        jnp.zeros((0,), dtype=dtype),       # v2_val
        jnp.zeros((0, 6), dtype=jnp.int32), # w3_idx
        jnp.zeros((0,), dtype=dtype),       # w3_val
    )


def test_unrolled_gradient_matches_two_level_analytic_result():
    dtype = jnp.float32
    v2_idx, v2_val, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 1)
    config = HFConfig(
        npart=1,
        mix=0.7,
        density_tol=1.0e-7,
        energy_tol=1.0e-7,
        scf_max_iter=40,
        eigensolver="dense",
    )

    def energy(t):
        h1 = jnp.array([[-1.0, t], [t, 1.0]], dtype=dtype)
        return solve_hf_unrolled(
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess0,
            config,
        ).energy

    t = jnp.asarray(0.3, dtype=dtype)
    value, derivative = jax.value_and_grad(energy)(t)
    expected_value = -jnp.sqrt(1.0 + t**2)
    expected_derivative = -t / jnp.sqrt(1.0 + t**2)

    assert jnp.allclose(value, expected_value, rtol=2.0e-6, atol=2.0e-6)
    assert jnp.allclose(
        derivative,
        expected_derivative,
        rtol=2.0e-6,
        atol=2.0e-6,
    )

