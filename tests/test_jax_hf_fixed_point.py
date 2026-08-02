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
