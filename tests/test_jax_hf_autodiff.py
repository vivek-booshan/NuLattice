import pytest

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


def test_implicit_gradient_matches_two_level_analytic_result():
    from NuLattice.jax.hf.hartree_fock import make_implicit_hf_solver

    dtype = jnp.float32
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
        adjoint_tol=1.0e-7,
        adjoint_max_iter=100,
    )
    solve = make_implicit_hf_solver(config)

    def energy(t):
        h1 = jnp.array([[-1.0, t], [t, 1.0]], dtype=dtype)
        return solve(
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess0,
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


def test_implicit_interaction_gradient_matches_directional_finite_difference():
    from NuLattice.jax.hf.hartree_fock import make_implicit_hf_solver

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
        [
            [0, 1, 0, 1],
            [0, 2, 0, 2],
            [1, 2, 1, 2],
            [1, 3, 1, 3],
        ],
        dtype=jnp.int32,
    )
    v2_val = jnp.array([0.12, -0.04, 0.06, 0.03], dtype=dtype)
    _, _, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 2)
    config = HFConfig(
        npart=2,
        mix=0.4,
        density_tol=2.0e-6,
        energy_tol=2.0e-6,
        scf_max_iter=150,
        eigensolver="dense",
        adjoint_tol=2.0e-6,
        adjoint_max_iter=200,
    )
    solve = make_implicit_hf_solver(config)

    def energy(values):
        return solve(
            h1,
            v2_idx,
            values,
            w3_idx,
            w3_val,
            dens0,
            guess0,
        ).energy

    direction = jnp.array([1.0, -0.7, 0.3, 0.2], dtype=dtype)
    direction /= jnp.linalg.norm(direction)
    gradient = jax.grad(energy)(v2_val)
    autodiff_directional = jnp.vdot(gradient, direction)

    step = jnp.asarray(1.0e-2, dtype=dtype)
    finite_difference = (
        energy(v2_val + step * direction)
        - energy(v2_val - step * direction)
    ) / (2.0 * step)

    assert jnp.allclose(
        autodiff_directional,
        finite_difference,
        rtol=5.0e-4,
        atol=5.0e-4,
    )


def test_unrolled_and_implicit_gradients_agree_after_convergence():
    from NuLattice.jax.hf.hartree_fock import make_implicit_hf_solver

    dtype = jnp.float32
    h1 = jnp.array([[-1.0, 0.25], [0.25, 1.0]], dtype=dtype)
    v2_idx, v2_val, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 1)
    config = HFConfig(
        npart=1,
        mix=0.6,
        density_tol=1.0e-7,
        energy_tol=1.0e-7,
        scf_max_iter=40,
        eigensolver="dense",
        adjoint_tol=1.0e-7,
        adjoint_max_iter=100,
    )
    implicit_solve = make_implicit_hf_solver(config)

    def implicit_energy(t):
        shifted = h1.at[0, 1].set(t).at[1, 0].set(t)
        return implicit_solve(
            shifted,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess0,
        ).energy

    def unrolled_energy(t):
        shifted = h1.at[0, 1].set(t).at[1, 0].set(t)
        return solve_hf_unrolled(
            shifted,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess0,
            config,
        ).energy

    t = jnp.asarray(0.25, dtype=dtype)
    implicit_gradient = jax.grad(implicit_energy)(t)
    unrolled_gradient = jax.grad(unrolled_energy)(t)
    assert jnp.allclose(
        implicit_gradient,
        unrolled_gradient,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

# @pytest.mark.xfail(
#     strict=True,
#     reason=(
#         "native eigenvector VJPs divide by zero inside a degenerate occupied "
#         "block; the next patch differentiates the projector"
#     ),
# )
# def test_degenerate_occupied_subspace_exposes_native_eigenvector_nan():
def test_degenerate_occupied_subspace_has_finite_projector():
    dtype = jnp.float32
    v2_idx, v2_val, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 2)
    config = HFConfig(
        npart=2,
        mix=1.0,
        density_tol=1.0e-7,
        energy_tol=1.0e-7,
        scf_max_iter=5,
        eigensolver="dense",
        adjoint_tol=1.0e-7,
    )

    from NuLattice.jax.hf.hartree_fock import make_implicit_hf_solver

    solve = make_implicit_hf_solver(config)

    def density_element(coupling):
        h1 = jnp.diag(jnp.array([-1.0, -1.0, 1.0, 2.0], dtype=dtype))
        h1 = h1.at[0, 2].set(coupling).at[2, 0].set(coupling)
        result = solve(
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess0,
        )
        return result.density[0, 2]

    derivative = jax.grad(density_element)(jnp.asarray(0.0, dtype=dtype))
    assert jnp.isfinite(derivative)
    assert jnp.allclose(derivative, -0.5, rtol=2.0e-5, atol=2.0e-5)


def test_implicit_density_gradient_is_independent_of_primal_mixing():
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
        [
            [0, 1, 0, 1],
            [0, 2, 0, 2],
            [1, 2, 1, 2],
            [1, 3, 1, 3],
        ],
        dtype=jnp.int32,
    )
    v2_val = jnp.array([0.12, -0.04, 0.06, 0.03], dtype=dtype)
    _, _, w3_idx, w3_val = _empty_interactions(dtype)
    dens0 = jnp.diag(jnp.array([1.0, 1.0, 0.0, 0.0], dtype=dtype))
    guess0 = orbitals_from_diagonal_density(dens0, 2)

    def derivative_for_mix(mix):
        from NuLattice.jax.hf.hartree_fock import make_implicit_hf_solver

        config = HFConfig(
            npart=2,
            mix=mix,
            density_tol=2.0e-6,
            energy_tol=2.0e-6,
            scf_max_iter=200,
            eigensolver="dense",
            adjoint_tol=2.0e-6,
            adjoint_max_iter=300,
        )
        solve = make_implicit_hf_solver(config)

        def density_element(scale):
            result = solve(
                h1,
                v2_idx,
                scale * v2_val,
                w3_idx,
                w3_val,
                dens0,
                guess0,
            )
            return result.density[0, 2]

        return jax.grad(density_element)(jnp.asarray(1.0, dtype=dtype))

    slow_mix = derivative_for_mix(0.2)
    fast_mix = derivative_for_mix(0.8)
    assert jnp.allclose(slow_mix, fast_mix, rtol=5.0e-5, atol=5.0e-7)

