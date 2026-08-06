import jax
import jax.numpy as jnp

from NuLattice.jax.hf.hartree_fock import (
    HFConfig,
    init_density,
    orbitals_from_diagonal_density,
    prepare_inputs,
    solve_hf_implicit,
    validate_hf_result,
)
from NuLattice.solver import HFSolver
from NuLattice.utils.constants import ReferenceState


def test_o16_l6_matches_new_operator_energy_and_solution_invariants():
    solver = HFSolver(
        6,
        2.5,
        ReferenceState.O16_GS,
        -9.0,
        -9.0,
        6.0,
        backend="jax",
    )
    holes = ReferenceState.holes(solver.state, solver.basis)
    density0 = init_density(len(solver.basis), holes)
    h1, v2_idx, v2_val, w3_idx, w3_val, density0 = prepare_inputs(
        solver.op1,
        solver.op2,
        solver.op3,
        density0,
    )
    config = HFConfig(
        npart=len(holes),
        mix=0.7,
        density_tol=1.0e-8,
        energy_tol=1.0e-8,
        scf_max_iter=100,
        eigensolver="davidson",
        davidson_max_iter=10,
    )
    guess0 = orbitals_from_diagonal_density(density0, config.npart)
    result = solve_hf_implicit(
        h1,
        v2_idx,
        v2_val,
        w3_idx,
        w3_val,
        density0,
        guess0,
        config,
    )
    checks = validate_hf_result(
        result,
        h1,
        v2_idx,
        v2_val,
        w3_idx,
        w3_val,
        config.npart,
    )

    energy_mev = result.energy * solver.phys_unit
    # The reorganized operator implementation gives -107.149730926... MeV in
    # float64.  Float32 SCF terminates at its precision floor and differs by
    # roughly 0.002 MeV on CPU.
    assert jnp.allclose(energy_mev, -107.1497309260, atol=3.0e-3, rtol=0.0)
    assert bool(result.converged)
    assert jnp.allclose(checks.particle_number, 16.0, atol=2.0e-5)
    assert checks.particle_number_error < 2.0e-5
    assert checks.idempotency_residual < 2.0e-5
    assert checks.commutator_residual < 2.0e-4
    assert checks.orbital_residual < 2.0e-2
    assert checks.energy_error < 1.0e-6


def test_real_lattice_implicit_gradient_matches_finite_difference():
    solver = HFSolver(
        3,
        2.5,
        ReferenceState.O16_GS,
        -9.0,
        -9.0,
        6.0,
        backend="jax",
    )
    holes = ReferenceState.holes(solver.state, solver.basis)
    density0 = init_density(len(solver.basis), holes)
    h1, v2_idx, v2_val, w3_idx, w3_val, density0 = prepare_inputs(
        solver.op1,
        solver.op2,
        solver.op3,
        density0,
    )
    config = HFConfig(
        npart=len(holes),
        mix=0.7,
        density_tol=1.0e-8,
        energy_tol=1.0e-8,
        scf_max_iter=100,
        eigensolver="dense",
        adjoint_solver="fixed_point",
        adjoint_tol=2.0e-7,
        adjoint_max_iter=150,
        projector_response_tol=2.0e-7,
    )
    guess0 = orbitals_from_diagonal_density(density0, config.npart)

    def energy(scale):
        result = solve_hf_implicit(
            h1,
            v2_idx,
            scale * v2_val,
            w3_idx,
            w3_val,
            density0,
            guess0,
            config,
        )
        return result.energy

    scale = jnp.asarray(1.0, dtype=h1.dtype)
    derivative = jax.grad(energy)(scale)
    step = jnp.asarray(5.0e-3, dtype=h1.dtype)
    finite_difference = (energy(scale + step) - energy(scale - step)) / (
        2.0 * step
    )

    assert jnp.isfinite(derivative)
    assert jnp.allclose(
        derivative,
        finite_difference,
        rtol=3.0e-4,
        atol=3.0e-3,
    )
