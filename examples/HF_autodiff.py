import jax
import jax.numpy as jnp

import NuLattice.jax.hf.hartree_fock as hf
import NuLattice.jax.lattice as lat
from NuLattice.utils.constants import ReferenceState


def main():
    lattice_size = 4
    a_lat = 2.0
    lattice = lat.get_lattice(lattice_size)
    basis = lat.get_sp_basis(lattice_size)

    op1 = lat.Tkin(lattice, lattice_size)
    op2 = lat.contacts(-8.0, -8.0, lattice, lattice_size)
    op3 = lat.NNNcontact(5.5, lattice, lattice_size)

    holes = ReferenceState.holes(ReferenceState.O16_GS, basis)
    density0 = hf.init_density(len(basis), holes)
    h1, v2_idx, v2_val, w3_idx, w3_val, density0 = hf.prepare_inputs(
        op1,
        op2,
        op3,
        density0,
    )

    config = hf.HFConfig(
        npart=len(holes),
        mix=0.5,
        scf_max_iter=100,
        eigensolver="davidson",
        adjoint_solver="fixed_point",
    )
    guess0 = hf.orbitals_from_diagonal_density(density0, config.npart)
    solve = jax.jit(hf.make_implicit_hf_solver(config))

    def energy(two_body_values, three_body_values):
        result = solve(
            h1,
            v2_idx,
            two_body_values,
            w3_idx,
            three_body_values,
            density0,
            guess0,
        )
        return result.energy

    value, gradients = jax.value_and_grad(energy, argnums=(0, 1))(
        v2_val,
        w3_val,
    )
    grad_v2, grad_w3 = gradients
    print("HF energy (lattice units):", value)
    print("HF energy (MeV):", value * lat.phys_unit(a_lat))
    print("two-body gradient norm:", jnp.linalg.norm(grad_v2))
    print("three-body gradient norm:", jnp.linalg.norm(grad_w3))


if __name__ == "__main__":
    main()
