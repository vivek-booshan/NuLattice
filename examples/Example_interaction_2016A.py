"""Run the 2016A NLEFT interaction with the JAX Hartree-Fock solver."""

import NuLattice.jax.lattice as lat
import NuLattice.jax.operators.one_body_operators as obops
import NuLattice.jax.operators.two_body_operators as tbops
from NuLattice.utils.constants import ReferenceState as Rs
from NuLattice.utils.constants import MASS, G_A, F_PI, M_PI_0, ReferenceState as Rs


BENCHMARK_ENERGY_MEV = -92.91408870324845


if __name__ == "__main__":
    thisL = 4
    a = 1.0 / 100.0
    my_basis = lat.get_sp_basis(thisL)
    lattice = lat.get_lattice(thisL)
    nstat = len(my_basis)

    # Build the legacy NumPy/SciPy interactions before importing JAX.  The
    # interaction routines use multiprocessing, while JAX starts worker
    # threads during import.
    tkin_list = obops.tKin(thisL, 3, a, mass=MASS)
    print("number of matrix elements from kinetic energy", len(tkin_list))

    bpi = 0.7
    verbose = True
    v_OPE = tbops.onePionEx(
        thisL,
        bpi,
        a,
        lattice,
        verbose=verbose,
        g_A=G_A,
        f_pi=F_PI,
        m_pi_0=M_PI_0,
    )

    cNL = -0.2268 / a
    sNL = 0.077
    cINL = 0.02184 / a
    sL = 0
    v_NL = tbops.shortRangeV_2body(
        lattice, thisL, sL, sNL, cNL, verbose=verbose
    )

    iso_ops = [
        obops.pauli_tau_x(lattice, thisL),
        obops.pauli_tau_y(lattice, thisL),
        obops.pauli_tau_z(lattice, thisL),
    ]
    for op in iso_ops:
        v_NL += tbops.shortRangeV_2body(
            lattice,
            thisL,
            sL,
            sNL,
            cINL,
            verbose=verbose,
            op1b=op,
        )

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import NuLattice.jax.hf.hartree_fock as hf
    from NuLattice.jax.operators.jax_adapters import (
        empty_three_body,
        one_body_from_list,
        two_body_from_sparse,
    )

    myTkin = one_body_from_list(tkin_list, nstat)
    my_VNN = two_body_from_sparse(v_NL + v_OPE, nstat)
    no_three_body = empty_three_body(nstat, dtype=my_VNN.values.dtype)
    print("number of two-body matrix elements", len(my_VNN))

    hole = Rs.holes(Rs.O16_GS, my_basis)
    dens = hf.init_density(nstat, hole, dtype=jnp.complex128)

    erg, trafo, conv = hf.solve_HF(
        thisL,
        a,
        myTkin,
        my_VNN,
        no_three_body,
        dens,
        mix=0.7,
        eps=1.0e-8,
        max_iter=100,
        verbose=True,
        diagonalizer="dense",
    )

    if conv:
        energy = float(erg)
        print("HF energy (MeV) = ", energy)
        error = abs(energy - BENCHMARK_ENERGY_MEV)
        print("benchmark error (MeV) = ", error)
        if error > 1.0e-6:
            raise AssertionError("2016A HF benchmark missed the 1e-6 tolerance")
    else:
        print("HF did not converge")
