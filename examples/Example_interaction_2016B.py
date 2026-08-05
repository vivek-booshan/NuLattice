import NuLattice.jax.operators.one_body_operators as obo
import NuLattice.jax.operators.two_body_operators as twbo
import NuLattice.jax.lattice as lat

from NuLattice.utils.constants import MASS

import NuLattice.HF.hartree_fock as hf
from NuLattice.utils import ReferenceState as RS
if __name__ == '__main__':
    myL = 4
    a = 1.0 / 100.0
    my_basis = lat.get_sp_basis(myL)
    lattice = lat.get_lattice(myL)

    myTkin=obo.tKin(myL, 3, a, mass = MASS)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    verbose = True
    cNL = -0.1171 / a
    sNL = 0.077
    cINL = 0.02607 / a
    v_OPE = twbo.onePionEx(myL, bpi, a, lattice, verbose=verbose)
    v_NL=twbo.shortRangeV_2body(lattice, myL, 0, sNL, cNL , verbose=verbose)
    iso_ops = [obo.pauli_tau_x(lattice, myL), obo.pauli_tau_y(lattice, myL), obo.pauli_tau_z(lattice, myL)]
    for op in iso_ops:
        v_NL += twbo.shortRangeV_2body(lattice, myL, 0, sNL, cINL, verbose = verbose, op1b = obo.list_to_sparse1b(op))

    cL = -0.01013 / a
    sL = 0.81
    cSL = - cL / 3.0
    cIL = cSL
    cSIL = cSL
    sp_ops = [obo.pauli_spin_x(lattice, myL), obo.pauli_spin_y(lattice, myL), obo.pauli_spin_z(lattice, myL)]
    v_L = twbo.shortRangeV_2body(lattice, myL, sL, 0, cL, verbose=verbose)
    for op in sp_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cSL, verbose = verbose, op1b = obo.list_to_sparse1b(op))
    for op in iso_ops:
        v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cIL, verbose = verbose, op1b = obo.list_to_sparse1b(op))
        for op2 in sp_ops:
            op1b = obo.list_to_sparse1b(op) @ obo.list_to_sparse1b(op2) 
            v_L += twbo.shortRangeV_2body(lattice, myL, sL, 0, cSL, verbose = verbose, op1b = op1b)
    
    my_VNN = twbo.sparse_to_list_2body(v_NL+v_L+v_OPE, myL)
    print("number of two-body matrix elements", len(my_VNN))

    # We compute oxygen-16
    my_ref = RS.O16_GS
    hole = RS.holes(my_ref,my_basis)
    hnum = len(hole)

    # Density must be defined as complex because Hamiltonian is complex Hermitian
    dens = hf.init_density(len(my_basis),hole,dtype=complex)

    eps=1.e-8
    mix = 0.7
    max_iter=100
    verbose = True
    erg, trafo, conv = hf.solve_HF(myTkin, my_VNN, [], dens,
                                mix=mix, eps=eps, max_iter=max_iter, verbose=verbose)

    if conv:
        print("HF energy (MeV) = ", erg)
    else:
        print("HF did not converge")
