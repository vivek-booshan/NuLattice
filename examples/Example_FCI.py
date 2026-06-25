import argparse
from scipy.sparse.linalg import eigsh as arpack_eigsh

import NuLattice.cpu.lattice as lat
import NuLattice.cpu.fci.few_body_diagonalization as fbd

def parse():
    parser = argparse.ArgumentParser(description="Run a NuLattice FCI calculation.")
    parser.add_argument("--L", type=int, default=3, help="Lattice size L. Default is 3")
    parser.add_argument("--a_lat", type=float, default=2.0, help="Lattice spacing in fm")

    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact")

def main():
    args = parse()

    L = args.L
    a_lat = args.a_lat
    vT1 = args.vT1
    vS1 = args.vS1
    cE = args.cE

    phys_unit = lat.phys_unit(a_lat)

    my_basis = lat.get_sp_basis(L)
    nstat =  len(my_basis)
    print("number of single-particle states =", nstat)
    lattice = lat.get_lattice(L)
    nsite = len(lattice)
    print("number of lattice sites =", nsite)

    # Compute operators for kinetic energy, two-body contacts, and three-body contact
    myTkin=lat.Tkin(lattice, L)
    print("number of matrix elements from kinetic energy", len(myTkin))

    mycontact=lat.contacts(vT1, vS1, lattice, L)
    print("number of matrix elements from two-body contacts", len(mycontact))

    my3body=lat.NNNcontact(cE, lattice, L)
    print("number of matrix elements from three-body contacts", len(my3body))

    # Compute the deuteron
    print("Computing deuteron")
    # additive quantum numbers
    numpart=2 # number of nucleons
    tz = 0    # twice the value of isospin projection
    sz = 2    # twice the value of spin projection

    # get two-body basis as a dictionary for lookup
    H2_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
    print("matrix dimension:", len(H2_lookup))

    # make scipy.sparse.csr_matrix for kinetic energy 
    T2_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, myTkin, nstat)
    print("kinetic energy done")

    # make scipy.sparse.csr_matrix for 2-body interactions 
    V2_csr_mat = fbd.get_csr_matrix_scalar_op(H2_lookup, mycontact, nstat)
    print("2-body interaction done")

    # add all into Hamiltonian
    H2_csr_mat = T2_csr_mat + V2_csr_mat

    # compute lowest eigenvalue(s)
    k_eig=10  # number of eigenvalues
    vals, vecs = arpack_eigsh(H2_csr_mat, k=k_eig, which='SA')
    print("Energies (MeV):", vals*phys_unit)

    # Compute He3
    print("Computing 3He")
    numpart=3
    tz = -1 # twice the value
    sz = -1 # twice the value

    # get three-body basis as a dictionary for lookup
    He3_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
    print("matrix dimension:", len(He3_lookup))

    # make scipy.sparse.csr_matrix for kinetic energy 
    T3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, myTkin, nstat)
    print("kinetic energy done")

    # make scipy.sparse.csr_matrix for 2-body interaction 
    V3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, mycontact, nstat)
    print("2-body interaction done")

    # make scipy.sparse.csr_matrix for 3-body interaction 
    W3_csr_mat = fbd.get_csr_matrix_scalar_op(He3_lookup, my3body, nstat)
    print("3-body interaction done")

    # add all into Hamiltonian
    H3_csr_mat = T3_csr_mat + V3_csr_mat + W3_csr_mat

    # compute lowest eigenvalue(s)
    k_eig=10  # number of eigenvalues
    vals, vecs = arpack_eigsh(H3_csr_mat, k=k_eig, which='SA')
    print("Energies (MeV):", vals*phys_unit)

    # Compute He4
    print("Computing 4He")
    numpart=4
    tz = 0 # twice the value
    sz = 0 # twice the value

    # get four-body basis as a dictionary for lookup
    He4_lookup = fbd.get_many_body_states(my_basis, numpart, total_tz=tz, total_sz=sz)
    print("matrix dimension:", len(He4_lookup))

    # make scipy.sparse.csr_matrix for kinetic energy 
    T4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, myTkin, nstat)
    print("kinetic energy done")

    # make scipy.sparse.csr_matrix for 2-body interaction 
    V4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, mycontact, nstat)
    print("2-body interaction done")

    # make scipy.sparse.csr_matrix for 3-body interaction 
    W4_csr_mat = fbd.get_csr_matrix_scalar_op(He4_lookup, my3body, nstat)
    print("3-body interaction done")

    # add all into Hamiltonian
    H4_csr_mat = T4_csr_mat + V4_csr_mat + W4_csr_mat

    # compute lowest eigenvalue(s)
    k_eig=2  # number of eigenvalues
    vals, vecs = arpack_eigsh(H4_csr_mat, k=k_eig, which='SA')
    print("Energies (MeV):", vals*phys_unit)
