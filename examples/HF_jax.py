import argparse
import NuLattice.jax.hf.hartree_fock as hf
import NuLattice.jax.lattice as lat
from NuLattice.utils.constants import ReferenceState
from NuLattice.utils._jax_types import ShardingManager as SM

def main():
    parser = argparse.ArgumentParser(description="Run a NuLattice Hartree-Fock calculation.")

    parser.add_argument("--element", type=str, default="O16", 
                        help="Reference state key (e.g., O16, C12, HE4)")
    parser.add_argument("--L", type=int, default=10, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.0, help="Lattice spacing in fm")

    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1.0e-8, help="Convergence threshold")
    parser.add_argument("--mixing", type=float, default=0.7, help="Mixing parameter (Damping)")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum HF iterations")
    parser.add_argument("--span_multiplier", type=float, default=1, help="set span k = c * num_particles")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")
    parser.add_argument("--shard", action="store_true", default=False, help="Enable JAX sharding")

    args = parser.parse_args()

    sm = None
    if args.shard:
        import jax
        jax.distributed.initialize()
        
        total_devices = jax.device_count()
        sm = SM(1, total_devices)
        
        if jax.process_index() == 0:
            print(f"--- Distributed HF Initialized: {total_devices} devices ---")

    phys_unit = lat.phys_unit(args.a_lat)
    my_basis = lat.get_sp_basis(args.L)
    lattice = lat.get_lattice(args.L)

    if args.verbose:
        print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")
        print(f"Single-particle states: {len(my_basis)}")
        print(f"Lattice sites: {len(lattice)}")

    op1 = lat.Tkin(lattice, args.L)
    op2 = lat.contacts(args.vT1, args.vS1, lattice, args.L)
    op3 = lat.NNNcontact(args.cE, lattice, args.L)

    my_ref = getattr(ReferenceState, f"{args.element.upper()}_GS")
    hole = ReferenceState.holes(my_ref, my_basis)
    dens = hf.init_density(len(my_basis), hole)

    erg, trafo, conv = hf.solve_HF(
        args.L,
        args.a_lat,
        op1, 
        op2, 
        op3, 
        dens, 
        eps=args.eps,
        mix=args.mixing,
        max_iter=args.max_iter,
        verbose=args.verbose, 
        sm=sm,
    )

    print("-" * 32)
    if conv:
        print("HF Converged!")
        print(f"Final HF Energy: {erg * phys_unit:.6f} MeV")
    else:
        print("HF did not converge.")
    print("-" * 32)

if __name__ == "__main__":
    main()
