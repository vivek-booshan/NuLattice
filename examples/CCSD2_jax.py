import argparse

import NuLattice.jax.lattice as lat
import NuLattice.jax.ccm as ccm
from NuLattice.utils._jax_types import ShardingManager
from NuLattice.utils.constants import ReferenceState

def main():
    parser = argparse.ArgumentParser(description="Run a NuLattice CCM calculation with custom parameters.")

    parser.add_argument("--reference", type=str, default="O16",  help="Reference state (HE4, ...)")
    parser.add_argument("--L", type=int, default=4, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.0, help="Lattice spacing in fm")

    parser.add_argument("--vT1", type=float, default=-8.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-8.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=5.5, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1.e-8, help="Convergence threshold")
    parser.add_argument("--maxSteps", type=int, default=100, help="Maximum CC iterations")
    parser.add_argument("--max_diis", type=int, default=10, help="Max DIIS subspace size")
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for iterations")
    parser.add_argument("--delta", type=float, default=0.0, help="Energy shift to avoid division by zero")
    
    parser.add_argument("--sparse", action="store_false", default=True, help="use sparse matrices (default True)")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")
    parser.add_argument("--shard", action="store_true", default=False, help="sharding flag")

    args = parser.parse_args()


    # sm = ShardingManager(1, 1)
    sm = None
    if args.shard:
        import jax
        jax.distributed.initialize()
        total_devices = jax.device_count()
        local_devices = jax.local_device_count()
        process_id = jax.process_index()
        total_processes = jax.process_count()

        if process_id == 0:
            print("--- Cluster Report ---")
            print(f"Total Processes (Nodes/Tasks): {total_processes}")
            print(f"Total GCDs across cluster: {total_devices}")
            print(f"Local GCDs per Node: {local_devices}")
            print(f"Devices: {jax.devices()}")
            print("----------------------")
        import math
        devices_sqrt = math.sqrt(len(jax.devices()))
        assert devices_sqrt**2 == len(jax.devices()), "total devices must be perfect square"
        sm = ShardingManager(devices_sqrt, devices_sqrt)
        # sm = ShardingManager(total_processes, local_devices)

    phys_unit = lat.phys_unit(args.a_lat)
    lattice = lat.get_lattice(args.L)
    
    print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")

    myTkin = lat.Tkin(lattice, args.L)
    mycontact = lat.contacts(args.vT1, args.vS1, lattice, args.L)
    my3body = lat.NNNcontact(args.cE, lattice, args.L)

    # reference state
    ref_state = getattr(ReferenceState, f"{args.reference.upper()}_GS")

    states = (args.L**3) * 4
    n_occ = len(ref_state)
    n_virt = states - n_occ
    print(f"States: {states} (particles: {n_virt} | holes: {n_occ})")
    print(f"Number of lattice sites = {len(lattice)}")
    print(f"Matrix elements - 1-body: {len(myTkin)}, 2-body: {len(mycontact)}, 3-body: {len(my3body)}")


    refEn, fock_mats, two_body_int = ccm.get_norm_ordered_ham(
        args.L,
        ref_state,
        myTkin,
        mycontact,
        my3body,
        NO2B=True,
        # sm=sm
    )

    print(f"Energy of reference: {refEn*phys_unit} MeV")

    corrEn, t1, t2 = ccm.ccsd_solver(
        fock_mats, 
        two_body_int, 
        eps=args.eps, 
        maxSteps=args.maxSteps, 
        max_diis=args.max_diis, 
        delta=args.delta, 
        mixing=args.mixing,
        verbose=args.verbose, 
        ccs=False,
        sm=sm,
    )

    gsEn = (corrEn + refEn) * phys_unit
    print("-" * 30)
    print(f"Final Ground State Energy: {gsEn:.6f} MeV")

if __name__ == "__main__":
    main()
