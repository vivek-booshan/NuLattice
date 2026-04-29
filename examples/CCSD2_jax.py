import argparse

import NuLattice.jax.lattice as lat
import NuLattice.utils.references as ref
import NuLattice.jax.ccm as ccm
from NuLattice.jax.ccm import stamps, stamp_setup as ss, setup
from NuLattice.utils._jax_types import Chef


def main():
    parser = argparse.ArgumentParser(
        description="Run a NuLattice CCM calculation with custom parameters."
    )

    parser.add_argument("--L", type=int, default=4, help="Lattice size L (L*L*L)")
    parser.add_argument(
        "--a_lat", type=float, default=2.0, help="Lattice spacing in fm"
    )

    parser.add_argument(
        "--vT1", type=float, default=-8.0, help="S-wave isospin-triplet contact"
    )
    parser.add_argument(
        "--vS1", type=float, default=-8.0, help="S-wave spin-triplet contact"
    )
    parser.add_argument("--cE", type=float, default=5.5, help="Three-body contact")

    parser.add_argument(
        "--eps", type=float, default=1.0e-8, help="Convergence threshold"
    )
    parser.add_argument(
        "--maxSteps", type=int, default=100, help="Maximum CC iterations"
    )
    parser.add_argument(
        "--max_diis", type=int, default=10, help="Max DIIS subspace size"
    )
    parser.add_argument(
        "--mixing", type=float, default=0.5, help="Mixing parameter for iterations"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Energy shift to avoid division by zero",
    )

    parser.add_argument(
        "--sparse",
        action="store_false",
        default=True,
        help="use sparse matrices (default True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        default=True,
        help="Suppress iteration output",
    )
    parser.add_argument(
        "--shard", action="store_true", default=False, help="sharding flag"
    )

    args = parser.parse_args()

    # chef = Chef(1, 1)
    chef = None
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
        assert devices_sqrt**2 == len(jax.devices()), (
            "total devices must be perfect square"
        )
        chef = Chef(devices_sqrt, devices_sqrt)
        # chef = Chef(total_processes, local_devices)

    phys_unit = lat.phys_unit(args.a_lat)
    my_basis = lat.get_sp_basis(args.L)
    lattice = lat.get_lattice(args.L)

    print(f"Lattice: {args.L}^3 | Spacing: {args.a_lat} fm")
    print(f"Number of single-particle states = {len(my_basis)}")
    print(f"Number of lattice sites = {len(lattice)}")

    spin, isospin = 2, 2
    stamper = stamps.Stamper(args.L, spin, isospin)
    stamp_1b, stamp_2b, stamp_3b = stamper.stamp(args.vT1, args.vS1, args.cE)

    for i, stamp in enumerate([stamp_1b, stamp_2b, stamp_3b]):
        print(f"{i+1} Body Stamp")
        print(f"Deltas: {stamp.deltas.shape}")
        print(f"Weights: {stamp.weights.shape}")

    # reference state
    ref_state = ref.ref_16O_gs

    mask_p, mask_h, energy = ss.normal_order_masks(
        args.L, ref_state, stamp_1b, stamp_2b, stamp_3b, spin, isospin
    )
    refEn, fock_mats, two_body_int = ss.stamp_to_legacy_wrapper(
        args.L, ref_state, stamp_1b, stamp_2b, stamp_3b, True, spin, isospin
    )

    print(f"Stamp energy: {energy * phys_unit} MeV")

    print(f"Energy of reference: {refEn * phys_unit} MeV")

    # need to integrate with Stamp and Stamper
    corrEn, t1, t2 = ccm.coupled_cluster.stamp_solver(
        args.L,
        stamp_1b,
        stamp_2b,
        stamp_3b,
        mask_p,
        mask_h,
        fock_mats,
        two_body_int,
        eps=args.eps,
        maxSteps=args.maxSteps,
        max_diis=args.max_diis,
        delta=args.delta,
        mixing=args.mixing,
        verbose=args.verbose,
        ccs=False,
        chef=chef,
    )

    gsEn = (corrEn + refEn) * phys_unit
    print("-" * 30)
    print(f"Final Ground State Energy: {gsEn:.6f} MeV")


if __name__ == "__main__":
    main()
