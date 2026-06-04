import argparse
import sys
import time

import jax 

from NuLattice.solver import HFSolver
from NuLattice.constants import ReferenceState

def parse():
    parser = argparse.ArgumentParser(description="Run a NuLattice Hartree-Fock calculation.")

    parser.add_argument("--L", type=int, default=4, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.5, help="Lattice spacing in fm")
    
    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1e-8, help="Convergence threshold")
    parser.add_argument("--mix", type=float, default=0.7, help="Mixing parameter for density iterations")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum HF iterations")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")
    parser.add_argument("--span_multiplier", type=float, default=1, help="k = span_multiplier * num_particles")
    parser.add_argument("--use_davidson", action="store_true", default=False, help="use davidson solver")

    parser.add_argument("--element", type=str, default="O16", 
                        help="Reference state key (e.g., O16, C12, HE4)")
    parser.add_argument("--backend", type=str, default="cpu", help="backend")

    args = parser.parse_args()
    return args

def main():
    args = parse()

    try:
        attr_name = f"{args.element.upper()}_GS"
        ref_state = getattr(ReferenceState, attr_name)
    except AttributeError:
        print(f"Error: Reference state for '{args.element}' not found.")
        sys.exit(1)

    print(f"Lattice L = {args.L}")

    solver = HFSolver(args.L, args.a_lat, ref_state, args.vT1, args.vS1, args.cE, backend=args.backend)
    method = "davidson" if args.use_davidson else None
    # start = time.perf_counter()
    # erg, trafo, conv = solver.solve(args.eps, args.mix, args.max_iter, verbose=False, chef=None, span_multiplier=args.span_multiplier, method=method)
    # end = time.perf_counter()
    # print("cold start:", end - start)
    start = time.time()
    erg, trafo, conv = solver.solve(args.eps, args.mix, args.max_iter, args.verbose, chef=None, span_multiplier=args.span_multiplier, method=method)
    end = time.time()
    try:
        stats = jax.local_devices()[0].memory_stats()
        peak_mb = stats.get("peak_bytes_in_use", 0) / 1e6
        print(f"Peak Memory Use: {peak_mb} Mb")
    except Exception:
        print("Peak Memory Use: 0 Mb")
    print("warm start:", end - start)
    final_energy = erg * solver.phys_unit
    print("-" * 30)
    if conv:
        print("HF Convergence: SUCCESS")
        print(f"Final HF Energy: {final_energy} MeV")
    else:
        print("HF Convergence: FAILED")
        print(f"Final HF Energy: {final_energy} MeV")

if __name__ == "__main__":
    main()
