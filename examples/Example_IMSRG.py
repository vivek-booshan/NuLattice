"""
Script to solve the IMSRG(2) equations with dynamic arguments.
"""

import argparse
import sys
import matplotlib.pyplot as plt

from NuLattice.constants import ReferenceState
from NuLattice.solver import IMSRGSolver

def parse():
    parser = argparse.ArgumentParser(description="Run a NuLattice IMSRG(2) calculation.")

    parser.add_argument("--L", type=int, default=2, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.5, help="Lattice spacing in fm")
    parser.add_argument("--vT1", type=float, default=-9.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-9.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=6.0, help="Three-body contact (D)")

    parser.add_argument("--s_max", type=float, default=40.0, help="Maximum flow parameter s")
    parser.add_argument("--eta_crit", type=float, default=1e-3, help="Convergence criterion for eta")
    parser.add_argument("--plot", action="store_true", help="Display the energy flow plot")
    
    parser.add_argument("--element", type=str, default="HE3", 
                        help="Reference state key (e.g., HE3, HE4, C12)")
    parser.add_argument("--backend", type=str, default="cpu")
    args = parser.parse_args()
    return args

def main():
    args = parse()

    try:
        ref_state = getattr(ReferenceState, f"{args.element.upper()}_GS")
    except AttributeError:
        print(f"Error: Reference state for '{args.element}' not found in constants.")
        sys.exit(1)

    solver = IMSRGSolver(args.L, args.a_lat, ref_state, args.vT1, args.vS1, args.cE, args.backend)
    e_imsrg, integration_data = solver.solve(args.s_max, args.eta_crit)

    print("-" * 30)
    print(f"Final IMSRG Energy: {e_imsrg * solver.phys_unit:.6f} MeV")
    print(f"Energy (Lattice Units): {e_imsrg:.6f}")

    if args.plot:
        s_vals = [x[0] for x in integration_data]
        e_vals = [x[1] for x in integration_data]
        plt.figure(figsize=(8, 5))
        plt.plot(s_vals, e_vals, label=f"{args.element} Flow")
        plt.xlabel("Flow Parameter (s)")
        plt.ylabel("Energy (Lattice Units)")
        plt.title(f"IMSRG(2) Energy Flow: {args.element}")
        plt.xlim(0, min(10.0, args.s_max))
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
