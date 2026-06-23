import argparse


import NuLattice.references as ref
from NuLattice.utils.constants import ReferenceState
from NuLattice.solver import CCMSolver

def parse():
    parser = argparse.ArgumentParser(
        description="Run a NuLattice CCM calculation with custom parameters."
    )

    parser.add_argument("--reference", type=str, default="O16",  help="Reference state (HE4, ...)")
    parser.add_argument("--L", type=int, default=4, help="Lattice size L (L*L*L)")
    parser.add_argument("--a_lat", type=float, default=2.0, help="Lattice spacing in fm")

    parser.add_argument("--vT1", type=float, default=-8.0, help="S-wave isospin-triplet contact")
    parser.add_argument("--vS1", type=float, default=-8.0, help="S-wave spin-triplet contact")
    parser.add_argument("--cE", type=float, default=5.5, help="Three-body contact")

    parser.add_argument("--eps", type=float, default=1.0e-8, help="Convergence threshold")
    parser.add_argument("--maxSteps", type=int, default=100, help="Maximum CC iterations")
    parser.add_argument("--max_diis", type=int, default=10, help="Max DIIS subspace size")
    parser.add_argument("--mixing", type=float, default=0.5, help="Mixing parameter for iterations")
    parser.add_argument("--delta", type=float, default=0.0, help="Energy shift to avoid division by zero")
    parser.add_argument("--sparse", action="store_false", default=True, help="Disable sparse matrices (defaults to True)")
    parser.add_argument("--quiet", action="store_false", dest="verbose", default=True, help="Suppress iteration output")
    parser.add_argument("--backend", type=str, default="cpu", help="Backend: <cpu|torch|jax>")

    args = parser.parse_args()
    return args


def main():

    args = parse()
    if args.backend == "jax":
        import os
        os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        import jax
        jax.config.update("jax_enable_x64", True)

    ref_state = getattr(ReferenceState, f"{args.reference.upper()}_GS")
    solver = CCMSolver(args.L, args.a_lat, ref_state, args.vT1, args.vS1, args.cE, backend=args.backend)
        
    print(f"Lattice: {solver.L}^3 | Spacing: {solver.a_lat} fm")
    print(f"Number of single-particle states = {len(solver.basis)}")
    print(f"Number of lattice sites = {len(solver.lattice)}")

    refEn, corrEn, t1, t2 = solver.solve(
        mixing=args.mixing,
        eps=args.eps,
        maxSteps=args.maxSteps,
        max_diis=args.max_diis,
        delta=args.delta,
        sparse=args.sparse,
        verbose=args.verbose,
    )

    gsEn = (corrEn + refEn) * solver.phys_unit
    print("-" * 30)
    print(f"Final Ground State Energy: {gsEn:.6f} MeV")

if __name__ == "__main__":
    main()
