High-Level API (Solvers)
========================

The :mod:`NuLattice.solver` module provides a unified object-oriented interface to run nuclear lattice calculations across different many-body methods.

.. automodule:: NuLattice.solver
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------
The solvers abstract away the lattice initialization. Here is an example of calculating the ground state energy of Oxygen-16 using the unified :class:`HFSolver`.

.. code-block:: python

   from NuLattice.solver import HFSolver
   from NuLattice.constants import ReferenceState

   # 1. Define physical parameters
   L = 4
   a_lat = 2.5
   vT1, vS1, cE = -9.0, -9.0, 6.0

   # 2. Select reference state
   ref_state = ReferenceState.O16_GS

   # 3. Initialize and run solver
   solver = HFSolver(L, a_lat, ref_state, vT1, vS1, cE, backend="cpu")
   energy, vecs, converged = solver.solve(
       eps=1e-8, 
       mix=0.7, 
       max_iter=100, 
       verbose=True
   )

   if converged:
       print(f"Final HF Energy: {energy * solver.phys_unit:.6f} MeV")
