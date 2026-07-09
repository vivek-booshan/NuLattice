from dataclasses import dataclass
from NuLattice.utils import ReferenceState
from NuLattice.utils._jax_types import ShardingManager


@dataclass(frozen=True, slots=True)
class Coupling:
    """
    Immutable container for the Low-Energy Constants (LECs) of the lattice Hamiltonian.

    This class encapsulates the strength of contact interactions in a pionless
    Effective Field Theory (EFT) framework. These constants define the
    short-range nuclear force at the leading order (LO) and are typically
    tuned to reproduce experimental observables such as scattering lengths
    or the binding energies of light nuclei (e.g., ^2H, ^3H).

    Parameters
    ----------
    vT1 : float
        S-wave Isospin-Triplet coupling constant. Governs the ^1S_0 channel (Spin S=0, Isospin T=1). This represents
        the interaction between identical nucleons (nn, pp) or the triplet
        state of a neutron-proton pair.
    vS1 : float
        S-wave Spin-Triplet coupling constant.
        Governs the ^3S_1 channel (Spin S=1, Isospin T=0). This is the
        primary interaction responsible for the bound deuteron state.
    cE : float
        Three-nucleon contact coupling constant.
        Governs the leading-order three-body force (S=1/2, T=1/2). Required
        to stabilize the triton binding energy and prevent the Thomas
        collapse in heavier systems on the lattice.

    Units
    -----
    The values stored in this class are provided in Lattice Units (LU).
    In the NuLattice framework, these are dimensionless coefficients
    defined relative to the kinetic energy scale:

    - vT1, vS1: Correspond to [Energy] * [Volume] in the continuum.
      On the lattice, they scale as a_lat^2 (fm^2).
    - cE: Corresponds to [Energy] * [Volume]^2 in the continuum.
      On the lattice, it scales as a_lat^5 (fm^5).
    """

    vT1: float
    vS1: float
    cE: float

    @property
    def is_su4_symmetric(self) -> bool:
        """True if SU(4) symmetry preserved (vT1 == vS1)"""
        return self.vT1 == self.vS1

    @property
    def is_attractive(self) -> bool:
        """True if the two-body channels are attractive (negative)"""
        return self.vT1 < 0 and self.vS1 < 0

    @property
    def spin_ratio(self) -> float:
        """Ratio of spin-triplet to isospin-triplet strength"""
        return self.vS1 / (self.vT1 + 1e-12)


class BaseSolver:
    def __init__(
        self,
        L: int,
        a_lat: float,
        state,
        vT1: float,
        vS1: float,
        cE: float,
        backend: str = "cpu",
    ):
        self.backend = backend
        self.L = L
        self.a_lat = a_lat
        self.state = state

        if backend == "cpu":
            import NuLattice.cpu.lattice as lat
        elif backend == "jax":
            import NuLattice.jax.lattice as lat
        else:
            raise ValueError("Unknown backend: <cpu|jax>")

        self.coupling = Coupling(vT1, vS1, cE)

        self.phys_unit = lat.phys_unit(a_lat)
        self.basis = lat.get_sp_basis(L)
        self.lattice = lat.get_lattice(L)

        self.op1 = lat.Tkin(self.lattice, self.L)
        self.op2 = lat.contacts(
            self.coupling.vT1, self.coupling.vS1, self.lattice, self.L
        )
        self.op3 = (
            lat.NNNcontact(self.coupling.cE, self.lattice, self.L)
            if self.coupling.cE
            else None
        )

    def jax_options(
        self,
        use_x64: bool = False,
        preallocate: bool = True,
        reserve_memory: float = None,
    ):
        assert self.backend == "jax", (
            f"passing jax options, but backend is {self.backend}"
        )

        if use_x64:
            import jax

            jax.config.update("jax_enable_x64", True)

        if not preallocate:
            import os

            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = False

        if reserve_memory:
            assert 0.0 < reserve_memory < 1.0, (
                f"memory fraction must be within (0.0, 1.0], got {reserve_memory}"
            )
            import os

            os.Environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = reserve_memory


class CCMSolver(BaseSolver):
    def normal_order(self, use_ham=True, sparse=True, NO2B=True, str_3NF=0.0):
        if self.backend == "cpu":
            import NuLattice.cpu.ccm.coupled_cluster as cc
            if use_ham:
                return cc.get_norm_ordered_ham(self.L, self.state, self.op1, self.op2, self.op3, sparse, NO2B)
            return cc.get_norm_ord_int(self.L, self.state, self.coupling.vT1, self.coupling.vS1, str_3NF, sparse)
        else:
            import NuLattice.jax.ccm as cc
            if use_ham:
                return cc.get_norm_ordered_ham(self.L, self.state, self.op1, self.op2, self.op3, NO2B)
            return cc.get_norm_ordered_int(self.L, self.state, self.coupling.vT1, self.coupling.vS1, self.coupling.cE)
                
    def solve(
        self,
        Eref,
        focks, # fpp fph fhh
        contacts, # vpppp vppph vpphh vphph vphhh vhhhh
        mixing=0.5,
        eps=1e-8,
        maxSteps=100,
        max_diis=10,
        delta=0.0,
        sparse=True,
        verbose=True,
        NO2B=True,
        sm: ShardingManager = None,
    ):
        if self.backend == "cpu":
            from NuLattice.cpu.ccm.coupled_cluster import ccsd_solver
        else:
            from NuLattice.jax.ccm import ccsd_solver

        if self.backend == "jax":
            Ecorr, t1, t2 = ccsd_solver(
                focks,
                contacts,
                eps=eps,
                maxSteps=maxSteps,
                max_diis=max_diis,
                delta=delta,
                mixing=mixing,
                verbose=verbose,
                ccs=False,
                sm=sm,
            )
        else:
            Ecorr, t1, t2 = ccsd_solver(
                focks,
                contacts,
                eps=eps,
                maxSteps=maxSteps,
                max_diis=max_diis,
                delta=delta,
                mixing=mixing,
                sparse=sparse,
                verbose=verbose,
                ccs=False,
            )
        return Eref, Ecorr, t1, t2


class HFSolver(BaseSolver):
    def solve(
        self,
        eps: float = 1e-8,
        mix: float = 0.7,
        max_iter: float = 100,
        verbose: float = False,
        sm: ShardingManager = None,
    ):
        if self.backend == "cpu":
            import NuLattice.cpu.hf.hartree_fock as hf
        else:
            import NuLattice.jax.hf.hartree_fock as hf

        nstat = len(self.basis)
        hole = ReferenceState.holes(self.state, self.basis)
        dens = hf.init_density(nstat, hole)

        if self.backend == "jax":
            energy, vecs, conv = hf.solve_HF(
                self.L,
                self.a_lat,
                self.op1,
                self.op2,
                self.op3,
                dens,
                mix=mix,
                eps=eps,
                max_iter=max_iter,
                verbose=verbose,
                sm=sm,
            )
        else:
            energy, vecs, conv = hf.solve_HF(
                self.op1,
                self.op2,
                self.op3,
                dens,
                mix=mix,
                eps=eps,
                max_iter=max_iter,
                verbose=verbose,
            )
        return energy, vecs, conv


class IMSRGSolver(BaseSolver):
    def solve(self, s_max=40, eta_crit=1e-3):
        if self.backend == "cpu":
            import NuLattice.cpu.imsrg as imsrg
        else:
            raise NotImplementedError("jax backend not yet implemented")

        occs = imsrg.normal_ordering.create_occupations(self.basis, self.state)
        e0, f, gamma = imsrg.normal_ordering.compute_normal_ordered_hamiltonian_no2b(
            occs, self.op1, self.op2, self.op3
        )

        e_imsrg, integration_data = imsrg.ode_solver.solve_imsrg2(
            occs, e0, f, gamma, s_max=s_max, eta_criterion=eta_crit
        )
        return e_imsrg, integration_data
