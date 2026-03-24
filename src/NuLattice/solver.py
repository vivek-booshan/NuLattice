from dataclasses import dataclass
import NuLattice.lattice as lat
from NuLattice.constants import ReferenceState

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
    def __init__(self, L, a_lat, state, vT1, vS1, cE):
        self.L = L
        self.a_lat = a_lat
        self.state = state
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


class CCMSolver(BaseSolver):
    def solve(
        self,
        mixing=0.5,
        eps=1e-8,
        maxSteps=100,
        max_diis=10,
        delta=0.0,
        sparse=True,
        verbose=True,
        NO2B=True,
    ):

        from NuLattice.CCM.coupled_cluster import get_norm_ordered_ham, ccsd_solver
        refEn, fock, v2_no = get_norm_ordered_ham(
            self.L,
            self.state,
            self.op1,
            self.op2,
            self.op3,
            NO2B=NO2B,
            sparse=sparse,
        )

        corrEn, t1, t2 = ccsd_solver(
            fock,
            v2_no,
            eps=eps,
            maxSteps=maxSteps,
            max_diis=max_diis,
            delta=delta,
            mixing=mixing,
            sparse=sparse,
            verbose=verbose,
            ccs=False,
        )
        return refEn, corrEn, t1, t2


class HFSolver(BaseSolver):
    def solve(self, eps=1e-8, mix=0.7, max_iter=100, verbose=False):
        import NuLattice.HF.hartree_fock as hf

        nstat = len(self.basis)
        hole = ReferenceState.holes(self.state, self.basis)
        dens = hf.init_density(nstat, hole)

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
        import NuLattice.IMSRG as imsrg

        occs = imsrg.normal_ordering.create_occupations(self.basis, self.state)
        e0, f, gamma = imsrg.normal_ordering.compute_normal_ordered_hamiltonian_no2b(
            occs, self.op1, self.op2, self.op3
        )

        e_imsrg, integration_data = imsrg.ode_solver.solve_imsrg2(
            occs, e0, f, gamma, s_max=s_max, eta_criterion=eta_crit
        )
        return e_imsrg, integration_data
