import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NamedTuple, Tuple, Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from NuLattice.utils._jax_types import ShardingManager

from .subspace_solver import (
    _occupied_orbitals as _davidson_occupied_orbitals,
    density_from_orbitals,
)

Array = jax.Array
EigenSolver = Literal["dense", "davidson"]
AdjointSolver = Literal["fixed_point", "gmres"]

@dataclass(frozen=True)
class HFConfig:
    npart: int
    mix: float = 0.5
    density_tol: float = 1.0e-8
    energy_tol: float = 1.0e-8
    verbose: bool = False

    scf_max_iter: int = 100

    eigensolver: EigenSolver = "davidson"
    davidson_max_iter: int = 10
    davidson_subspace_factor: int = 2
    davidson_shift_regularization: float = 1.0e-12


    adjoint_solver: AdjointSolver = "fixed_point"
    adjoint_mix: float = 1.0
    adjoint_tol: float = 1.0e-7
    adjoint_max_iter: int = 100

    gmres_restart: int = 8
    gmres_max_iter: int = 20

    projector_response_tol: float = 1.0e-7
    projector_response_restart: int = 8
    projector_response_max_iter: int = 20

    def __post_init__(self) -> None:
        if self.npart <= 0:
            raise ValueError("npart must be positive")
        if not (0.0 < self.mix <= 1.0):
            raise ValueError("mix must lie in (0, 1]")
        if self.density_tol < 0.0 or self.energy_tol < 0.0:
            raise ValueError("SCF tolerances must be non-negative")
        if self.scf_max_iter <= 0:
            raise ValueError("scf_max_iter must be positive")
        if self.eigensolver not in ("dense", "davidson"):
            raise ValueError(f"unknown eigensolver: {self.eigensolver}")
        if self.davidson_max_iter <= 0:
            raise ValueError("davidson_max_iter must be positive")
        if self.davidson_subspace_factor != 2: # hardcode
            raise ValueError("fixed at 2*npart and should not be changed")
        if self.davidson_shift_regularization <= 0.0: # hardcode
            raise ValueError("davidson_diag_shift must be positive")
        if self.adjoint_solver not in ("fixed_point", "gmres"):
            raise ValueError(f"unknown adjoint solver: {self.adjoint_solver}")
        if self.adjoint_tol < 0.0:
            raise ValueError("adjoint_tol must be non-negative")
        if self.adjoint_max_iter <= 0:
            raise ValueError("adjoint_max_iter must be positive")
        if not (0.0 < self.adjoint_mix <= 1.0):
            raise ValueError("adjoint_mix must lie in (0, 1]")
        if self.gmres_restart <= 0 or self.gmres_max_iter <= 0:
            raise ValueError("adjoint GMRES iteration limits must be positive")
        if self.projector_response_tol < 0.0:
            raise ValueError("projector_response_tol must be non-negative")
        if (
            self.projector_response_restart <= 0
            or self.projector_response_max_iter <= 0
        ):
            raise ValueError("projector response iteration limits must be positive")

        machine_epsilon = _get_machine_epsilon()
        if self.energy_tol < machine_epsilon:
            raise Warning("energy tolerance below machine epsilon.")
        if self.density_tol < machine_epsilon:
            raise Warning("density tolerance below machine epsilon.")
        if self.adjoint_tol < machine_epsilon:
            raise Warning("adjoint tolerance below machine epsilon.")
        if self.projector_response_tol < machine_epsilon:
            raise Warning("projector_response tolerance below machine epsilon.")
        if self.davidson_shift_regularization < machine_epsilon:
            raise Warning("davidson shift regularization below machine epsilon.")

class HFResult(NamedTuple):
    energy: Array
    density: Array
    orbital_energies: Array
    orbitals: Array
    residual: Array
    energy_change: Array
    iterations: Array
    converged: Array

class HFValidation(NamedTuple):
    particle_number: Array
    particle_number_error: Array
    idempotency_residual: Array
    commutator_residual: Array
    orbital_residual: Array
    energy_recomputed: Array
    energy_error: Array

def _get_machine_epsilon():
    dtype: jnp.dtype = jnp.float64 if os.getenv("JAX_ENABLE_X64") else jnp.float32
    epsilon: float = float(jnp.finfo(dtype).eps)
    return epsilon

def _adjoint(x: Array) -> Array:
    return jnp.swapaxes(jnp.conj(x), -1, -2)


def hermitianize(x: Array) -> Array:
    return 0.5 * (x + _adjoint(x))


@jax.jit
def contract_2nf_fused(indices: Array, values: Array, dens: Array) -> Array:
    """Contract the sparse two-body interaction with a one-body density."""
    p, q, r, s = (indices[:, i] for i in range(4))
    n = dens.shape[0]
    dtype = jnp.result_type(values.dtype, dens.dtype)
    res = jnp.zeros((n, n), dtype=dtype)
    res = res.at[p, r].add(+values * dens[q, s])
    res = res.at[q, r].add(-values * dens[p, s])
    res = res.at[p, s].add(-values * dens[q, r])
    res = res.at[q, s].add(+values * dens[p, r])
    return res


@jax.jit
def contract_3nf_fused(indices: Array, values: Array, dens: Array) -> Array:
    """Contract the sparse three-body interaction with two densities."""
    a, b, c, d, e, f = (indices[:, i] for i in range(6))
    n = dens.shape[0]
    dtype = jnp.result_type(values.dtype, dens.dtype)
    v2 = values * 2.0
    res = jnp.zeros((n, n), dtype=dtype)

    res = res.at[a, d].add(v2 * (dens[b, e] * dens[c, f] - dens[c, e] * dens[b, f]))
    res = res.at[b, d].add(v2 * (dens[c, e] * dens[a, f] - dens[a, e] * dens[c, f]))
    res = res.at[c, d].add(v2 * (dens[a, e] * dens[b, f] - dens[b, e] * dens[a, f]))

    res = res.at[a, e].add(v2 * (dens[b, f] * dens[c, d] - dens[c, f] * dens[b, d]))
    res = res.at[b, e].add(v2 * (dens[c, f] * dens[a, d] - dens[a, f] * dens[c, d]))
    res = res.at[c, e].add(v2 * (dens[a, f] * dens[b, d] - dens[b, f] * dens[a, d]))

    res = res.at[a, f].add(v2 * (dens[b, d] * dens[c, e] - dens[c, d] * dens[b, e]))
    res = res.at[b, f].add(v2 * (dens[c, d] * dens[a, e] - dens[a, d] * dens[c, e]))
    res = res.at[c, f].add(v2 * (dens[a, d] * dens[b, e] - dens[b, d] * dens[a, e]))
    return res


def build_mean_fields(
    dens: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array] = None,
    w3_val: Optional[Array] = None,
) -> tuple[Array, Optional[Array]]:
    """Build Hermitian two- and optional three-body mean fields."""
    gamma = hermitianize(contract_2nf_fused(v2_idx, v2_val, dens))
    omega = None
    if w3_idx is not None and w3_val is not None:
        omega = hermitianize(contract_3nf_fused(w3_idx, w3_val, dens))
    return gamma, omega


def build_fock(h1: Array, gamma: Array, omega: Array | None = None) -> Array:
    """Assemble a Fock matrix from precomputed mean fields."""
    if omega is None:
        return hermitianize(h1 + gamma)
    return hermitianize(h1 + gamma + 0.5 * omega)


def build_fock_from_density(
    dens: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array] = None,
    w3_val: Optional[Array] = None,
) -> Array:
    """Evaluate the density-to-Fock map used by every SCF step."""
    gamma, omega = build_mean_fields(dens, v2_idx, v2_val, w3_idx, w3_val)
    return build_fock(h1, gamma, omega)


def hf_energy(
    dens: Array,
    h1: Array,
    gamma: Array,
    omega: Optional[Array] = None,
) -> Array:
    """Evaluate the HF functional from precomputed mean fields."""
    e_h1 = jnp.einsum("ij,ji->", h1, dens)
    e_gamma = jnp.einsum("ij,ji->", gamma, dens)
    e_omega = jnp.asarray(0, dtype=jnp.real(dens[0, 0]).dtype)
    if omega is not None:
        e_omega = jnp.einsum("ij,ji->", omega, dens)
    return jnp.real(e_h1 + 0.5 * e_gamma + (1.0 / 6.0) * e_omega)


def hf_energy_from_density(
    dens: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array] = None,
    w3_val: Optional[Array] = None,
) -> Array:
    """Evaluate the HF functional at exactly ``dens``."""
    gamma, omega = build_mean_fields(dens, v2_idx, v2_val, w3_idx, w3_val)
    return hf_energy(dens, h1, gamma, omega)


def init_density(nstat: int, hole: Tuple[int, ...], dtype=None) -> Array:
    dens = jnp.zeros((nstat, nstat), dtype=dtype)
    hole_indices = jnp.asarray(hole, dtype=jnp.int32)
    return dens.at[hole_indices, hole_indices].set(1.0)


def orbitals_from_diagonal_density(dens: Array, npart: int) -> Array:
    """Choose occupied columns from a diagonal/idempotent initial density."""
    indices = jnp.argsort(jnp.real(jnp.diag(dens)))[-npart:]
    return dens[:, indices]

def _occupied_orbitals(
    fock: Array,
    guess_vecs: Array,
    config: HFConfig,
) -> tuple[Array, Array]:
    if config.eigensolver == "dense" or 2 * config.npart > fock.shape[0]:
        vals, vecs = jnp.linalg.eigh(hermitianize(fock))
        return jnp.real(vals[: config.npart]), vecs[:, : config.npart]
    return _davidson_occupied_orbitals(
        fock,
        config.npart,
        guess_vecs,
        config.davidson_max_iter,
    )


 
@lru_cache(maxsize=None)
def _make_occupied_subspace_solver(config: HFConfig):


   @jax.custom_vjp
   def occupied_subspace(
       fock: Array,
       guess_vecs: Array,
   ) -> tuple[Array, Array, Array]:
       orbital_energies, orbitals = _occupied_orbitals(
           fock, guess_vecs, config
       )
       projector = density_from_orbitals(orbitals)
       return (
           projector,
           jax.lax.stop_gradient(orbital_energies),
           jax.lax.stop_gradient(orbitals),
       )


   def occupied_subspace_fwd(fock: Array, guess_vecs: Array):
       fock = hermitianize(fock)
       orbital_energies, orbitals = _occupied_orbitals(
           fock, guess_vecs, config
       )
       projector = density_from_orbitals(orbitals)
       output = (
           projector,
           jax.lax.stop_gradient(orbital_energies),
           jax.lax.stop_gradient(orbitals),
       )
       return output, (fock, orbital_energies, orbitals, guess_vecs)


   def occupied_subspace_bwd(saved, output_cotangents):
       fock, orbital_energies, orbitals, guess_vecs = saved
       projector_bar = hermitianize(output_cotangents[0])
       def occupied_action(x: Array) -> Array:
           return orbitals @ (_adjoint(orbitals) @ x)
       def virtual_action(x: Array) -> Array:
           return x - occupied_action(x)
       # For each occupied orbital i solve
       #   Q (F - eps_i) Q z_i = Q G u_i.
       # The projector pullback is F_bar = -(Z U^H + U Z^H).
       rhs = virtual_action(projector_bar @ orbitals)
       def response_operator(x: Array) -> Array:
           virtual_x = virtual_action(x)
           shifted = fock @ virtual_x - virtual_x * orbital_energies[None, :]
           # P x acts as an identity on the null space without changing the
           # desired virtual solution because rhs is purely virtual.
           return virtual_action(shifted) + occupied_action(x)


       response, _ = gmres(
           response_operator,
           rhs,
           tol=config.projector_response_tol,
           atol=0.0,
           restart=config.projector_response_restart,
           maxiter=config.projector_response_max_iter,
           solve_method="batched",
       )
       fock_bar = -hermitianize(
           response @ _adjoint(orbitals)
           + orbitals @ _adjoint(response)
       )

       return fock_bar, jnp.zeros_like(guess_vecs)

   occupied_subspace.defvjp(occupied_subspace_fwd, occupied_subspace_bwd)

   return occupied_subspace


def _scf_step(
    dens: Array,
    guess_vecs: Array,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array],
    w3_val: Optional[Array],
    config: HFConfig,
) -> tuple[Array, Array, Array, Array]:
    """Apply one mixed occupied-projector update."""
    fock = build_fock_from_density(dens, h1, v2_idx, v2_val, w3_idx, w3_val)
    projector, orbital_energies, orbitals = _make_occupied_subspace_solver(config)(fock, guess_vecs)
    mixed = hermitianize((1.0 - config.mix) * dens + config.mix * projector)
    return mixed, orbitals, orbital_energies, projector


def _primal_scf_solve(
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array],
    w3_val: Optional[Array],
    dens0: Array,
    guess_vecs0: Array,
    config: HFConfig,
) -> HFResult:

    energy0 = hf_energy_from_density(dens0, h1, v2_idx, v2_val, w3_idx, w3_val)
    state0 = (
        jnp.asarray(0, dtype=jnp.int32),
        hermitianize(dens0),
        guess_vecs0,
        jnp.asarray(jnp.inf, dtype=dens0.real.dtype),
        jnp.asarray(jnp.inf, dtype=energy0.dtype),
        energy0,
    )

    def cond(state: tuple[Array, ...]) -> Array:
        iteration, _, _, density_residual, energy_change, _ = state

        not_converged = jnp.logical_or(
            density_residual > config.density_tol,
            energy_change > config.energy_tol,
        )
        return jnp.logical_and(iteration < config.scf_max_iter, not_converged)

    def body(state: tuple[Array, ...]) -> tuple[Array, ...]:
        iteration, dens, guess, _, _, energy = state
        mixed, orbitals, _, projector = _scf_step(
            dens, guess, h1, v2_idx, v2_val, w3_idx, w3_val, config
        )
        next_energy = hf_energy_from_density(
            mixed, h1, v2_idx, v2_val, w3_idx, w3_val
        )
        density_residual = jnp.max(jnp.abs(projector - dens))
        energy_change = jnp.abs(next_energy - energy)
        if config.verbose:
            jax.debug.print(
                "Iter {iteration}: E={energy:.8f}, dE={de:.4e}, dRho={drho:.4e}",
                iteration=iteration,
                energy=next_energy,
                de=energy_change,
                drho=density_residual,
            )
        return (
            iteration + 1,
            mixed,
            orbitals,
            density_residual,
            energy_change,
            next_energy,
        )

    iteration, dens, warm_orbitals, _, energy_change, _ = jax.lax.while_loop(
        cond, body, state0
    )

    fock = build_fock_from_density(dens, h1, v2_idx, v2_val, w3_idx, w3_val)
    projector, orbital_energies, orbitals = _make_occupied_subspace_solver(config)(fock, warm_orbitals)
    residual = jnp.max(jnp.abs(projector - dens))
    energy = hf_energy_from_density(dens, h1, v2_idx, v2_val, w3_idx, w3_val)
    converged = jnp.logical_and(
        residual <= config.density_tol,
        energy_change <= config.energy_tol,
    )
    return HFResult(
        energy,
        dens,
        orbital_energies,
        orbitals,
        residual,
        energy_change,
        iteration,
        converged,
    )


def make_hf_solver(config: HFConfig) -> Callable:

    def solve(
        h1: Array,
        v2_idx: Array,
        v2_val: Array,
        w3_idx: Optional[Array],
        w3_val: Optional[Array],
        dens0: Array,
        guess_vecs0: Array,
    ) -> HFResult:
        return _primal_scf_solve(
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens0,
            guess_vecs0,
            config,
        )

    return solve

 
def solve_hf_unrolled(
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array | None,
    w3_val: Array | None,
    dens0: Array,
    guess_vecs0: Array,
    config: HFConfig,
) -> HFResult:

    @jax.checkpoint
    def step(carry, _):
        dens, guess = carry
        mixed, orbitals, _, _ = _scf_step(
            dens, guess, h1, v2_idx, v2_val, w3_idx, w3_val, config
        )
        return (mixed, orbitals), None

    (dens, warm_orbitals), _ = jax.lax.scan(
        step,
        (hermitianize(dens0), guess_vecs0),
        xs=None,
        length=config.scf_max_iter,
    )
    fock = build_fock_from_density(dens, h1, v2_idx, v2_val, w3_idx, w3_val)
    projector, orbital_energies, orbitals = _make_occupied_subspace_solver(config)(fock, warm_orbitals)
    residual = jnp.max(jnp.abs(projector - dens))
    energy = hf_energy_from_density(dens, h1, v2_idx, v2_val, w3_idx, w3_val)
    return HFResult(
        energy,
        dens,
        orbital_energies,
        orbitals,
        residual,
        jnp.asarray(jnp.nan, dtype=energy.dtype),
        jnp.asarray(config.scf_max_iter, dtype=jnp.int32),
        residual <= config.density_tol,
    )


 
def validate_hf_result(
    result: HFResult,
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Optional[Array],
    w3_val: Optional[Array],
    npart: int,
) -> HFValidation:
    dens = hermitianize(result.density)
    fock = build_fock_from_density(
        dens, h1, v2_idx, v2_val, w3_idx, w3_val
    )
    particle_number = jnp.real(jnp.trace(dens))
    particle_number_error = jnp.abs(
        particle_number - jnp.asarray(npart, dtype=particle_number.dtype)
    )
    idempotency_residual = jnp.max(jnp.abs(dens @ dens - dens))
    commutator_residual = jnp.max(jnp.abs(fock @ dens - dens @ fock))
    orbital_residual_matrix = (
        fock @ result.orbitals
        - result.orbitals * result.orbital_energies[None, :]
    )
    orbital_residual = jnp.max(
        jnp.linalg.norm(orbital_residual_matrix, axis=0)
    )
    energy_recomputed = hf_energy_from_density(
        dens, h1, v2_idx, v2_val, w3_idx, w3_val
    )
    energy_error = jnp.abs(energy_recomputed - result.energy)
    return HFValidation(
        particle_number,
        particle_number_error,
        idempotency_residual,
        commutator_residual,
        orbital_residual,
        energy_recomputed,
        energy_error,
    )



def prepare_inputs(op1, op2, op3, dens, sm: Optional[ShardingManager] = None):
    has_three_body = op3 is not None and len(op3) > 0

    if sm is not None:
        if sm.num_nodes != 1 and sm.num_gpus != 1:
            raise ValueError(
                "HF expects a 1D mesh; ensure sm.num_nodes or sm.num_gpus is 1"
            )
        h1 = sm.prepare(op1.to_dense(), rank=0)
        dens_array = sm.prepare(dens, rank=0)
        v2_idx = sm.prepare(op2.indices)
        v2_val = sm.prepare(op2.values)
        if has_three_body:
            w3_idx = sm.prepare(op3.indices)
            w3_val = sm.prepare(op3.values)
        else:
            w3_idx = None
            w3_val = None
    else:
        h1 = jnp.asarray(op1.to_dense())
        dens_array = jnp.asarray(dens)
        v2_idx = jnp.asarray(op2.indices, dtype=jnp.int32)
        v2_val = jnp.asarray(op2.values)
        if has_three_body:
            w3_idx = jnp.asarray(op3.indices, dtype=jnp.int32)
            w3_val = jnp.asarray(op3.values)
        else:
            w3_idx = None
            w3_val = None

    return h1, v2_idx, v2_val, w3_idx, w3_val, dens_array


def solve_HF(
    L,
    a_lat,
    op1,
    op2,
    op3,
    dens,
    mix=0.5,
    eps=1.0e-8,
    max_iter=100,
    verbose=False,
    sm: ShardingManager | None = None,
    diagonalizer="davidson",
):
    del L, a_lat
    if diagonalizer not in {"davidson", "dense"}:
        raise ValueError("diagonalizer must be 'davidson' or 'dense'")

    h1, v2_idx, v2_val, w3_idx, w3_val, dens0 = prepare_inputs(
        op1, op2, op3, dens, sm
    )
    npart = int(jax.device_get(jnp.rint(jnp.real(jnp.trace(dens0)))))
    config = HFConfig(
        npart=npart,
        mix=float(mix),
        density_tol=float(eps),
        energy_tol=float(eps),
        scf_max_iter=int(max_iter),
        eigensolver=diagonalizer,
        verbose=bool(verbose),
    )
    guess0 = orbitals_from_diagonal_density(dens0, npart)
    result = jax.jit(make_hf_solver(config))(
        h1, v2_idx, v2_val, w3_idx, w3_val, dens0, guess0
    )
    return (
        float(jax.device_get(result.energy)),
        result.orbitals,
        bool(jax.device_get(result.converged)),
    )

# (1 - J_rho Phi)^H lambda = rho_bar
def make_implicit_hf_solver(config: HFConfig) -> Callable:

    @jax.custom_vjp
    def implicit_density(
        h1: Array,
        v2_idx: Array,
        v2_val: Array,
        w3_idx: Array,
        w3_val: Array,
        init_dens: Array,
        init_vecs: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        result = _primal_scf_solve(
            h1, v2_idx, v2_val, w3_idx, w3_val, init_dens, init_vecs, config
        )
        return (
            result.density,
            jax.lax.stop_gradient(result.orbitals),
            jax.lax.stop_gradient(result.iterations),
            jax.lax.stop_gradient(result.residual),
            jax.lax.stop_gradient(result.energy_change),
        )

    def implicit_density_fwd(
        h1: Array,
        v2_idx: Array,
        v2_val: Array,
        w3_idx: Array,
        w3_val: Array,
        init_dens: Array,
        init_vecs: Array,
    ):
        result = _primal_scf_solve(
            h1, v2_idx, v2_val, w3_idx, w3_val, init_dens, init_vecs, config
        )
        output = (
            result.density,
            jax.lax.stop_gradient(result.orbitals),
            jax.lax.stop_gradient(result.iterations),
            jax.lax.stop_gradient(result.residual),
            jax.lax.stop_gradient(result.energy_change),
        )
        saved = (
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            result.density,
            result.orbitals,
            init_dens,
            init_vecs,
        )
        return output, saved


    def implicit_density_bwd(saved, output_cotangents):
        (
            h1,
            v2_idx,
            v2_val,
            w3_idx,
            w3_val,
            dens_star,
            orbitals_star,
            init_dens,
            init_vecs,
        ) = saved

        def phi(
            h1_arg: Array,
            v2_val_arg: Array,
            w3_val_arg: Array,
            dens_arg: Array,
        ) -> Array:
            mixed, _, _, _ = _scf_step(
                dens_arg,
                jax.lax.stop_gradient(orbitals_star),
                h1_arg,
                v2_idx,
                v2_val_arg,
                w3_idx,
                w3_val_arg,
                config,
            )
            return mixed

        dens_bar = hermitianize(output_cotangents[0])

        # NOTE(vivek): rematerialize the one-step map for each adjoint matvec instead of
        # retaining every NxN intermediate from the primal history
        phi_remat = jax.checkpoint(phi)
        _, pullback = jax.vjp(phi_remat, h1, v2_val, w3_val, dens_star)

        def jacobian_transpose_times(cotangent: Array) -> Array:
            return hermitianize(pullback(hermitianize(cotangent))[3])

        if config.adjoint_solver == "fixed_point":
            state0 = (
                jnp.asarray(0, dtype=jnp.int32),
                jnp.zeros_like(dens_bar),
                jnp.asarray(jnp.inf, dtype=dens_bar.real.dtype),
            )

            def adjoint_cond(state: tuple[Array, Array, Array]) -> Array:
                iteration, _, residual = state
                return jnp.logical_and(
                    iteration < config.adjoint_max_iter,
                    residual > config.adjoint_tol,
                )

            def adjoint_body(state: tuple[Array, Array, Array]):
                iteration, lam, _ = state
                candidate = dens_bar + jacobian_transpose_times(lam)
                next_lam = hermitianize(
                    (1.0 - config.adjoint_mix) * lam
                    + config.adjoint_mix * candidate
                )
                residual = jnp.max(jnp.abs(next_lam - lam))
                return iteration + 1, next_lam, residual

            _, lam, _ = jax.lax.while_loop(
                adjoint_cond, adjoint_body, state0
            )
        else:
            def adjoint_operator(lam: Array) -> Array:
                return hermitianize(lam - jacobian_transpose_times(lam))

            lam, _ = gmres(
                adjoint_operator,
                dens_bar,
                tol=config.adjoint_tol,
                atol=0.0,
                restart=config.gmres_restart,
                maxiter=config.gmres_max_iter,
                solve_method="batched",
            )
            lam = hermitianize(lam)

        h1_bar, v2_val_bar, w3_val_bar, _ = pullback(lam)

        return (
            hermitianize(h1_bar),
            None, # v2_idx
            v2_val_bar,
            None, # w3_idx
            w3_val_bar,
            jnp.zeros_like(init_dens),
            jnp.zeros_like(init_vecs),
        )

    implicit_density.defvjp(implicit_density_fwd, implicit_density_bwd)

    def solve(
        h1: Array,
        v2_idx: Array,
        v2_val: Array,
        w3_idx: Array,
        w3_val: Array,
        init_dens: Array,
        init_vecs: Array,
    ) -> HFResult:
        dens, warm_orbitals, iterations, _, energy_change = implicit_density(
            h1, v2_idx, v2_val, w3_idx, w3_val, init_dens, init_vecs
        )
        fock = build_fock_from_density(
            dens, h1, v2_idx, v2_val, w3_idx, w3_val
        )
        projector, orbital_energies, orbitals = _make_occupied_subspace_solver(config)(fock, jax.lax.stop_gradient(warm_orbitals))
        residual = jnp.max(jnp.abs(projector - dens))
        energy = hf_energy_from_density(
            dens, h1, v2_idx, v2_val, w3_idx, w3_val
        )
        converged = jnp.logical_and(
            residual <= config.density_tol,
            energy_change <= config.energy_tol,
        )
        return HFResult(
            energy,
            dens,
            jax.lax.stop_gradient(orbital_energies),
            jax.lax.stop_gradient(orbitals),
            residual,
            energy_change,
            iterations,
            converged,
        )

    return solve

# NOTE(vivek) when set to none the cache can grow without bound, but HFConfig doesn't change so it should be fine ... probably
# NOTE(vivek) lru cache avoids jax re-tracing and re-compiling on every call, contingent on static hashable dataclass
@lru_cache(maxsize=None)
def _cached_jitted_implicit_solver(config: HFConfig):
    return jax.jit(make_implicit_hf_solver(config))

def solve_hf_implicit(
    h1: Array,
    v2_idx: Array,
    v2_val: Array,
    w3_idx: Array,
    w3_val: Array,
    init_dens: Array,
    init_vecs: Array,
    config: HFConfig,
) -> HFResult: 
    return _cached_jitted_implicit_solver(config)(
        h1, v2_idx, v2_val, w3_idx, w3_val, init_dens, init_vecs
    )
