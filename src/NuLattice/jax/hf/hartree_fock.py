from dataclasses import dataclass
from typing import Literal, NamedTuple, Tuple, Callable, Optional

import jax
import jax.numpy as jnp

from NuLattice.utils._jax_types import ShardingManager

from .subspace_solver import (
    _occupied_orbitals as _davidson_occupied_orbitals,
    density_from_orbitals,
)

Array = jax.Array
Eigensolver = Literal["dense", "davidson"]


@dataclass(frozen=True)
class HFConfig:
    npart: int
    mix: float = 0.5
    density_tol: float = 1.0e-8
    energy_tol: float = 1.0e-8
    scf_max_iter: int = 100
    eigensolver: Eigensolver = "davidson"
    davidson_max_iter: int = 10
    verbose: bool = False

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
    orbital_energies, orbitals = _occupied_orbitals(fock, guess_vecs, config)
    projector = density_from_orbitals(orbitals)
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
    orbital_energies, orbitals = _occupied_orbitals(fock, warm_orbitals, config)
    projector = density_from_orbitals(orbitals)
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
    orbital_energies, orbitals = _occupied_orbitals(
        fock, warm_orbitals, config
    )
    projector = density_from_orbitals(orbitals)
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
