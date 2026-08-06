from functools import partial

import jax
import jax.numpy as jnp

Array = jax.Array

DIVISION_BY_ZERO_THRESHOLD = 1e12
SHIFT_REGULARIZATION = 1e-12

def _adjoint(x: Array) -> Array:
    return jnp.swapaxes(jnp.conj(x), -1, -2)

def hermitianize(x: Array) -> Array:
    return 0.5 * (x + _adjoint(x))

def _occupied_orbitals(fock: Array, npart, guess: Array, max_iter: float = 4) -> tuple[Array, Array]:
    orbital_energies, orbitals = davidson_eigh(fock, npart, guess, max_iter)
    return orbital_energies[:npart], orbitals[:, :npart]

def density_from_orbitals(orbitals: Array) -> Array:
    return hermitianize(orbitals @ _adjoint(orbitals))


def _deterministic_block(n: int, k: int, dtype: jnp.dtype) -> Array:
    """deterministic seed directions"""
    real_dtype = jnp.empty((), dtype=dtype).real.dtype
    rows = jnp.arange(n, dtype=real_dtype)[:, None] + 1.0
    cols = jnp.arange(k, dtype=real_dtype)[None, :] + 1.0
    block = jnp.sin(rows * cols * 0.731) + jnp.cos(
        rows * (cols + 1.0) * 0.193
    )
    return block.astype(dtype)


def _thin_qr(x: Array) -> Array:
    q, _ = jnp.linalg.qr(x, mode="reduced")
    return q


def _initial_davidson_basis(guess_vecs: Array) -> Array:
    """fill full-rank 2*k search basis with occupied guess"""
    n, k = guess_vecs.shape
    seed = _deterministic_block(n, k, guess_vecs.dtype)

    # orthonormalized guess
    q0 = _thin_qr(guess_vecs + 1.0e-7 * seed)

    # remove occupied component
    seed_perp = seed - q0 @ (_adjoint(q0) @ seed)

    # orthogonal complement
    q1 = _thin_qr(seed_perp + 1.0e-7 * jnp.roll(seed, 1, axis=0))
    return jnp.concatenate((q0, q1), axis=1)


def _regularize_denominator(denom: Array, shift: float) -> Array:
    """Bound small Davidson denominators without reversing their sign."""
    signed_shift = jnp.where(denom >= 0.0, shift, -shift)
    return jnp.where(jnp.abs(denom) < shift, signed_shift, denom)


@partial(jax.jit, static_argnames=("npart", "max_iter"))
def davidson_eigh(
    hamiltonian: Array,
    npart: int,
    guess_vecs: Array,
    *,
    max_iter: int = 10,
    diag_shift: float = SHIFT_REGULARIZATION,
) -> tuple[Array, Array]:
    """Return the lowest ``npart`` Ritz pairs in a static 2*k subspace."""

    hamiltonian = hermitianize(hamiltonian)
    diagonal = jnp.real(jnp.diag(hamiltonian))
    guess_vecs = guess_vecs.astype(hamiltonian.dtype)
    basis0 = _initial_davidson_basis(guess_vecs)

    @jax.checkpoint
    def body(_: int, basis: Array) -> Array:
        hb = hamiltonian @ basis
        projected = hermitianize(_adjoint(basis) @ hb)
        vals, coeff = jnp.linalg.eigh(projected)
        x = basis @ coeff[:, :npart]
        x = _thin_qr(x)

        hx = hamiltonian @ x
        rayleigh = jnp.real(jnp.sum(jnp.conj(x) * hx, axis=0))
        residual = hx - x * rayleigh[None, :]

        denom = diagonal[:, None] - rayleigh[None, :]
        denom = _regularize_denominator(denom, diag_shift)
        correction = -residual / denom

        # Remove the occupied component.  A tiny deterministic virtual fallback
        # keeps QR well-defined once the true residual has converged to zero.
        correction -= x @ (_adjoint(x) @ correction)
        correction = correction
        qcorr = _thin_qr(correction)
        return jnp.concatenate((x, qcorr), axis=1)

    basis = jax.lax.fori_loop(0, max_iter, body, basis0)
    projected = hermitianize(_adjoint(basis) @ (hamiltonian @ basis))
    _, coeff = jnp.linalg.eigh(projected)
    orbitals = _thin_qr(basis @ coeff[:, :npart])
    orbital_energies = jnp.real(
        jnp.sum(jnp.conj(orbitals) * (hamiltonian @ orbitals), axis=0)
    )
    order = jnp.argsort(orbital_energies)
    return orbital_energies[order], orbitals[:, order]

def _occupied_orbitals(
    fock: Array,
    npart: int,
    guess: Array,
    max_iter: int = 4,
    diag_shift: float = SHIFT_REGULARIZATION,
) -> tuple[Array, Array]:
    return davidson_eigh(
        fock,
        npart,
        guess,
        max_iter=max_iter,
        diag_shift=diag_shift,
    )
