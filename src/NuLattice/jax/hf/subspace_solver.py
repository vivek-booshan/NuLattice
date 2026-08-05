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

@jax.jit
def _local_orthonormalize(V: Array) -> Array:
    # Compute the small overlap matrix (2k x 2k).
    S = jnp.dot(_adjoint(V), V)  # calls (cheap) AllReduce on mesh
    S += SHIFT_REGULARIZATION * jnp.eye(S.shape[0], dtype=S.dtype)
    L = jnp.linalg.cholesky(S)
    L_inv = jnp.linalg.inv(L)
    return jnp.dot(V, _adjoint(L_inv))

@jax.jit
def _cqr2(V: Array) -> Array:
    return _local_orthonormalize(_local_orthonormalize(V))

@partial(jax.jit, static_argnames=("npart", ))
def davidson_eigh(H: Array, npart: int, guess_vecs: Array, max_iter: int = 10):
    """
    Finds the lowest `npart` eigenvalues/eigenvectors of a sharded dense Hamiltonian H.

    Args:
        H: Sharded Hamiltonian matrix of shape (nstat, nstat)
        npart: Number of occupied states (lowest roots needed)
        guess_vecs: Initial guess vectors of shape (nstat, npart) from previous SCF step
        max_iter: Number of subspace expansion steps (try 3-5 for warm starts)

    Frankensteined from https://joshuagoings.com/2013/08/23/davidsons-method/
    """
    nstat = H.shape[0]
    D = jnp.diag(H) # Extract diagonal for the preconditioner

    # Initialize a static subspace V of size (nstat, 2 * npart)
    V = jnp.zeros((nstat, 2 * npart), dtype=H.dtype)
    V = V.at[:, :npart].set(guess_vecs)
    V = _cqr2(V)

    def body_fun(i, state):
        V_sub, _ = state

        # Project into subspace: M = VT H V -> (2k, 2k)
        HV = jnp.dot(H, V_sub)
        M = jnp.dot(_adjoint(V_sub), HV)

        # local eigen solution
        vals, evecs = jnp.linalg.eigh(M)

        best_vals = vals[:npart]
        best_evecs = evecs[:, :npart]

        X = jnp.dot(V_sub, best_evecs)
        HX = jnp.dot(H, X)
        R = HX - X * best_vals[None, :]

        # preconditioner: (D - energy)^{-1} * R
        denom = D[:, None] - best_vals[None, :]
        denom = jnp.where(jnp.abs(denom) < DIVISION_BY_ZERO_THRESHOLD, DIVISION_BY_ZERO_THRESHOLD, denom)
        Y = R / denom

        # Collapse and expand the subspace statically
        V_next = jnp.concatenate([X, Y], axis=1)
        V_next = _cqr2(V_next)

        return V_next, best_vals

    # Run fixed-iteration loop to avoid dynamic compilation tracing
    initial_vals = jnp.zeros((npart,), dtype=jnp.real(H).dtype)
    final_V, final_vals = jax.lax.fori_loop(0, max_iter, body_fun, (V, initial_vals))

    # Final extraction of the converged vectors
    final_M = jnp.dot(_adjoint(final_V), jnp.dot(H, final_V))
    _, final_evecs = jnp.linalg.eigh(final_M)
    vecs_out = jnp.dot(final_V, final_evecs[:, :npart])

    return final_vals, vecs_out
