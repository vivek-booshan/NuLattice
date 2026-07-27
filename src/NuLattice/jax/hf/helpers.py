import jax.numpy as jnp

def _adjoint(x):
    return jnp.swapaxes(jnp.conj(x), -1, -2)

def hermitianize(x):
    return 0.5 * (x + _adjoint(x))


