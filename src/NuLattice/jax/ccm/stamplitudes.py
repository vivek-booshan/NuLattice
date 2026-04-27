from functools import partial

import jax
import jax.numpy as jnp

@jax.jit
def apply_spatial_shift(tensor: jax.Array, delta: jax.Array, L: int, dof: int = 4, axis: int = 0) -> jax.Array:
    """
    Unpacks a flat spatial axis into 3D geometry, applies the topological shift 
    with periodic boundary conditions, and re-flattens.
    """
    is_zero_shift = jnp.all(delta == 0)
    
    def do_shift(t):
        # 1. Move the target axis to the front
        t_moved = jnp.moveaxis(t, axis, 0)
        shape_moved = t_moved.shape
        
        # 2. Unflatten the front axis into (L, L, L, dof)
        spatial_shape = (L, L, L, dof) + shape_moved[1:]
        grid_tensor = jnp.reshape(t_moved, spatial_shape)
        
        # 3. Apply the shift to the spatial dimensions (x, y, z)
        shifted_grid = grid_tensor
        for i in range(3):
            shifted_grid = jax.lax.cond(
                delta[i] != 0,
                lambda g: jnp.roll(g, shift=delta[i], axis=i),
                lambda g: g,
                shifted_grid
            )
            
        # 4. Reflatten and restore axis position
        shifted_flat = jnp.reshape(shifted_grid, shape_moved)
        return jnp.moveaxis(shifted_flat, 0, axis)
    
    return jax.lax.cond(is_zero_shift, lambda t: t, do_shift, tensor)


@partial(jax.jit, static_argnames=("contraction_str", "L", "dof", "shift_axes"))
def stamp_einsum(
    contraction_str: str, 
    tensor: jax.Array, 
    stamps: tuple, 
    L: int, 
    dof: int = 4, 
    shift_axes: tuple = (0,)
) -> jax.Array:
    """
    Applies a topological stamp to a tensor using a fused jax.lax.scan loop.
    
    dof: spin * isospin
    shift_axes: A tuple of integers specifying which axes of the input `tensor` 
                should undergo the spatial shift before the internal algebra is applied.
    """
    deltas, weights = stamps
    
    # Evaluate a dummy contraction to trace the exact shape/dtype for the scan accumulator
    dummy_out = jnp.einsum(contraction_str, tensor, weights[0])
    
    def scan_body(accum, stamp_data):
        delta, w_matrix = stamp_data
        
        # 1. Geometry: Apply the spatial shift to all interacting legs
        shifted_tensor = tensor
        for ax in shift_axes:
            shifted_tensor = apply_spatial_shift(shifted_tensor, delta, L, dof, axis=ax)
            
        # 2. Algebra: Contract with the internal spin/isospin transition matrix
        term = jnp.einsum(contraction_str, shifted_tensor, w_matrix)
        
        return accum + term, None

    # jax.lax.scan compiles the loop into a single, fused kernel execution on the GPU
    final_result, _ = jax.lax.scan(scan_body, jnp.zeros_like(dummy_out), (deltas, weights))
    
    return final_result
