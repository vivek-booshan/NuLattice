from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .stamps import get_global_indices_jax

def apply_spatial_shift(tensor: jax.Array, delta: jax.Array, L: int, dof: int = 4, axis: int = 0) -> jax.Array:

    is_zero_shift = jnp.all(delta == 0)
    
    def do_shift(t):
        # All shape/axis logic here uses concrete integers now
        t_moved = jnp.moveaxis(t, axis, 0)
        
        # Unflatten to 3D Grid
        shape_moved = t_moved.shape
        grid_shape = (L, L, L, dof) + shape_moved[1:]
        grid_tensor = jnp.reshape(t_moved, grid_shape)
        
        # Periodic Boundary Shifts
        shifted_grid = grid_tensor
        for i in range(3):
            shifted_grid = jax.lax.cond(
                delta[i] != 0,
                lambda g, d_val, ax_idx: jnp.roll(g, shift=d_val, axis=ax_idx),
                lambda g, d_val, ax_idx: g,
                shifted_grid, delta[i], i
            )
            
        shifted_flat = jnp.reshape(shifted_grid, shape_moved)
        return jnp.moveaxis(shifted_flat, 0, axis)
    
    return jax.lax.cond(is_zero_shift, lambda t: t, do_shift, tensor)

def parse_stamp_to_rules(stamp):
    """
    Converts dense stamp matrices into a static list of valid scattering rules.
    This prevents JAX from trying to compile dynamic NaN-filtering logic.
    """
    deltas, weights = stamp
    rules = []
    for d, W in zip(deltas, weights):
        nz = np.where(~np.isnan(W))
        for combo in zip(*nz):
            # Tuple of: (delta_matrix, internal_spin_indices, weight_value)
            rules.append((tuple(map(tuple, d)), tuple(combo), float(W[combo])))
    return tuple(rules) # Must be a tuple to be hashable for static_argnames

@partial(jax.jit, static_argnames=("t_str", "v_str", "out_str", "stamp_rules", "L", "dof"))
def stamp_einsum(
    t_str: str, 
    v_str: str, 
    out_str: str, 
    tensor: jax.Array, 
    out_tensor: jax.Array, 
    stamp_rules: tuple, 
    is_p: jax.Array, 
    local_map: jax.Array, 
    L: int, 
    dof: int = 4,
    factor: float = 1.0
) -> jax.Array:
    """
    Subspace-aware Matrix-Free Contraction.
    """
    # CC Conventions determine the masks dynamically
    req_p = [c in "abcd" for c in v_str]
    free_idx = [c for c in out_str if c not in v_str]
    has_free = len(free_idx) > 0
    if has_free:
        free_pos = out_str.index(free_idx[0])

    # Unroll the static rules entirely at compile-time
    for d, combo, val in stamp_rules:
        idx = get_global_indices_jax(L, dof, d, combo)
        
        # Enforce Topography (Masking)
        mask = jnp.ones(L**3, dtype=bool)
        for i in range(len(v_str)):
            mask &= is_p[idx[:, i]] if req_p[i] else ~is_p[idx[:, i]]
            
        # Map Global to P/H Subspace
        loc = {c: jnp.where(mask, local_map[idx[:, i]], 0) for i, c in enumerate(v_str)}
        
        # Gather T
        t_idx = tuple(loc[c] if c in loc else slice(None) for c in t_str)
        t_val = tensor[t_idx] 
        
        # Compute Algebraic Update
        update = (val * factor) * (mask[:, None] if has_free else mask) * t_val
        
        # Scatter to Output
        if not has_free:
            # 0D update: scatter point-to-point
            out_tensor = out_tensor.at[loc[out_str[0]], loc[out_str[1]]].add(update)
        else:
            # 1D update: scatter array slices
            bound_char = out_str[1] if free_pos == 0 else out_str[0]
            if free_pos == 0:
                out_tensor = out_tensor.at[:, loc[bound_char]].add(update.T)
            else:
                out_tensor = out_tensor.at[loc[bound_char], :].add(update)
                
    return out_tensor

@partial(jax.jit, static_argnames=("L", "stamp_rules"))
def stamp_t1(
    t1: jax.Array, 
    t2: jax.Array, 
    f_ph: jax.Array, 
    f_pp: jax.Array, 
    f_hh: jax.Array, 
    stamp_rules: tuple, 
    is_p: jax.Array, 
    local_map: jax.Array, 
    L: int
) -> jax.Array:
    
    # We define the shapes dynamically from the dense fock mats
    P, H = f_ph.shape
    
    # --- H1 Residuals ---
    H1 = f_ph
    H1 = stamp_einsum("ck", "akci", "ai", t1, H1, stamp_rules, is_p, local_map, L, factor=-1.0)
    H1 += jnp.einsum("ck, acik -> ai", f_ph, t2)
    H1 = stamp_einsum("cakl", "cikl", "ai", t2, H1, stamp_rules, is_p, local_map, L, factor=-0.5)

    I_dl = stamp_einsum("ck", "cdkl", "dl", t1, jnp.zeros((P, H)), stamp_rules, is_p, local_map, L)
    H1 += jnp.einsum("dl, dali -> ai", I_dl, t2)
    
    H1 = stamp_einsum("cdki", "cdak", "ai", t2, H1, stamp_rules, is_p, local_map, L, factor=-0.5)

    # --- X_hh Intermediates ---
    X_hh = -f_hh
    X_hh -= 0.5 * jnp.einsum("ck, ci -> ki", f_ph, t1)
    X_hh = stamp_einsum("bj", "bijk", "ki", t1, X_hh, stamp_rules, is_p, local_map, L, factor=-1.0)
    X_hh = stamp_einsum("cdli", "cdlk", "ki", t2, X_hh, stamp_rules, is_p, local_map, L, factor=-1.0)

    I_dk = stamp_einsum("cl", "cdlk", "dk", t1, jnp.zeros((P, H)), stamp_rules, is_p, local_map, L)
    X_hh -= 0.5 * jnp.einsum("dk, di -> ki", I_dk, t1)

    # --- X_pp Intermediates ---
    X_pp = f_pp
    X_pp -= 0.5 * jnp.einsum("ck, ak -> ac", f_ph, t1)
    X_pp = stamp_einsum("dakl", "dckl", "ac", t2, X_pp, stamp_rules, is_p, local_map, L, factor=-0.5)

    I_cl = stamp_einsum("dk", "cdkl", "cl", t1, jnp.zeros((P, H)), stamp_rules, is_p, local_map, L)
    X_pp += 0.5 * jnp.einsum("cl, al -> ac", I_cl, t1)

    X_pp = stamp_einsum("ck", "cdak", "ad", t1, X_pp, stamp_rules, is_p, local_map, L, factor=-1.0)

    # --- Finalize ---
    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    denom = jnp.diag(X_pp)[:, None] + jnp.diag(X_hh)[None, :]
    denom = jnp.where(denom == 0, 1e-10, denom)

    return t1 - (H1 / denom)
