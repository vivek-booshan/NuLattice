from dataclasses import dataclass, astuple

import numpy as np
import jax.numpy as jnp

from NuLattice.utils._jax_types import (
    OneBodyOperator,
    TwoBodyOperator,
    ThreeBodyOperator,
)

@dataclass
class Stamp:
    deltas: np.ndarray
    weights: np.ndarray

    # tuple unpacking
    def __iter__(self):
        return iter(astuple(self))

    # indexing
    def __getitem__(self, index):
        return astuple(self)[index]

    @property
    def rules(self):
        """
        Converts dense stamp matrices into a static list of valid scattering rules.
        This prevents JAX from trying to compile dynamic NaN-filtering logic.
        """
        rules = []
        for d, W in zip(self.deltas, self.weights):
            nz = np.where(~np.isnan(W))
            for combo in zip(*nz):
                # Tuple of: (delta_matrix, internal_spin_indices, weight_value)
                rules.append((tuple(map(tuple, d)), tuple(combo), float(W[combo])))
        return tuple(rules) # Must be a tuple to be hashable for static_argnames

@dataclass
class Stamper:
    L: int
    spin: int
    isospin: int

    def stamp(self, vT1, vS1, v3NF=None):
        stamp_1b = self.stamp_1b()
        stamp_2b = self.stamp_2b(vT1, vS1)
        stamp_3b = self.stamp_3b(v3NF) if v3NF else None

        return stamp_1b, stamp_2b, stamp_3b

    def stamp_1b(self) -> Stamp:
        deltas = []
        weights = []
        I4 = np.eye(self.spin * self.isospin, dtype=np.float64)

        # On-site diagonal
        deltas.append([0, 0, 0])
        weights.append(6.0 * I4)

        # Nearest-neighbor hopping (3D)
        for dim in range(3):
            for direction in [1, -1]:
                shift = [0, 0, 0]
                shift[dim] = direction
                deltas.append(shift)
                weights.append(-1.0 * I4)

        deltas = np.array(deltas)
        weights = np.array(weights)
        return Stamp(deltas, weights)

    def stamp_2b(self, vT1: float, vS1: float) -> Stamp:
        num_local_states = self.spin * self.isospin
        W = np.full((num_local_states, num_local_states, num_local_states, num_local_states), np.nan, dtype=np.float64)
        # p > q & r > s
        for p in range(4):
            for q in range(p + 1, 4):
                tz_p, sz_p = divmod(p, 2)
                tz_q, sz_q = divmod(q, 2)
                for r in range(4):
                    for s in range(r + 1, 4):
                        tz_r, sz_r = divmod(r, 2)
                        tz_s, sz_s = divmod(s, 2)

                        # Conservation of total Tz and Sz
                        if tz_p + tz_q != tz_r + tz_s:
                            continue
                        if sz_p + sz_q != sz_r + sz_s:
                            continue

                        if tz_p == tz_q:
                            val = vT1
                        elif sz_p == sz_q:
                            val = vS1
                        else:
                            val = (vS1 + vT1) * 0.5 if (p == r) else (vS1 - vT1) * 0.5

                        W[p, q, r, s] = val

        deltas = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.int64)
        weights = np.array([W], dtype=np.float64)
        return Stamp(deltas, weights)

    def stamp_3b(self, v3NF: float) -> Stamp:
        num_local_states = self.spin * self.isospin

        W = np.full(
            (num_local_states, num_local_states, num_local_states, 
             num_local_states, num_local_states, num_local_states), 
            np.nan, dtype=np.float64
        )

        # p > q > r
        for p in range(num_local_states):
            for q in range(p + 1, num_local_states):
                for r in range(q + 1, num_local_states):
                    W[p, q, r, p, q, r] = v3NF

        deltas = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]], dtype=np.int64)
        weights = np.array([W], dtype=np.float64)

        return Stamp(deltas, weights)

def stamp_to_one_body(
    deltas: np.ndarray, weights: np.ndarray, L: int
) -> OneBodyOperator:
    nstat = (L**3) * 4
    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T

    i_stride = L * L * 4
    j_stride = L * 4
    k_stride = 4
    strides = np.array([i_stride, j_stride, k_stride])

    all_indices, all_values = [], []

    for d, W in zip(deltas, weights):
        t_coords = (spatial_coords + d) % L

        p_spatial_base = np.sum(spatial_coords * strides, axis=1)
        q_spatial_base = np.sum(t_coords * strides, axis=1)

        for a in range(4):
            for b in range(4):
                if W[a, b] != 0:
                    p_idx = p_spatial_base + a
                    q_idx = q_spatial_base + b

                    idx_pair = np.column_stack([p_idx, q_idx])
                    vals = np.full(L**3, W[a, b])

                    all_indices.append(idx_pair)
                    all_values.append(vals)

    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    # Lexicographical sort to ensure exact match with legacy output ordering
    sort_idx = np.lexsort((final_indices[:, 1], final_indices[:, 0]))

    return OneBodyOperator(final_indices[sort_idx], final_values[sort_idx], nstat)


def stamp_to_two_body(
    deltas: np.ndarray, weights: np.ndarray, L: int, spin: int = 2, isospin: int = 2
): # -> TwoBodyOperator 
    num_local_states = spin * isospin
    basis_size = (L**3) * num_local_states

    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    N_sites = L**3

    i_stride = L * L * num_local_states
    j_stride = L * num_local_states
    k_stride = num_local_states
    strides = np.array([i_stride, j_stride, k_stride])

    all_indices = []
    all_values = []

    for delta_matrix, W in zip(deltas, weights):
        dq, dr, ds = delta_matrix

        # Calculate periodic boundaries for q, r, s based on their shift from p
        p_coords = spatial_coords
        q_coords = (spatial_coords + dq) % L
        r_coords = (spatial_coords + dr) % L
        s_coords = (spatial_coords + ds) % L

        p_base = np.sum(p_coords * strides, axis=1)
        q_base = np.sum(q_coords * strides, axis=1)
        r_base = np.sum(r_coords * strides, axis=1)
        s_base = np.sum(s_coords * strides, axis=1)

        # Retrieve the coordinates of non-zero internal weights in the 4D Tensor
        valid_mask = ~np.isnan(W)
        nz_p, nz_q, nz_r, nz_s = np.where(valid_mask)

        for a, b, c, d in zip(nz_p, nz_q, nz_r, nz_s):
            val = W[a, b, c, d]

            p_idx = p_base + a
            q_idx = q_base + b
            r_idx = r_base + c
            s_idx = s_base + d

            idx_tuple = np.column_stack([p_idx, q_idx, r_idx, s_idx])
            vals = np.full(N_sites, val)

            all_indices.append(idx_tuple)
            all_values.append(vals)

    if not all_indices:
        # Edge case handling if no interactions are generated
        return TwoBodyOperator(np.empty((0, 4), dtype=int), np.empty(0, dtype=float), basis_size)

    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    # Standard lexicographical sort (p, then q, then r, then s) to ensure exact match with legacy output ordering
    sort_idx = np.lexsort((
        final_indices[:, 3],
        final_indices[:, 2],
        final_indices[:, 1],
        final_indices[:, 0]
    ))

    return TwoBodyOperator(final_indices[sort_idx], final_values[sort_idx], basis_size)


def stamp_to_three_body(
    deltas: np.ndarray, weights: np.ndarray, L: int, spin: int = 2, isospin: int = 2
): # -> ThreeBodyOperator
    num_local_states = spin * isospin
    basis_size = (L**3) * num_local_states
    
    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    N_sites = L**3

    i_stride = L * L * num_local_states
    j_stride = L * num_local_states
    k_stride = num_local_states
    strides = np.array([i_stride, j_stride, k_stride])

    all_indices = []
    all_values = []

    for delta_matrix, W in zip(deltas, weights):
        dq, dr, ds, dt, du = delta_matrix

        # Calculate periodic boundaries based on shifts from p
        p_coords = spatial_coords
        q_coords = (spatial_coords + dq) % L
        r_coords = (spatial_coords + dr) % L
        s_coords = (spatial_coords + ds) % L
        t_coords = (spatial_coords + dt) % L
        u_coords = (spatial_coords + du) % L

        p_base = np.sum(p_coords * strides, axis=1)
        q_base = np.sum(q_coords * strides, axis=1)
        r_base = np.sum(r_coords * strides, axis=1)
        s_base = np.sum(s_coords * strides, axis=1)
        t_base = np.sum(t_coords * strides, axis=1)
        u_base = np.sum(u_coords * strides, axis=1)

        # 4. Extract valid indices by finding what IS NOT NaN
        valid_mask = ~np.isnan(W)
        nz_p, nz_q, nz_r, nz_s, nz_t, nz_u = np.where(valid_mask)

        for a, b, c, e, f, g in zip(nz_p, nz_q, nz_r, nz_s, nz_t, nz_u):
            val = W[a, b, c, e, f, g]

            p_idx = p_base + a
            q_idx = q_base + b
            r_idx = r_base + c
            s_idx = s_base + e
            t_idx = t_base + f
            u_idx = u_base + g

            idx_hex = np.column_stack([p_idx, q_idx, r_idx, s_idx, t_idx, u_idx])
            vals = np.full(N_sites, val)

            all_indices.append(idx_hex)
            all_values.append(vals)

    if not all_indices:
        return ThreeBodyOperator(np.empty((0, 6), dtype=np.int64), np.empty(0, dtype=float), basis_size)

    final_indices = np.vstack(all_indices)
    final_values = np.concatenate(all_values)

    # Full Lexicographical sort to ensure exact match with legacy ordering
    sort_idx = np.lexsort((
        final_indices[:, 5],
        final_indices[:, 4],
        final_indices[:, 3],
        final_indices[:, 2],
        final_indices[:, 1],
        final_indices[:, 0]
    ))

    return ThreeBodyOperator(final_indices[sort_idx], final_values[sort_idx], basis_size)


# NOTE: this should be only one after jax/numpy mixed usage resolved
def get_global_indices_jax(L, dof, delta, combo):
    delta = jnp.atleast_2d(delta)
    spatial = jnp.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = jnp.array([L**2, L, 1]) * dof
    
    indices = [jnp.sum(spatial * strides, axis=1) + combo[0]]
    for i, d in enumerate(delta):
        shifted = (spatial + jnp.array(d)) % L
        indices.append(jnp.sum(shifted * strides, axis=1) + combo[i+1])
        
    return jnp.column_stack(indices)

def get_global_indices_np(
    L: int, num_local_states: int, deltas: np.ndarray, internal_combo: list
) -> np.ndarray:
    """
    Translates local topological shifts into global 1D array indices.

    deltas: (N-1, 3)array of spatial shifts relative to the primary particle.
    internal_combo: List of length N containing the spin/isospin indices (e.g., [p, q, r, s]).
    """
    deltas = np.atleast_2d(deltas)
    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = np.array([L**2, L, 1]) * num_local_states

    global_idx = []

    # 1. Base particle (Implicit delta = [0, 0, 0])
    base_idx = np.sum(spatial_coords * strides, axis=1) + internal_combo[0]
    global_idx.append(base_idx)

    # 2. Shifted particles
    for i, d in enumerate(deltas):
        shifted_coords = (spatial_coords + d) % L
        idx = np.sum(shifted_coords * strides, axis=1) + internal_combo[i + 1]
        global_idx.append(idx)

    return np.column_stack(global_idx)
