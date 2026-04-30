from dataclasses import dataclass, astuple

import numpy as np
import jax.numpy as jnp

from NuLattice.jax.lattice import states2PHSpace

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
    Translates local topological shifts into global 1D array indices
    """
    deltas = np.atleast_2d(deltas)
    full_deltas = np.vstack([np.zeros((1, 3), dtype=int), deltas])
    
    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T[:, np.newaxis, :]
    
    shifted_coords = (spatial_coords + full_deltas) % L
    
    strides = np.array([L**2, L, 1]) * num_local_states
    spatial_part = np.sum(shifted_coords * strides, axis=2) # (L**3, N_legs)
    
    return spatial_part + np.array(internal_combo)

@dataclass(frozen=True)
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

# NOTE: getting monolithic
class Stamper:
    def __init__(self, L: int, spin: int, isospin: int):
        self.L = L
        self.spin = spin
        self.isospin = isospin
        self.one_body = None
        self.two_body = None
        self.three_body = None
        self.hmask = None
        self.pmask = None

    def get_global_indices(self, deltas, internal_combo):
        return get_global_indices_np(self.L, self.spin * self.isospin, deltas, internal_combo)

    def normal_order_masks(self, ref_state):
        assert self.one_body is not None, "one_body must not be NoneType"
        assert self.two_body is not None, "one_body must not be NoneType"

        num_local_states = self.spin * self.isospin
        nstat = (self.L**3) * num_local_states

        hole_idx, _ = states2PHSpace(ref_state, self.L)
        hole_idx = np.array(hole_idx)

        mask_H = np.zeros(nstat, dtype=np.float64)
        if len(hole_idx) > 0:
            mask_H[hole_idx] = 1.0

        mask_P = 1.0 - mask_H

        self.pmask = jnp.array(mask_P)
        self.hmask = jnp.array(mask_H)
        return self.pmask, self.hmask

    def get_reference_energy(self):
        assert self.hmask is not None, "self.hmask is NoneType. First call normal_order_masks"
        """
        Computes the vacuum reference energy directly from topological stamps.
        E_ref = <T> + <V> + <W> 
        """
        e_ref = 0.0
        mask_H = self.hmask

        for d, W in zip(*self.one_body):
            # A global diagonal element MUST have no spatial shift
            if not np.allclose(d, 0): 
                continue 
            
            nz = np.where(W != 0)
            for a, b in zip(*nz):
                if a != b:
                    continue  # Trace requires internal diagonal too
                
                idx = self.get_global_indices(d, [a, b])
                # Only need to check one mask since a=b and d=0 implies idx[:,0] == idx[:,1]
                e_ref += W[a, a] * np.sum(mask_H[idx[:, 0]])

        for d, W in zip(*self.two_body):
            if not np.allclose(d, 0): 
                continue
            
            nz = np.where(~np.isnan(W))
            for p, q, r, s in zip(*nz):
                # Trace requires p->p and q->q
                if p != r or q != s:
                    continue
                
                idx = self.get_global_indices(d, [p, q, p, q])
                overlap = mask_H[idx[:, 0]] * mask_H[idx[:, 1]]
            
                # Bare expectation <V>: 1/2 sum_ij V_ijij. 
                # Since V_ijij + V_jiji = 2*W, this resolves exactly to +1.0 * W
                e_ref += 1.0 * W[p, q, p, q] * np.sum(overlap)

        if self.three_body is not None:
            for d, W in zip(*self.three_body):
                if not np.allclose(d, 0): 
                    continue
                
                nz = np.where(~np.isnan(W))
                for p, q, r, s, t, u in zip(*nz):
                    if p != s or q != t or r != u:
                        continue
                    
                    idx = self.get_global_indices(d, [p, q, r, p, q, r])
                    overlap = mask_H[idx[:, 0]] * mask_H[idx[:, 1]] * mask_H[idx[:, 2]]
                
                    # Bare expectation <W>: 1/6 sum_ijk W_ijkijk. 
                    # The 3! permutations cancel the 1/6 factor to exactly +1.0 * W
                    e_ref += 1.0 * W[p, q, r, p, q, r] * np.sum(overlap)

        return float(e_ref)
        
        
    def stamp(self, vT1, vS1, v3NF=None):
        self.one_body = self._stamp_1b()
        self.two_body = self._stamp_2b(vT1, vS1)
        self.three_body = self._stamp_3b(v3NF) if v3NF else None

        return self.one_body, self.two_body, self.three_body

    def _stamp_1b(self) -> Stamp:
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

    def _stamp_2b(self, vT1: float, vS1: float) -> Stamp:
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

    def _stamp_3b(self, v3NF: float) -> Stamp:
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
