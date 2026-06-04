"""
Module provides functions to build a many-body basis and to construct
Hamiltonian matrices.
Moved from lookup table to bitstrings
Symmetry sector filtering (apply tz and sz constraints during basis creation to avoid invalid configuration states)
"""
__authors__   = "Thomas Papenbrock"
__credits__   = ["Thomas Papenbrock"]
__copyright__ = "(c) Thomas Papenbrock"
__license__   = "BSD-3-Clause"
__date__      = "2026-06-05"

from itertools import combinations
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse as sci_sparse
from jax.experimental import sparse

class ManyBodyBasis(dict):
    """
    wrapper to give custom len func to dict
    """
    def __init__(self, bitstrings, dim, nstat):
        super().__init__()
        self["bitstrings"] = bitstrings
        self["dim"] = dim
        self["nstat"] = nstat

    def __len__(self):
        return self["dim"]

def get_many_body_states(basis, num_part, total_tz=None, total_sz=None):
    nstat = len(basis)
    combos = list(combinations(range(nstat), num_part))
    combos_arr = jnp.array(combos, dtype=jnp.int32)
    
    bitstrings = jnp.sum(1 << combos_arr, axis=-1, dtype=jnp.int64)
    basis_matrix = jnp.array(basis, dtype=jnp.int32)
    
    total_tz_arr = jnp.sum(basis_matrix[:, 3][combos_arr], axis=-1)
    total_tz_arr = 2 * total_tz_arr - num_part
    
    total_sz_arr = jnp.sum(basis_matrix[:, 4][combos_arr], axis=-1)
    total_sz_arr = 2 * total_sz_arr - num_part
    
    mask = jnp.ones(len(combos), dtype=jnp.bool_)
    if total_tz is not None:
        mask = mask & (total_tz_arr == total_tz)
    if total_sz is not None:
        mask = mask & (total_sz_arr == total_sz)
        
    filtered_bitstrings = bitstrings[mask]
    sort_idx = jnp.argsort(filtered_bitstrings)
    filtered_bitstrings = filtered_bitstrings[sort_idx]
    
    return ManyBodyBasis(filtered_bitstrings, len(filtered_bitstrings), nstat)

@jax.jit
def _process_operator_element(basis_bits, final_idx, initial_idx, val):
    initial_mask = (jnp.sum(1 << jnp.array(initial_idx))).astype(int)
    final_mask = (jnp.sum(1 << jnp.array(final_idx))).astype(int)
    
    has_initial = (basis_bits & initial_mask) == initial_mask
    basis_sub = basis_bits ^ initial_mask
    no_final_overlap = (basis_sub & final_mask) == 0
    
    valid = has_initial & no_final_overlap
    basis_new = jnp.where(valid, basis_sub | final_mask, 0)
    
    target_idx = jnp.searchsorted(basis_bits, basis_new)
    dim = len(basis_bits)
    idx_safe = jnp.clip(target_idx, 0, dim - 1)
    match = valid & (basis_bits[idx_safe] == basis_new)
    
    all_indices = list(initial_idx) + list(final_idx)
    total_count = jnp.zeros_like(basis_bits, dtype=jnp.int32)
    for k in all_indices:
        mask_below = (1 << k.astype(int)) - 1
        total_count += jax.lax.population_count(basis_sub & mask_below)
        
    sign = jnp.where(total_count % 2 == 0, 1.0, -1.0)
    
    row_indices = idx_safe
    col_indices = jnp.arange(dim, dtype=jnp.int32)
    data_values = val * sign
    valid_transitions = match
    
    return row_indices, col_indices, data_values, valid_transitions

def get_csr_matrix_scalar_op(lookup, operator, num_sp_stat):
    basis_bits = lookup["bitstrings"]
    dim = lookup["dim"]
    
    operator_arr = jnp.array(operator, dtype=jnp.float64)
    rank_op = (operator_arr.shape[1] - 1) // 2
    
    all_rows, all_cols, all_data = [], [], []
    
    for row in operator:
        val = row[-1]
        if rank_op == 1:
            final_idx = [int(row[0])]
            initial_idx = [int(row[1])]
        elif rank_op == 2:
            final_idx = [int(row[0]), int(row[1])]
            initial_idx = [int(row[2]), int(row[3])]
        elif rank_op == 3:
            final_idx = [int(row[0]), int(row[1]), int(row[2])]
            initial_idx = [int(row[3]), int(row[4]), int(row[5])]
        else:
            continue
            
        r, c, d, valid = _process_operator_element(basis_bits, final_idx, initial_idx, val)
        
        all_rows.append(r[valid])
        all_cols.append(c[valid])
        all_data.append(d[valid])
        
    if len(all_data) == 0:
        indices = jnp.zeros((0, 2), dtype=jnp.int32)
        data = jnp.zeros((0,), dtype=jnp.float64)
        return sparse.BCOO((data, indices), shape=(dim, dim))
        
    sparse_rows = jnp.concatenate(all_rows)
    sparse_cols = jnp.concatenate(all_cols)
    sparse_data = jnp.concatenate(all_data)
    
    indices = jnp.stack([sparse_rows, sparse_cols], axis=-1)
    bcoo_matrix = sparse.BCOO((sparse_data, indices), shape=(dim, dim))
    
    bcoo_matrix.sum_duplicates()

    # Bridge back to scipy sparse
    indices_np = np.array(bcoo_matrix.indices)
    data_np = np.array(bcoo_matrix.data)
    
    return sci_sparse.csr_matrix((data_np, (indices_np[:, 0], indices_np[:, 1])), shape=(dim, dim))
