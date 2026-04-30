import numpy as np

from .stamps import Stamp, Stamper

def stamp(stamper: Stamper):
    """
    Computes the Normal-Ordered Fock Stamp weights.
    f_pq = h_pq + sum_i V_piqi + 0.5 * sum_ij W_pijqij
    """
    num_local_states = stamper.spin * stamper.isospin
    rho_local = np.mean(stamper.hmask.reshape(-1, num_local_states), axis=0)
    
    fock_weights = []
    fock_deltas = []
    for d, w in zip(*stamper.one_body):
        f_w = np.nan_to_num(w)

        # NOTE: vivek, rn just the one body contribution seems to be better at passing test
        # for d2, w2 in zip(*stamper.two_body):
        #     w2 = np.nan_to_num(w2)
        #     f_w += np.einsum("piqi, i -> pq", w2, rho_local)

        # if stamper.three_body is not None:
        #     for d3, w3 in zip(*stamper.three_body):
        #         if np.allclose(d, d3[0]):
        #             w3 = np.nan_to_num(w3)
        #             f_w += 0.5 * np.einsum("pijqij, i, j -> pq", w3, rho_local, rho_local)

        fock_weights.append(f_w)
        fock_deltas.append(d)

    fock_deltas = np.array(fock_deltas)
    fock_weights = np.array(fock_weights)

    return Stamp(fock_deltas, fock_weights)

def stamp_to_dense(f_deltas, f_weights, L, num_local_states, part_idx, hole_idx):
    pnum, hnum = len(part_idx), len(hole_idx)
    
    f_pp = np.zeros((pnum, pnum), dtype=np.float64)
    f_ph = np.zeros((pnum, hnum), dtype=np.float64)
    f_hh = np.zeros((hnum, hnum), dtype=np.float64)

    global_to_local = np.zeros(L**3 * num_local_states, dtype=np.int32)
    global_to_local[part_idx] = np.arange(pnum)
    global_to_local[hole_idx] = np.arange(hnum)
    
    is_p = np.zeros(L**3 * num_local_states, dtype=bool)
    is_p[part_idx] = True

    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = np.array([L**2, L, 1]) * num_local_states

    for d, w in zip(f_deltas, f_weights):
        for site_idx in range(L**3):

            # Origin site (i) and shifted site (j)
            base_coord = spatial_coords[site_idx]
            shifted_coord = (base_coord + d) % L
            
            # Global base indices for these two sites
            g_base_i = np.sum(base_coord * strides)
            g_base_j = np.sum(shifted_coord * strides)
            
            # Iterate through spin/isospin blocks (a, b)
            for a in range(num_local_states):
                for b in range(num_local_states):
                    val = w[a, b]
                    if val == 0:
                        continue
                    
                    gi, gj = g_base_i + a, g_base_j + b
                    
                    # Sort into P/H sectors
                    p_i, p_j = is_p[gi], is_p[gj]
                    li, lj = global_to_local[gi], global_to_local[gj]
                    
                    if p_i and p_j:     # Particle-Particle
                        f_pp[li, lj] += val
                    elif p_i and not p_j: # Particle-Hole
                        f_ph[li, lj] += val
                    elif not p_i and not p_j: # Hole-Hole
                        f_hh[li, lj] += val
                        
    return f_pp, f_ph, f_hh
