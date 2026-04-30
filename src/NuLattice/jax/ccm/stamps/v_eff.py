import numpy as np
from .stamps import Stamp, Stamper

def stamp(stamper: Stamper):
    """
    v_pqrs = v_pqrs + sum_i w_pqirsi
    """

    num_local_states = stamper.spin * stamper.isospin
    rho_local = np.mean(stamper.hmask.reshape(-1, num_local_states), axis=0)

    v_eff_weights = []
    v_eff_deltas = []
    for d, w in zip(*stamper.two_body):    
        v_w = np.nan_to_num(w) 

        if stamper.three_body:
            for d3, w3 in zip(*stamper.three_body):
                v_w += np.einsum("pqirsi, i -> pqrs", w3, rho_local)

        v_eff_weights.append(v_w)
        v_eff_deltas.append(d)

    v_eff_deltas = np.array(v_eff_deltas)
    v_eff_weights = np.array(v_eff_weights)

    return Stamp(v_eff_deltas, v_eff_weights)

def stamp_to_dense(v_deltas, v_weights, L, num_local_states, part_idx, hole_idx):
    pnum, hnum = len(part_idx), len(hole_idx)
    nstat = L**3 * num_local_states
    
    v_pphh = np.zeros((pnum, pnum, hnum, hnum))
    v_phph = np.zeros((pnum, hnum, pnum, hnum))
    v_phhh = np.zeros((pnum, hnum, hnum, hnum))
    v_hhhh = np.zeros((hnum, hnum, hnum, hnum))

    global_to_local = np.zeros(nstat, dtype=np.int64)
    global_to_local[part_idx] = np.arange(pnum)
    global_to_local[hole_idx] = np.arange(hnum)
    
    mask_P = np.zeros(nstat, dtype=bool)
    mask_P[part_idx] = True

    spatial_coords = np.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = np.array([L**2, L, 1]) * num_local_states

    for d, W in zip(v_deltas, v_weights):
        # d is the vector [dx, dy, dz] between site1 and site2
        for site_idx in range(L**3):
            base_coord = spatial_coords[site_idx]
            shifted_coord = (base_coord + d) % L
            
            g_base_1 = np.sum(base_coord * strides)
            g_base_2 = np.sum(shifted_coord * strides)
            
            nz = np.where(np.abs(W) > 1e-15)
            for p, q, r, s in zip(*nz):
                val = W[p, q, r, s]
                
                # Global indices
                # NOTE: Stamper logic usually assumes site1 for (p,r) and site2 for (q,s)
                g1, g2 = g_base_1 + p, g_base_2 + q
                g3, g4 = g_base_1 + r, g_base_2 + s
                
                # Check P/H status
                P = [mask_P[g1], mask_P[g2], mask_P[g3], mask_P[g4]]
                score = P[0]*8 + P[1]*4 + P[2]*2 + P[3]*1
                
                m1, m2, m3, m4 = (
                    global_to_local[g1], global_to_local[g2], 
                    global_to_local[g3], global_to_local[g4]
                )
                
                if score == 12: # PPHH
                    v_pphh[m1, m2, m3, m4] += val
                elif score == 10: # PHPH
                    v_phph[m1, m2, m3, m4] += val
                elif score == 8: # PHHH
                    v_phhh[m1, m2, m3, m4] += val
                elif score == 0: # HHHH
                    v_hhhh[m1, m2, m3, m4] += val
                    
    return v_pphh, v_phph, v_phhh, v_hhhh


