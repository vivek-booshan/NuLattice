import numpy as np
from collections import defaultdict

from NuLattice.utils._types import ThreeBodyOperator


def shortRangeV_3body(myL, sL, sNL, c0, a_lat, spin=2, isospin=2, min_val=1e-12):
    scale = c0 / a_lat

    def get_offsets(smear_val):
        O = {(0, 0, 0): 1.0}
        if smear_val != 0:
            for d in [
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ]:
                O[d] = smear_val
        return O

    O_V = get_offsets(sNL)
    O_S = get_offsets(sL)

    def add_3d(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def sub_3d(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def mod_3d(a):
        return (a[0] % myL, a[1] % myL, a[2] % myL)

    # Calculate 1-body local transition map U(dA, dD)
    U = defaultdict(float)
    for d3, s_val in O_S.items():
        for v1, v1_val in O_V.items():
            for v2, v2_val in O_V.items():
                dA = add_3d(d3, v1)
                dD = add_3d(d3, v2)
                U[(dA, dD)] += s_val * v1_val * v2_val

    # Filter out near-zero transitions to aggressively prune the combinatorics
    U_clean = {k: v for k, v in U.items() if abs(v) > 1e-12}
    U_items = list(U_clean.items())
    K = len(U_items)

    # Build the 3-body spatial diagram map W
    W = defaultdict(float)
    for i in range(K):
        (dA, dD), u1 = U_items[i]
        for j in range(K):
            (dB, dE), u2 = U_items[j]
            for k in range(K):
                (dC, dF), u3 = U_items[k]

                # Shift all coordinates so dA is at the origin (0,0,0)
                d1 = mod_3d(sub_3d(dB, dA))
                d2 = mod_3d(sub_3d(dC, dA))
                d3 = mod_3d(sub_3d(dD, dA))
                d4 = mod_3d(sub_3d(dE, dA))
                d5 = mod_3d(sub_3d(dF, dA))

                W[(d1, d2, d3, d4, d5)] += u1 * u2 * u3

    L3 = myL**3
    rA = np.arange(L3)
    xA = rA // (myL**2)
    yA = (rA // myL) % myL
    zA = rA % myL

    # states array for Particle 1, 2, and 3
    internal_dim = spin * isospin
    int_arr = np.arange(internal_dim)
    i1, i2, i3 = np.meshgrid(int_arr, int_arr, int_arr, indexing="ij")
    i1 = i1.flatten()
    i2 = i2.flatten()
    i3 = i3.flatten()

    all_A, all_B, all_C, all_D, all_E, all_F, all_V = [], [], [], [], [], [], []

    for (d1, d2, d3, d4, d5), weight in W.items():
        if abs(weight) < 1e-12:
            continue

        val_scalar = weight * scale

        xB = (xA + d1[0]) % myL
        yB = (yA + d1[1]) % myL
        zB = (zA + d1[2]) % myL
        rB = xB * (myL**2) + yB * myL + zB

        xC = (xA + d2[0]) % myL
        yC = (yA + d2[1]) % myL
        zC = (zA + d2[2]) % myL
        rC = xC * (myL**2) + yC * myL + zC

        xD = (xA + d3[0]) % myL
        yD = (yA + d3[1]) % myL
        zD = (zA + d3[2]) % myL
        rD = xD * (myL**2) + yD * myL + zD

        xE = (xA + d4[0]) % myL
        yE = (yA + d4[1]) % myL
        zE = (zA + d4[2]) % myL
        rE = xE * (myL**2) + yE * myL + zE

        xF = (xA + d5[0]) % myL
        yF = (yA + d5[1]) % myL
        zF = (zA + d5[2]) % myL
        rF = xF * (myL**2) + yF * myL + zF

        # Broadcast spatial vectors with internal degrees of freedom
        # and flatten instantly to 1D index arrays
        A = (rA[:, None] * internal_dim + i1[None, :]).flatten()
        B = (rB[:, None] * internal_dim + i2[None, :]).flatten()
        C = (rC[:, None] * internal_dim + i3[None, :]).flatten()
        D = (rD[:, None] * internal_dim + i1[None, :]).flatten()
        E = (rE[:, None] * internal_dim + i2[None, :]).flatten()
        F = (rF[:, None] * internal_dim + i3[None, :]).flatten()

        # Normal Ordering Filter: D < E < F and no repeated outgoing states
        mask = (D < E) & (E < F) & (A != B) & (B != C) & (A != C)
        if not np.any(mask):
            continue

        Am, Bm, Cm = A[mask], B[mask], C[mask]
        Dm, Em, Fm = D[mask], E[mask], F[mask]
        Vm = np.full(Am.shape, val_scalar, dtype=np.float64)

        diff1 = np.sign(Bm - Am)
        diff2 = np.sign(Cm - Am)
        diff3 = np.sign(Cm - Bm)
        parity = diff1 * diff2 * diff3

        Vm *= parity

        # Sort A, B, C along the feature axis to match output requirements
        stacked_ABC = np.vstack([Am, Bm, Cm])
        sorted_ABC = np.sort(stacked_ABC, axis=0)
        Am, Bm, Cm = sorted_ABC[0], sorted_ABC[1], sorted_ABC[2]

        all_A.append(Am)
        all_B.append(Bm)
        all_C.append(Cm)
        all_D.append(Dm)
        all_E.append(Em)
        all_F.append(Fm)
        all_V.append(Vm)

    raw_indices = np.column_stack(
        [
            np.concatenate(all_A),
            np.concatenate(all_B),
            np.concatenate(all_C),
            np.concatenate(all_D),
            np.concatenate(all_E),
            np.concatenate(all_F),
        ]
    )
    raw_vals = np.concatenate(all_V)

    # Because different lattice centers (rA) can scatter identical global states,
    # must uniquely group the indices and sum their values.
    unique_indices, unique_inv = np.unique(raw_indices, axis=0, return_inverse=True)
    summed_values = np.bincount(unique_inv, weights=raw_vals)

    nz_mask = np.abs(summed_values) > min_val

    nstat = L3 * internal_dim
    return ThreeBodyOperator(unique_indices[nz_mask], summed_values[nz_mask], nstat)
