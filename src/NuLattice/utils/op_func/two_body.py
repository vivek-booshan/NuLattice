import numpy as np
from collections import defaultdict

import NuLattice.constants as consts
from NuLattice.utils._types import TwoBodyOperator


def _get_pauli_half():
    s_x = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    s_y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    s_z = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    return np.array([s_x, s_y, s_z])


def _build_internal_operators():
    I_2 = np.eye(2, dtype=complex)
    pauli_half = _get_pauli_half()

    # 4x4 1-body operators in the (tz, sz) basis
    spin_ops = np.array([np.kron(I_2, p) for p in pauli_half])
    iso_ops = np.array([np.kron(p, I_2) for p in pauli_half])

    # 16x16 2-body Isospin dot product: (tau_1 . tau_2)
    # tau_1 acts on particle 1, tau_2 acts on particle 2.
    # The 2-body space is 4x4 (p1) tensor 4x4 (p2) = 16x16.
    tau_dot_tau = np.zeros((16, 16), dtype=complex)
    for t in iso_ops:
        # Since these are half-paulis, we must multiply by 4 to get standard tau.tau
        tau_dot_tau += 4.0 * np.kron(t, t)

    # 16x16 2-body Spin outer products: S_kron_S[i, j] = sigma_i (x) sigma_j
    S_kron_S = np.zeros((3, 3, 16, 16), dtype=complex)
    for i in range(3):
        for j in range(3):
            # NOTE: multiply this by 4?
            S_kron_S[i, j] = 4.0 * np.kron(spin_ops[i], spin_ops[j])

    return S_kron_S, tau_dot_tau


def f_SS(myL, bpi, a_lat):
    m_pi = consts.M_PI_0 * a_lat
    q = np.fft.fftfreq(myL, 1 / (2.0 * np.pi))
    qx, qy, qz = np.meshgrid(q, q, q, indexing="ij")
    q2 = qx**2 + qy**2 + qz**2
    ft_fss = np.exp(-bpi * q2) / (q2 + m_pi**2)

    q_vec = np.array([qx, qy, qz])
    fSS = np.zeros((3, 3, myL, myL, myL), dtype=complex)
    for s1 in range(3):
        for s2 in range(3):
            fSS[s1, s2] = np.fft.ifftn(q_vec[s1] * q_vec[s2] * ft_fss)
    return fSS


def onePionEx(myL, bpi, a_lat, mult=1):
    scale = -((consts.G_A / (2.0 * a_lat * consts.F_PI)) ** 2) * mult / 2.0
    fSS_grid = f_SS(myL, bpi, a_lat)  # Shape: (3, 3, L, L, L)

    # Build the internal 16x16 transition operators
    S_kron_S, tau_dot_tau = _build_internal_operators()

    # Contract Spin, Isospin, and Spatial components into a single tensor
    # Shape of V_internal: (L, L, L, 16, 16)
    # 16x16 interaction matrix for any distance (dx, dy, dz)
    V_internal = np.zeros((myL, myL, myL, 16, 16), dtype=complex)
    for i in range(3):
        for j in range(3):
            # O_ij = (sigma_i x sigma_j) * (tau . tau)
            O_ij = np.dot(S_kron_S[i, j], tau_dot_tau)
            # Broadcast scalar spatial grid against the 16x16 operator
            V_internal += fSS_grid[i, j, :, :, :, np.newaxis, np.newaxis] * O_ij

    V_internal *= scale / a_lat

    L3 = myL**3
    r = np.arange(L3)
    z = r % myL
    y = (r // myL) % myL
    x = r // (myL**2)

    dx = (x[:, None] - x[None, :]) % myL
    dy = (y[:, None] - y[None, :]) % myL
    dz = (z[:, None] - z[None, :]) % myL

    # V_dense shape: (L^3, L^3, 16, 16).
    # V_dense[r1, r2] gives the 16x16 internal transition matrix.
    V_dense = V_internal[dx, dy, dz]

    return _antisymmetrize_and_extract(V_dense, myL)


def shortRangeV_2body(myL, sL, sNL, c0, a_lat):
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

    O_V = get_offsets(sNL)  # Non-local smearing bounds
    O_S = get_offsets(sL)  # Local smearing bounds

    def add_3d(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def sub_3d(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def mod_3d(a):
        return (a[0] % myL, a[1] % myL, a[2] % myL)

    # U(dA, dC) represents the smeared density matrix element for a single particle
    U = defaultdict(float)
    for d3, s_val in O_S.items():
        for v1, v1_val in O_V.items():
            for v2, v2_val in O_V.items():
                delta_A = add_3d(d3, v1)
                delta_C = add_3d(d3, v2)
                U[(delta_A, delta_C)] += s_val * v1_val * v2_val

    # W holds the unique non-zero spatial diagrams for the 2-body interaction
    W = defaultdict(float)
    for (dA, dC), u1 in U.items():
        for (dB, dD), u2 in U.items():
            # Shift the coordinate system so dA is the origin (0,0,0)
            d1 = mod_3d(sub_3d(dB, dA))
            d2 = mod_3d(sub_3d(dC, dA))
            d3 = mod_3d(sub_3d(dD, dA))
            W[(d1, d2, d3)] += u1 * u2

    L3 = myL**3
    rA = np.arange(L3)
    xA = rA // (myL**2)
    yA = (rA // myL) % myL
    zA = rA % myL

    # Density operators preserve spin/isospin exactly.
    # A and C share an internal state; B and D share an internal state.
    int1 = np.arange(4)
    int2 = np.arange(4)

    all_A, all_B, all_C, all_D, all_V = [], [], [], [], []

    for (d1, d2, d3), weight in W.items():
        if abs(weight) < 1e-12:
            continue

        val = weight * scale

        # Shift coordinates using periodic boundaries
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

        # Broadcast spatial shifts against the 4x4 internal states
        # Shape becomes (L^3, 4, 4) -> flatten to 1D
        A = (rA[:, None, None] * 4 + int1[None, :, None]).flatten()
        B = (rB[:, None, None] * 4 + int2[None, None, :]).flatten()
        C = (rC[:, None, None] * 4 + int1[None, :, None]).flatten()
        D = (rD[:, None, None] * 4 + int2[None, None, :]).flatten()

        # Normal Ordering / Antisymmetrization Filters
        mask_fwd = (A < B) & (C < D)
        mask_exc = (A < B) & (D < C)

        if np.any(mask_fwd):
            all_A.append(A[mask_fwd])
            all_B.append(B[mask_fwd])
            all_C.append(C[mask_fwd])
            all_D.append(D[mask_fwd])
            all_V.append(np.full(np.sum(mask_fwd), val, dtype=np.float64))

        if np.any(mask_exc):
            all_A.append(A[mask_exc])
            all_B.append(B[mask_exc])
            all_C.append(D[mask_exc])  # Exchange swaps C and D
            all_D.append(C[mask_exc])
            all_V.append(np.full(np.sum(mask_exc), -val, dtype=np.float64))

    final_A = np.concatenate(all_A)
    final_B = np.concatenate(all_B)
    final_C = np.concatenate(all_C)
    final_D = np.concatenate(all_D)
    final_V = np.concatenate(all_V)

    nstat = L3 * 4
    indices = np.column_stack([final_A, final_B, final_C, final_D])

    return TwoBodyOperator(indices, final_V, nstat)


def _antisymmetrize_and_extract(V_dense, myL):
    """
    Replaces the massive sparse matrix memory bloat.
    Antisymmetrizes V^ab_cd directly in dense space, filters for
    a < b and c < d, and returns flattened 1D index arrays ready for JAX BCOO.
    """
    L3 = myL**3

    # Unpack the 4D tensor (L^3, L^3, 16, 16) into global (A, B, C, D) indices
    # internal incoming states: p_in = (c % 4) * 4 + (d % 4)
    # internal outgoing states: p_out = (a % 4) * 4 + (b % 4)

    # Create global index grids
    r1, r2 = np.meshgrid(np.arange(L3), np.arange(L3), indexing="ij")
    int_out, int_in = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")

    c_int = int_in // 4
    d_int = int_in % 4
    a_int = int_out // 4
    b_int = int_out % 4

    A = r1[:, :, None, None] * 4 + a_int[None, None, :, :]
    B = r2[:, :, None, None] * 4 + b_int[None, None, :, :]
    C = r1[:, :, None, None] * 4 + c_int[None, None, :, :]
    D = r2[:, :, None, None] * 4 + d_int[None, None, :, :]

    # Mask for non-zero entries to save memory before antisymmetrization
    nz_mask = np.abs(V_dense) > 1e-10

    A_nz = A[nz_mask]
    B_nz = B[nz_mask]
    C_nz = C[nz_mask]
    D_nz = D[nz_mask]
    V_nz = V_dense[nz_mask]

    # Apply Normal Ordering Constraints
    # Only A < B and C < D.
    # Antisymmetrization: V_AS = V_abcd - V_abdc

    # Forward term: V_abcd
    mask_fwd = (A_nz < B_nz) & (C_nz < D_nz)

    # Exchange term: -V_abdc
    mask_exc = (A_nz < B_nz) & (D_nz < C_nz)

    A_final = np.concatenate([A_nz[mask_fwd], A_nz[mask_exc]])
    B_final = np.concatenate([B_nz[mask_fwd], B_nz[mask_exc]])

    # C and D are swapped for the exchange term
    C_final = np.concatenate([C_nz[mask_fwd], D_nz[mask_exc]])
    D_final = np.concatenate([D_nz[mask_fwd], C_nz[mask_exc]])

    # exchange term acquires a negative sign
    V_final = np.concatenate([V_nz[mask_fwd], -V_nz[mask_exc]])

    return TwoBodyOperator(
        np.column_stack([A_final, B_final, C_final, D_final]), V_final, 4 * myL**3
    )
