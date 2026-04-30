from functools import partial

import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=("subscripts", "L", "dof", "factor"))
def stamp_einsum(
    subscripts: str,
    tensor: jax.Array,
    out_tensor: jax.Array,
    deltas: jax.Array,
    weights: jax.Array,
    is_p: jax.Array,
    local_map: jax.Array,
    L: int,
    dof: int = 4,
    factor: float = 1.0,
) -> jax.Array:

    lhs, out_str = subscripts.split("->")
    t_str, v_str = lhs.split(",")
    t_str, v_str, out_str = t_str.strip(), v_str.strip(), out_str.strip()

    req_p = {c: True for c in v_str if c in "abcd"}
    req_p.update({c: False for c in v_str if c in "ijkl"})

    t_adv = [c for c in t_str if c in v_str]
    out_adv = [c for c in out_str if c in v_str]

    def get_layout(string, adv_chars):
        if not adv_chars:
            return "Q" + string
        adv_ordered = [c for c in string if c in adv_chars]
        free_ordered = [c for c in string if c not in adv_chars]
        adv_idx = sorted([string.index(c) for c in adv_chars])

        if adv_idx == list(range(min(adv_idx), max(adv_idx) + 1)):
            return (
                "".join(free_ordered[: min(adv_idx)])
                + "Q"
                + "".join(adv_ordered)
                + "".join(free_ordered[min(adv_idx) :])
            )
        return "Q" + "".join(adv_ordered) + "".join(free_ordered)

    t_layout = get_layout(t_str, t_adv)
    out_layout = get_layout(out_str, out_adv)

    v_perms = [(v_str, 1.0)]
    if req_p[v_str[0]] == req_p[v_str[1]]:
        v_perms.append((v_str[1] + v_str[0] + v_str[2:], -1.0))
    if req_p[v_str[2]] == req_p[v_str[3]]:
        new_perms = [(s[:2] + s[3] + s[2], -f) for s, f in v_perms]
        v_perms.extend(new_perms)

    # Pre-compute spatial grid and reshapes ONCE outside the loop
    base_spatial = jnp.mgrid[0:L, 0:L, 0:L].reshape(3, -1).T
    strides = jnp.array([L**2, L, 1], dtype=jnp.int32)
    local_map_reshaped = local_map.reshape(L**3, dof)
    is_p_reshaped = is_p.reshape(L**3, dof)

    def scan_step(current_out, carry_in):
        d, W = carry_in
        W = jnp.nan_to_num(W)

        full_d = jnp.vstack([jnp.zeros((1, 3), dtype=jnp.int32), d])

        maps, masks = {}, {}
        for i, char in enumerate(v_str):
            if char in t_adv or char in out_adv:
                shift_vec = full_d[i]
                shifted = (base_spatial + shift_vec) % L
                shift_idx = jnp.sum(shifted * strides, axis=1)

                maps[char] = local_map_reshaped[shift_idx]
                raw_mask = is_p_reshaped[shift_idx]
                masks[char] = raw_mask if req_p[char] else ~raw_mask

        # gather
        t_idx = []
        for char in t_str:
            if char in t_adv:
                shape = [L**3] + [1] * len(t_adv)
                shape[t_adv.index(char) + 1] = dof
                t_idx.append(maps[char].reshape(shape))
            else:
                t_idx.append(slice(None))

        t_val = (
            tensor[tuple(t_idx)]
            if t_adv
            else jnp.broadcast_to(tensor, (L**3,) + tensor.shape)
        )

        valid_t = jnp.ones((L**3,) + (1,) * len(t_adv), dtype=bool)
        for char in t_adv:
            shape = [L**3] + [1] * len(t_adv)
            shape[t_adv.index(char) + 1] = dof
            valid_t &= masks[char].reshape(shape)

        valid_t_shape = [1] * len(t_layout)
        valid_t_shape[t_layout.index("Q")] = L**3
        for char in t_adv:
            valid_t_shape[t_layout.index(char)] = dof
        t_val = jnp.where(valid_t.reshape(valid_t_shape), t_val, 0.0)

        # einsum logic
        valid_out = jnp.ones((L**3,) + (1,) * len(out_adv), dtype=bool)
        for char in out_adv:
            shape = [L**3] + [1] * len(out_adv)
            shape[out_adv.index(char) + 1] = dof
            valid_out &= masks[char].reshape(shape)

        valid_out_shape = [1] * len(out_layout)
        valid_out_shape[out_layout.index("Q")] = L**3
        for char in out_adv:
            valid_out_shape[out_layout.index(char)] = dof

        w_dim_str = v_str if W.ndim == 4 else "Q" + v_str

        accum = 0
        for perm_v_str, sign in v_perms:
            ein_str = (
                f"{t_layout}, {w_dim_str.replace(v_str, perm_v_str)} -> {out_layout}"
            )
            accum += jnp.einsum(ein_str, t_val, W) * (sign * factor)

        accum = jnp.where(valid_out.reshape(valid_out_shape), accum, 0.0)

        # scatter
        out_idx = []
        for char in out_str:
            if char in out_adv:
                shape = [L**3] + [1] * len(out_adv)
                shape[out_adv.index(char) + 1] = dof
                out_idx.append(maps[char].reshape(shape))
            else:
                out_idx.append(slice(None))

        if out_adv:
            updated_out = current_out.at[tuple(out_idx)].add(accum)
        else:
            updated_out = current_out + jnp.sum(accum)

        return updated_out, None

    final_out_tensor, _ = jax.lax.scan(scan_step, out_tensor, (deltas, weights))

    return final_out_tensor


@partial(jax.jit, static_argnames=("L"))
def stamp_t1(
    t1: jax.Array,
    t2: jax.Array,
    f_ph: jax.Array,
    f_pp: jax.Array,
    f_hh: jax.Array,
    deltas,
    weights,
    is_p: jax.Array,
    local_map: jax.Array,
    L: int,
) -> jax.Array:

    P, H = f_ph.shape

    def _einsum(contraction, in_tensor, out_tensor, factor=1.0):
        return stamp_einsum(
            contraction,
            in_tensor,
            out_tensor,
            deltas,
            weights,
            is_p,
            local_map,
            L,
            factor=factor,
        )

    H1 = f_ph
    H1 = _einsum("ck, akci -> ai", t1, H1, factor=-1.0)
    H1 += jnp.einsum("ck, acik -> ai", f_ph, t2)
    H1 = _einsum("cakl, cikl -> ai", t2, H1, factor=-0.5)

    I_dl = _einsum("ck, cdkl -> dl", t1, jnp.zeros((P, H)))
    H1 += jnp.einsum("dl, dali -> ai", I_dl, t2)

    # ppph
    H1 = _einsum("cdki, cdak -> ai", t2, H1, factor=-0.5)

    X_hh = -f_hh
    X_hh -= 0.5 * jnp.einsum("ck, ci -> ki", f_ph, t1)
    X_hh = _einsum("bj, bijk -> ki", t1, X_hh, factor=-1.0)
    X_hh = _einsum("cdli, cdlk -> ki", t2, X_hh, factor=-1.0)

    I_dk = _einsum("cl, cdlk -> dk", t1, jnp.zeros((P, H)))
    X_hh -= 0.5 * jnp.einsum("dk, di -> ki", I_dk, t1)

    X_pp = f_pp
    X_pp -= 0.5 * jnp.einsum("ck, ak -> ac", f_ph, t1)
    X_pp = _einsum("dakl, dckl -> ac", t2, X_pp, factor=-0.5)

    I_cl = _einsum("dk, cdkl -> cl", t1, jnp.zeros((P, H)))
    X_pp += 0.5 * jnp.einsum("cl, al -> ac", I_cl, t1)

    # ppph
    X_pp = _einsum("ck, cdak -> ad", t1, X_pp, factor=-1.0)

    H1 += jnp.einsum("ac, ci -> ai", X_pp, t1)
    H1 += jnp.einsum("ki, ak -> ai", X_hh, t1)

    denom = jnp.diag(X_pp)[:, None] + jnp.diag(X_hh)[None, :]
    denom = jnp.where(denom == 0, 1e-10, denom)

    return t1 - (H1 / denom)
