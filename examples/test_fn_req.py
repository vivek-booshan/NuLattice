import jax
import jax.numpy as jnp
from functools import partial

# --- Mock Dimensions for 100GB+ scale ---
# (P=4000, H=100) -> ~12.8 GB per tensor. 
# Scale up as needed to match your production constraints.
P_dim = 6896
H_dim = 16

pphh_shape = jax.ShapeDtypeStruct((P_dim, P_dim, H_dim, H_dim), jnp.float64)
val_shape = jax.ShapeDtypeStruct((P_dim, P_dim, H_dim, H_dim), jnp.float64)

@jax.jit
def add_AB(target, val):
    return target.at[:].add(val - val.transpose(1, 0, 2, 3))

@jax.jit
def add_IJ(target, val):
    return target.at[:].add(val - val.transpose(0, 1, 3, 2))

# @partial(jax.jit, donate_argnums=(0,))
@jax.jit
def dckl_dakl__ac(a, b):
    return 0.5 * jnp.einsum("dckl, dakl -> ac", a, b)

@jax.jit
def dckl_dakl__ca_ac(a, b):
    ca = 0.5 * jnp.einsum("dckl, dakl -> ca", a, b)
    return ca.transpose()
# @partial(jax.jit, donate_argnums=(0,))
@jax.jit
def tiled_dckl_dakl(v_pphh, t2, chunks=4):
    """
    Decomposes the contraction over the 'd' axis into chunks 
    to prevent XLA from materializing the entire transposed temp tensor.
    """
    n_d = v_pphh.shape[0]
    size = n_d // chunks
    X_ac = jnp.zeros((v_pphh.shape[1], t2.shape[1]))

    for i in range(chunks):
        start = i * size
        end = (i + 1) * size if i != chunks - 1 else n_d
        
        # Slice only a portion of the contracted index
        v_slice = v_pphh[start:end]
        t_slice = t2[start:end]
        
        # Contract the slice
        X_ac += jnp.tensordot(v_slice, t_slice, axes=((0, 2, 3), (0, 2, 3)))
        
    return X_ac

def run_analysis(name, func):
    print(f"\n{'='*20} Analysis: {name} {'='*20}")
    
    # 1. Lower to HLO
    lowered = func.lower(pphh_shape, pphh_shape)
    
    # 2. Compile to executable
    compiled = lowered.compile()
    
    # 3. Print Memory Analysis
    # This shows peak memory and tells you which buffers are 'Live'
    print(compiled.cost_analysis())
    print(compiled.memory_analysis())
    
    # 4. Check for Buffer Donation potential
    # If the computation is done 'in-place', the output should alias an input
    # (Only works if you add donate_argnums to the jit decorator)

if __name__ == "__main__":
    # run_analysis("add_AB (Standard)", add_AB)
    # run_analysis("add_IJ (Standard)", add_IJ)
    run_analysis("dckl_dakl__ac", dckl_dakl__ac)
    run_analysis("dckl_dakl__ca_ac", dckl_dakl__ca_ac)
    # run_analysis("dckl_dakl_ac2", tiled_dckl_dakl)
