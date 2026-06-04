import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.shard_map import shard_map
import numpy as np

# 1. Simulate 4 CPU devices for local testing

def shard_transpose(tensor: jax.Array, axes, out_spec, mesh=None, in_spec=None):
    if mesh is None:
        mesh = tensor.sharding.mesh
    if in_spec is None:
        in_spec = tensor.sharding.spec

    # Ensure out_spec is a PartitionSpec object
    if not isinstance(out_spec, P):
        out_spec = P(*out_spec)

    def local_transpose(local_chunk):
        return jnp.transpose(local_chunk, axes)

    shlt = shard_map(
        local_transpose,
        mesh=mesh,
        in_specs=in_spec,
        out_specs=out_spec
    )

    return shlt(tensor)

# 2. Setup Mesh (2x2)
devices = np.array(jax.devices()).reshape(2, 2)
mesh = Mesh(devices, axis_names=("nodes", "gpus"))

# 3. Create a global 4x4 array with unique values to track movement
# Matrix looks like:
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11],
#  [12, 13, 14, 15]]
data = jnp.arange(10000).reshape(100, 100)
sharding = NamedSharding(mesh, P("nodes", "gpus"))
x = jax.device_put(data, sharding)

print("Original Global Matrix:\n", x)

# 4. Perform Zero-Comm Transpose
# Global goal: Transpose (0, 1)
# Physical goal: Each device transposes its local 2x2 chunk. 
# Partition (0, 1) on Device 1 remains on Device 1 but is relabeled as (1, 0).
x_transposed = shard_transpose(
    x, 
    axes=(1, 0), 
    out_spec=P("gpus", "nodes"), # Swap the hardware axes
    mesh=mesh,
    in_spec=P("nodes", "gpus")
)

print("\nTransposed Global Matrix:\n", x_transposed)

# 5. Verification
expected = jnp.transpose(data, (1, 0))
assert jnp.all(x_transposed == expected)
print("\n✅ Verification Successful: Global transpose matches local-relabeled transpose.")

# 6. Performance & Compilation Analysis
# We use a dummy function to trigger AOT compilation
def run_transpose(arr):
    return shard_transpose(data, axes=(1, 0), out_spec=P("gpus", "nodes"), mesh=mesh, in_spec=P("nodes", "gpus"))
    # return jnp.transpose(data)

# Lower to HLO
lowered = jax.jit(run_transpose).lower(x)
compiled = lowered.compile()

print("\n--- COST ANALYSIS ---")
# This shows the flops and bytes moved
cost = compiled.cost_analysis()
for k, v in cost.items():
    print(f"{k}: {v}")

print("\n--- MEMORY ANALYSIS ---")
# This shows the peak memory and buffer assignments
mem_stats = compiled.memory_analysis()
print(mem_stats)

print("\n--- HLO VERIFICATION ---")
hlo = compiled.as_text()
# We check for the 'collective' keyword which indicates network communication
if "collective" in hlo.lower() or "all-to-all" in hlo.lower():
    print("❌ Communication detected in HLO.")
else:
    print("✅ Zero communication confirmed: No collective ops found in HLO.")

# Optional: Print HLO to see the 'transpose' happening inside the shard-loop
# print(hlo)
