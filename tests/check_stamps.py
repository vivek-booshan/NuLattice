import numpy as np
import NuLattice.jax.lattice as lat
import NuLattice.jax.ccm.stamps as stamps 

# 1. Generate Legacy
L = 4
vT1 = -8.0
vS1 = -2.0
cE = 5.5
v3NF = cE

lattice_sites = lat.get_lattice(L)
legacy_tkin = lat.Tkin(lattice_sites, L)
legacy_sort_idx = np.lexsort((legacy_tkin.indices[:, 1], legacy_tkin.indices[:, 0]))
sorted_legacy_indices = legacy_tkin.indices[legacy_sort_idx]
sorted_legacy_values = legacy_tkin.values[legacy_sort_idx]

legacy_cont = lat.contacts(vT1, vS1, lattice_sites, L)
cont_legacy_sort_idx = np.lexsort((legacy_cont.indices[:, 1], legacy_cont.indices[:, 0]))
cont_sorted_legacy_indices = legacy_cont.indices[legacy_sort_idx]
cont_sorted_legacy_values = legacy_cont.values[legacy_sort_idx]

legacy_nnn = lat.NNNcontact(v3NF, lattice_sites, L)
nnn_legacy_sort_idx = np.lexsort((legacy_nnn.indices[:, 1], legacy_nnn.indices[:, 0]))
nnn_sorted_legacy_indices = legacy_nnn.indices[legacy_sort_idx]
nnn_sorted_legacy_values = legacy_nnn.values[legacy_sort_idx]

# 2. Generate Stamps -> Convert
d_tkin, w_tkin = stamps.stamp_one_body(2, 2)
stamp_tkin = stamps.stamp_to_one_body(d_tkin, w_tkin, L)

d_cont, w_cont = stamps.stamp_two_body(vT1, vS1, 2, 2)
print(w_cont.shape)
stamp_cont = stamps.stamp_to_two_body(d_cont, w_cont, L)

d_nnn, w_nnn = stamps.stamp_three_body(v3NF, 2, 2)
print(w_nnn.shape)
stamp_nnn = stamps.stamp_to_three_body(d_nnn, w_nnn, L)

# 3. Assert exact match
assert np.allclose(sorted_legacy_indices, stamp_tkin.indices)
assert np.allclose(sorted_legacy_values, stamp_tkin.values)
assert np.allclose(legacy_cont.indices, stamp_cont.indices)
assert np.allclose(legacy_cont.values, stamp_cont.values)
assert np.allclose(legacy_nnn.indices, stamp_nnn.indices)
assert np.allclose(legacy_nnn.values, stamp_nnn.values)

print("yay")

