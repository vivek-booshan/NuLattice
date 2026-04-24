import numpy as np

import NuLattice.utils.op_func.one_body as obops
import NuLattice.utils.op_func.two_body as tbops
import NuLattice.lattice as lat
from NuLattice.utils._types import TwoBodyOperator

if __name__ == '__main__':
    thisL = 4
    a = 1.0 / 100.0
    lattice = lat.get_lattice(thisL)

    myTkin=obops.Tkin(lattice, thisL, 3, a)
    print("number of matrix elements from kinetic energy", len(myTkin))

    bpi = 0.7
    sNL = 0.08
    sL = 0.08
    c0 = -0.185

    v_OPE = tbops.onePionEx(thisL, bpi, a)
    v_0 = tbops.shortRangeV_2body(thisL, sL, sNL, c0, a)

    combined_indices = np.vstack([v_0.indices, v_OPE.indices])
    combined_values = np.concatenate([v_0.values, v_OPE.values])

    mycontact = TwoBodyOperator(combined_indices, combined_values, 4*thisL**3)
    print("number of matrix elements from two-body contacts", len(mycontact))
