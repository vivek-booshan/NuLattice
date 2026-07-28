"""
Provides useful constants to be used in the rest of the code
"""

HBARC = 197.3269804  # MeV fm
MASS_P = 938.27208943 # MeV
MASS_N = 939.56542194 # MeV
# MASS  = 0.5*(MASS_P+MASS_N)
MASS = 938.92
G_A = 1.287
M_PI_0 = 134.98 # MeV
M_PI_C = 139.57 #MeV
F_PI = 92.2 #MeV



def generate_alpha(coords):
    reference = []
    for x, y, z in coords:
        for sz in [0, 1]:
            for tz in [0, 1]:
                reference.append([x, y, z, tz, sz])
    return reference

def generate_deuteron(coords):
    reference = []
    for coord in coords:
        reference.append([*coord, 0, 0])
        reference.append([*coord, 1, 0])
    return reference

def generate_proton(coords):
    reference = []
    for coord in coords:
        reference.append([*coord, 0, 0])
        reference.append([*coord, 0, 1])
    return reference

def generate_neutron(coords):
    reference = []
    for coord in coords:
        reference.append([*coord, 1, 0])
        reference.append([*coord, 1, 1])
    return reference

def generate_triton(coords):
    reference = []
    for coord in coords:
        reference.append([*coord, 0, 0])
        reference.append([*coord, 1, 0])
        reference.append([*coord, 1, 1])
    return reference

def generate_helion(coords):
    reference = []
    for coord in coords:
        reference.append([*coord, 0, 0])
        reference.append([*coord, 1, 0])
        reference.append([*coord, 0, 1])
    return reference

class ReferenceState:
    # [x, y, z, isospin, spin]

    H2_GS = generate_deuteron([(0, 0, 0)])
    H3_GS = generate_triton([(0, 0, 0)])
    HE3_GS = generate_helion([(0, 0, 0)])
    HE4_GS = generate_alpha([(0, 0, 0)])

    LI6_GS = generate_alpha([(0, 0, 0)]) + generate_deuteron([(1, 0, 0)])
    LI6_3HE3H = generate_triton([(0, 0, 0)]) + generate_helion([(1, 0, 0)])

    BE8_GS  = generate_alpha([(0, 0, 0), (1, 0, 0)])
    C12_GS  = generate_alpha([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    
    C12_HOYLE  = generate_alpha([(0, 0, 0), (1, 0, 0), (2, 1, 0)])
    C12_LINEAR = generate_alpha([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    C12_LOOSE  = generate_alpha([(0, 0, 0), (2, 1, 2), (1, 2, 1)])

    O16_GS = generate_alpha([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    O16_EX = generate_alpha([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    O28_GS = (
        generate_alpha([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]) + 
        generate_neutron([(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (2, 0, 0), (0, 2, 0)])
    )

    NE20_GS = generate_alpha([(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 1), (0, 1, 1)])
    MG24_GS = generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1)])
    SI28_GS = generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1)])
    S32_GS  = generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (0, 1, 0)])
    AR36_GS = generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (0, 1, 0), (2, 1, 0)])
    CA40_GS = generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (2, 0, 1), (2, 1, 2), (2, 0, 2)])
    CA48_GS = (
        generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (2, 0, 1), (2, 1, 2), (2, 0, 2),]) +
        generate_neutron([(0, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0),])
    
    )

    NI48_GS = (
        generate_alpha([(1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1), (2, 0, 1), (2, 1, 2), (2, 0, 2)]) +
        generate_proton([(0, 1, 0), (2, 1, 0), (1, 0, 0), (1, 2, 0)])
    )
    NI78_GS = (
        generate_alpha([
            (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 1),
            (2, 0, 1), (2, 1, 2), (2, 0, 2), (0, 2, 1), (1, 2, 2), (0, 1, 2), (2, 2, 1)
        ]) +
        generate_neutron([
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0), (0, 2, 0), 
            (0, 0, 2), (2, 2, 0), (2, 0, 2), (0, 2, 2), (1, 1, 3)
        ])
    )

    @staticmethod
    def holes(ref, basis):
        """
        given a reference state, and a lattice basis, this function returns the corresponding holes
        as a tuple

        :param ref:   reference state as list of states [lx, ly, lz, tz, sz] where the first three integers
                      lx, ly, lzx are the lattice site, and the last two integers are the
                      isospin and spin (with values 0, 1 for -1/2, 1/2)
        :type ref:    list[list[int,int,int,int,int]]
        :param basis: list of basis states in the lattice
        :type basis:  list[list[int,int,int,int,int]]
        :return:      tuple of A integers that are the indices of the hole states
        :rtype:       tuple(int,int,...)
        """
        holes = []
        for state in ref:
            i = basis.index(state)
            holes.append(i)
        return tuple(holes)
