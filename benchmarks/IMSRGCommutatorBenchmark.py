# Copyright 2025 Matthias Heinz. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
"""Script to test IMSRG(2) fundamental commutators against benchmark data."""
__authors__ = ["Matthias Heinz"]
__credits__ = ["Matthias Heinz"]
__copyright__ = "(c) Matthias Heinz"
__license__ = "BSD-3-Clause"
__date__ = "2025-09-03"

import numpy as np

from NuLattice.cpu.imsrg import commutators
from data.read_file import read_file, read_file_basis_occs


def evaluate_test(test):
    """
    Evaluates a single commutator test against reference benchmark data
    
    Loads input operators and reference results from files, computes the specified
    commutator using the implemented functions, and compares against benchmark values
    to verify correctness within numerical tolerance
    
    :param test:    Test specification tuple containing test parameters
    :type test:     tuple[str, function, tuple[int], str, str, str, float]
    """
    name, func, ranks, in1, in2, out, factor = test
    rank1, rank2, rank3 = ranks
    x1 = read_file(in1 + "_{}B.dat".format(rank1))
    y2 = read_file(in2 + "_{}B.dat".format(rank2))
    x2 = read_file(in1 + "_{}B.dat".format(rank2))
    y1 = read_file(in2 + "_{}B.dat".format(rank1))
    out = out + "_{}B.dat".format(rank3)
    occs = read_file_basis_occs(out)
    zref = read_file(out)

    zref_norm = np.sum(zref ** 2)

    if rank1 == rank2:
        z = factor * func(occs, x1, y2)
    else:
        # If ranks of X and Y are different, we also need to swap the ranks
        # between X and Y and evaluate a second commutator with minus sign
        z = factor * (func(occs, x1, y2) - func(occs, y1, x2))
    z_norm = np.sum(z ** 2)

    diff_norm = np.sum((z - zref) ** 2)

    if diff_norm > 1e-4:
        print(
            "Test {} failed: {} (reference) vs {} (actual)".format(
                name, zref_norm, z_norm
            )
        )
    else:
        print("Test {} passed".format(name))


"""
Test specifications for IMSRG commutator validation

Each test is a tuple containing:
- Test name (str): Identifier for the specific commutator test
- Function (callable): The commutator function to be tested
- Ranks (tuple[int]): Input and output operator ranks (rank1, rank2, rank3)
- Input filenames (str): Base names for input operator files
- Output filename (str): Base name for reference result file  
- Factor (float): Sign factor accounting for commutator antisymmetry

The tests cover all fundamental IMSRG(2) commutators:
- [1B,1B] → 0B, 1B: One-body operator commutators
- [1B,2B] → 1B, 2B: Mixed one-body/two-body commutators  
- [2B,2B] → 0B, 1B, 2B: Two-body operator commutators

Each commutator is tested in both orientations [X,Y] and [Y,X] = -[X,Y]
to verify proper antisymmetry implementation
"""
tests = [
    # Test for [X, Y] -> Z, (1B, 1B) -> 0B
    (
        "COMM_110",
        commutators.evaluate_comm_110,
        (1, 1, 0),
        "X",
        "Y",
        "Z_comm110",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (1B, 1B) -> 0B
    (
        "COMM_110_rev",
        commutators.evaluate_comm_110,
        (1, 1, 0),
        "Y",
        "X",
        "Z_comm110",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (1B, 1B) -> 1B
    (
        "COMM_111",
        commutators.evaluate_comm_111,
        (1, 1, 1),
        "X",
        "Y",
        "Z_comm111",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (1B, 1B) -> 1B
    (
        "COMM_111_rev",
        commutators.evaluate_comm_111,
        (1, 1, 1),
        "Y",
        "X",
        "Z_comm111",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (1B, 2B) -> 1B
    (
        "COMM_121",
        commutators.evaluate_comm_121,
        (1, 2, 1),
        "X",
        "Y",
        "Z_comm121",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (1B, 2B) -> 1B
    (
        "COMM_121_rev",
        commutators.evaluate_comm_121,
        (1, 2, 1),
        "Y",
        "X",
        "Z_comm121",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (1B, 2B) -> 2B
    (
        "COMM_122",
        commutators.evaluate_comm_122,
        (1, 2, 2),
        "X",
        "Y",
        "Z_comm122",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (1B, 2B) -> 2B
    (
        "COMM_122_rev",
        commutators.evaluate_comm_122,
        (1, 2, 2),
        "Y",
        "X",
        "Z_comm122",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (2B, 2B) -> 0B
    (
        "COMM_220",
        commutators.evaluate_comm_220,
        (2, 2, 0),
        "X",
        "Y",
        "Z_comm220",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (2B, 2B) -> 0B
    (
        "COMM_220_rev",
        commutators.evaluate_comm_220,
        (2, 2, 0),
        "Y",
        "X",
        "Z_comm220",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (2B, 2B) -> 1B
    (
        "COMM_221",
        commutators.evaluate_comm_221,
        (2, 2, 1),
        "X",
        "Y",
        "Z_comm221",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (2B, 2B) -> 1B
    (
        "COMM_221_rev",
        commutators.evaluate_comm_221,
        (2, 2, 1),
        "Y",
        "X",
        "Z_comm221",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (2B, 2B) -> 2B (particle-particle/hole-hole)
    (
        "COMM_222pphh",
        commutators.evaluate_comm_222_pphh,
        (2, 2, 2),
        "X",
        "Y",
        "Z_comm222_pp_hh",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (2B, 2B) -> 2B (particle-particle/hole-hole)
    (
        "COMM_222pphh_rev",
        commutators.evaluate_comm_222_pphh,
        (2, 2, 2),
        "Y",
        "X",
        "Z_comm222_pp_hh",
        -1.0,
    ),
    # Test for [X, Y] -> Z, (2B, 2B) -> 2B (particle-hole)
    (
        "COMM_222ph",
        commutators.evaluate_comm_222_ph,
        (2, 2, 2),
        "X",
        "Y",
        "Z_comm222_ph",
        1.0,
    ),
    # Reversed test for [Y, X] -> Z, (2B, 2B) -> 2B (particle-hole)
    (
        "COMM_222ph_rev",
        commutators.evaluate_comm_222_ph,
        (2, 2, 2),
        "Y",
        "X",
        "Z_comm222_ph",
        -1.0,
    ),
]

for test in tests:
    evaluate_test(test)
