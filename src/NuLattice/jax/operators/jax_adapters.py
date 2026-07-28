import numpy as np

from NuLattice.utils._jax_types import (
    OneBodyOperator,
    ThreeBodyOperator,
    TwoBodyOperator,
)


def one_body_from_list(elements, nstat):
    if len(elements) == 0:
        return OneBodyOperator(
            np.empty((0, 2), dtype=np.int32),
            np.empty(0, dtype=np.float64),
            nstat,
        )

    indices = np.asarray([row[:2] for row in elements], dtype=np.int32)
    values = np.asarray([row[2] for row in elements])
    return OneBodyOperator(indices, values, nstat)


def two_body_from_sparse(matrix, nstat):
    matrix = matrix.tocoo()

    p = matrix.row % nstat
    q = matrix.row // nstat
    r = matrix.col % nstat
    s = matrix.col // nstat
    indices = np.column_stack((p, q, r, s)).astype(np.int32, copy=False)

    return TwoBodyOperator(indices, matrix.data, nstat)


def three_body_from_sparse(matrix, nstat):
    matrix = matrix.tocoo()
    nstat2 = nstat * nstat

    p = matrix.row % nstat
    q = (matrix.row // nstat) % nstat
    r = matrix.row // nstat2
    s = matrix.col % nstat
    t = (matrix.col // nstat) % nstat
    u = matrix.col // nstat2
    indices = np.column_stack((p, q, r, s, t, u)).astype(np.int32, copy=False)

    return ThreeBodyOperator(indices, matrix.data, nstat)


def empty_three_body(nstat, dtype=np.float64):
    return ThreeBodyOperator(
        np.empty((0, 6), dtype=np.int32),
        np.empty(0, dtype=dtype),
        nstat,
    )
