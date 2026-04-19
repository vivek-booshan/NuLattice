import jax
import jax.numpy as jnp
import numpy as np

from jax.experimental.sparse import BCOO
from jax.sharding import NamedSharding, PartitionSpec as P


class Operator:
    def __init__(self, indices: jnp.ndarray, values: jnp.ndarray, nstat: int):
        self.nstat = nstat
        # JAX arrays are the primary backend here
        self.indices = jnp.asarray(indices, dtype=jnp.int32)
        self.values = jnp.asarray(values)

        if self.indices.ndim == 1:
            self.indices = self.indices[:, jnp.newaxis]

    def __len__(self):
        return len(self.values)

    def to_list(self):
        if len(self) == 0:
            return []

        out_list = []
        for i in range(len(self.values)):
            row = self.indices[i].tolist()
            row.append(self.values[i])
            out_list.append(row)
        return out_list

    def to_bcoo(self, mesh=None):
        """Converts operator to a JAX BCOO sparse array, optionally sharded."""
        rank = self._get_expected_rank()
        shape = (self.nstat,) * rank

        # If a mesh is provided, shard the NNZ dimension
        if mesh:
            sharding = NamedSharding(mesh, P("data"))
            indices = jax.device_put(self.indices, sharding)
            data = jax.device_put(self.values, sharding)
            return BCOO((data, indices), shape=shape)

        return BCOO((self.values, self.indices), shape=shape)

    def to_dense(self, mesh=None):
        rank = self.indices.shape[1]
        shape = (self.nstat,) * rank
        mat = jnp.zeros(shape, dtype=self.values.dtype)
        mat = mat.at[tuple(self.indices[:, i] for i in range(rank))].add(self.values)
        if mesh:
            sharding = NamedSharding(mesh, P("data", *((None,) * (rank - 1))))
            return jax.device_put(mat, sharding)
        return mat

    @classmethod
    def from_list(
        cls,
        operator_list,
        nstat: int,
    ):
        """Operator from a legacy list of lists [[p, q, ..., val], ...]"""
        if not operator_list:
            rank = cls._get_expected_rank()
            return cls(
                jnp.empty((0, rank), dtype=jnp.int32),
                jnp.empty((0,), dtype=jnp.float64),
                nstat,
            )

        data = jnp.array(operator_list, dtype=jnp.float64)
        indices = jnp.round(data[:, :-1]).astype(jnp.int32)
        values = data[:, -1]

        return cls(indices, values, nstat)


class OneBodyOperator(Operator):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 2:
            raise ValueError(f"Expected (N, 2) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 2

class TwoBodyOperator(Operator):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 4

class ThreeBodyOperator(Operator):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 6

class Chef:
    def __init__(self, num_nodes=1, num_gpus=1):
        self.mesh = jax.make_mesh(axis_shapes=(num_nodes, num_gpus), axis_names=("nodes", "gpus"))

    def prepare(self, arr, rank: int = None, spec: NamedSharding = None):
        r = rank if rank is not None else arr.ndim 

        if spec is not None:
            spec = spec
        else:
            if r == 0: 
                spec = P() # alternatively can be used for replication
            elif r == 1:
                spec = P(('nodes', 'gpus')) # 1d array should be split across everything
            else:
                spec = P("nodes", "gpus", *([None] * (r - 2)))

        sharding = NamedSharding(self.mesh, spec)

        # cpu check
        if isinstance(arr, np.ndarray):
            if arr.nbytes > 1e9: # only if > 1 gb
                # calculate bounding box per gpu, slice on cpu, move to gpu
                return jax.make_array_from_callback(
                    arr.shape,
                    sharding,
                    lambda idx: arr[idx])
            else:
                return jax.device_put(arr, sharding)
        return jax.device_put(arr, sharding)
