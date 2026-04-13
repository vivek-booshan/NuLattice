import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec as P, Mesh


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
    # def __new__(cls):
    #     if jax.device_count() == 1:
    #         return None
    #     return super().__new__(cls)
        
    def __init__(self):
        # self.devices = mesh_utils.create_device_mesh((len(jax.devices()),))
        self.mesh = Mesh(jax.devices(), axis_names=("data",))

    # def prepare_operator(self, op):
    #     return op.to_bcoo(mesh=self.mesh)

    # def prepare_op_dense(self, op):
    #     return op.to_dense(mesh=self.mesh)

    def prepare(self, arr: jnp.array, rank: int = None):
        r = rank if rank is not None else arr.ndim 
        if r == 0: 
            spec = P() # alternatively can be used for replication
        elif r == 1:
            spec = P('data')
        else:
            spec = P("data", *([None] * (r - 1)))

        return jax.device_put(arr, NamedSharding(self.mesh, spec))
