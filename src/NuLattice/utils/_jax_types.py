import jax.numpy as jnp

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

    def to_dense(self):
        raise NotImplementedError

    def to_list(self):
        if len(self) == 0:
            return []

        out_list = []
        for i in range(len(self.values)):
            row = self.indices[i].tolist()
            row.append(self.values[i])
            out_list.append(row)
        return out_list

    @classmethod
    def from_list(
        cls, operator_list, nstat: int,
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

    def to_dense(self):
        mat = jnp.zeros((self.nstat, self.nstat), dtype=self.values.dtype)
        return mat.at[self.indices[:, 0], self.indices[:, 1]].add(self.values)

class TwoBodyOperator(Operator):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 4:
            raise ValueError(f"Expected (N, 4) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 4

    def to_dense(self):
        shape = (self.nstat,) * 4
        mat = jnp.zeros(shape, dtype=self.values.dtype)
        return mat.at[self.indices[:, 0], self.indices[:, 1], 
                      self.indices[:, 2], self.indices[:, 3]].add(self.values)

class ThreeBodyOperator(Operator):
    def __init__(self, indices, values, nstat):
        super().__init__(indices, values, nstat)
        if len(self) > 0 and self.indices.shape[1] != 6:
            raise ValueError(f"Expected (N, 6) indices, got {self.indices.shape}")

    @classmethod
    def _get_expected_rank(cls):
        return 6

    def to_dense(self):
        shape = (self.nstat,) * 6
        mat = jnp.zeros(shape, dtype=self.values.dtype)
        return mat.at[tuple(self.indices[:, i] for i in range(6))].add(self.values)

