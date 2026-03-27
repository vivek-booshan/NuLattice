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

