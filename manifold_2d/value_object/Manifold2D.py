from dataclasses import dataclass
import numpy as np
from typing import Tuple
from .ManifoldDiff2D import ManifoldDiff2D


@dataclass(frozen=True)
class Manifold2D:
    data           : np.ndarray
    _min_value     : float = 0.0
    _max_value     : float = 1.0
    _expected_shape: Tuple[int, ...] = (6,)

    def __post_init__(self):
        object.__setattr__(self, 'data', np.clip(self.data, self._min_value, self._max_value))
        self._validate()

    def _validate(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy ndarray")

        if self.data.shape != self._expected_shape:
            raise ValueError(f"Expected shape {self._expected_shape}, but got {self.data.shape}")

    def __eq__(self, other):
        if not isinstance(other, Manifold2D):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def __add__(self, other):
        if not isinstance(other, ManifoldDiff2D):
            return NotImplemented
        return Manifold2D(self.data + other.data)

    def __repr__(self):
        return (f"Manifold2D(data={self.data}, min_value={self._min_value}, "
                f"max_value={self._max_value}, expected_shape={self._expected_shape})")


if __name__ == '__main__':
    data1 = np.random.randn(6)
    obj1  = Manifold2D(data1)
    # ----
    data2 = np.random.randn(6) * 0.1
    obj2  = ManifoldDiff2D(data2)
    # ----
    obj3  = (obj1 + obj2)
    # ----
    print(obj3)
