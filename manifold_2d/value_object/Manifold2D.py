from dataclasses import dataclass
import numpy as np
from typing import Tuple
from .ManifoldDiff2D import ManifoldDiff2D
from ...utils import AbstractTaskSpaceObject


@dataclass(frozen=True)
class Manifold2D(AbstractTaskSpaceObject):
    value           : np.ndarray
    _min_value     : float = 0.0
    _max_value     : float = 1.0
    _expected_shape: Tuple[int, ...] = (6,)

    def __post_init__(self):
        object.__setattr__(self, 'value', np.clip(self.value, self._min_value, self._max_value))
        self.validate()

    def __eq__(self, other):
        if not isinstance(other, Manifold2D):
            return NotImplemented
        return np.array_equal(self.value, other.value)

    def __add__(self, other):
        if not isinstance(other, ManifoldDiff2D):
            return NotImplemented
        return Manifold2D(self.value + other.value)

    def __repr__(self):
        return (f"Manifold2D(value={self.value}, min_value={self._min_value}, "
                f"max_value={self._max_value}, expected_shape={self._expected_shape})")

if __name__ == '__main__':
    value1 = np.random.randn(6)
    obj1   = Manifold2D(value1)
    # ----
    print(obj1)
