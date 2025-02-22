from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass(frozen=True)
class ManifoldDiff2D:
    data           : np.ndarray
    _min_value     : float = -0.15
    _max_value     : float =  0.15
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
        if not isinstance(other, ManifoldDiff2D):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def __repr__(self):
        return (f"ManifoldDiff2D(data={self.data}, min_value={self._min_value}, "
                f"max_value={self._max_value}, expected_shape={self._expected_shape})")


if __name__ == '__main__':
    data = np.random.randn(6)
    obj  = ManifoldDiff2D(data)
    print(data)
    print(obj.data)
    # print(obj)

