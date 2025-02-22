from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass(frozen=True)
class ManifoldDiff2D:
    value          : np.ndarray
    _min_value     : float = -0.15
    _max_value     : float =  0.15
    _expected_shape: Tuple[int, ...] = (6,)

    def __post_init__(self):
        object.__setattr__(self, 'value', np.clip(self.value, self._min_value, self._max_value))
        self._validate()

    def _validate(self):
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy ndarray")

        if self.value.shape != self._expected_shape:
            raise ValueError(f"Expected shape {self._expected_shape}, but got {self.value.shape}")

    def __eq__(self, other):
        if not isinstance(other, ManifoldDiff2D):
            return NotImplemented
        return np.array_equal(self.value, other.value)

    def __repr__(self):
        return (f"ManifoldDiff2D(value={self.value}, min_value={self._min_value}, "
                f"max_value={self._max_value}, expected_shape={self._expected_shape})")


if __name__ == '__main__':
    value = np.random.randn(6)
    obj   = ManifoldDiff2D(value)
    print(value)
    print(obj.value)
    # print(obj)

