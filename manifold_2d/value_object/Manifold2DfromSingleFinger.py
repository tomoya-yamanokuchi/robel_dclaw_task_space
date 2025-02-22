from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional
from .Manifold2D import Manifold2D


@dataclass(frozen=True)
class Manifold2DfromSingleFinger:
    value        : np.ndarray
    default_value: float = 0.5  # 指定されていない指のデフォルト値
    full_shape   : Tuple[int, ...] = (6,)  # ValueObject や ValueObjectB に合わせる

    def __post_init__(self):
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy ndarray")
        if self.value.shape[0] > self.full_shape[0]:
            raise ValueError(f"value exceeds expected shape {self.full_shape}")

    def to_all_fingers_object(self) -> Manifold2D:
        full_data = np.full(self.full_shape, self.default_value)
        full_data[:self.value.shape[0]] = self.value
        return Manifold2D(full_data)
