from dataclasses import dataclass
import numpy as np
from typing import Tuple, Optional, List, Any
from .Manifold2D import Manifold2D
from ...utils import AbstractTaskSpaceSingeleFingerInterface


@dataclass(frozen=True)
class Manifold2DSingleFinger(AbstractTaskSpaceSingeleFingerInterface):
    value            : np.ndarray
    default_value    : Any = (0.5, 0.0)  # 指定されていない指のデフォルト値
    full_shape       : Tuple[int, ...] = (6,)  # ValueObject や ValueObjectB に合わせる
    dim_single_finger: int = 2

    def __post_init__(self):
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy ndarray")
        if self.value.shape[0] > self.full_shape[0]:
            raise ValueError(f"value exceeds expected shape {self.full_shape}")

    def to_all_fingers_object(self) -> Manifold2D:
        full_data = np.zeros(self.full_shape)
        full_data[(self.dim_single_finger*0):(self.dim_single_finger*1)] = self.value
        full_data[(self.dim_single_finger*1):(self.dim_single_finger*2)] = self.default_value
        full_data[(self.dim_single_finger*2):(self.dim_single_finger*3)] = self.default_value
        # ----
        return Manifold2D(full_data)
