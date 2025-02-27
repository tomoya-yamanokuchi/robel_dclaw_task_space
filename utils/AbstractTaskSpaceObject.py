from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class AbstractTaskSpaceObject(ABC):
    def __init__(self, value: np.ndarray, min_value: float, max_value: float, expected_shape: Tuple[int, ...]):
        self.value           = np.clip(value, min_value, max_value)  # 値の制約を適用
        self._min_value      = min_value
        self._max_value      = max_value
        self._expected_shape = expected_shape
        self.validate()

    def validate(self):
        if not isinstance(self.value, np.ndarray):
            raise TypeError("value must be a numpy ndarray")

        if self.value.shape != self._expected_shape:
            raise ValueError(f"Expected shape {self._expected_shape}, but got {self.value.shape}")

    @abstractmethod
    def __repr__(self):
        pass
