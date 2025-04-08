import numpy as np
from ...utils import AbstractTaskSpaceObject


class Manifold1D(AbstractTaskSpaceObject):
    _min = 0.0
    _max = 1.0

    def __init__(self, value: np.ndarray):
        self.value : np.ndarray = value % self._max
        self.__validation__()

    def __validation__ (self):
        assert len(self.value.shape) == 3
        assert  self.value.shape[-1] == 3 #(1dim * 3claw)
        self.value = self.value.clip(self._min, self._max)

    def __eq__(self, other: 'Manifold1D') -> bool:
        return True if (other.value == self.value).all() else False

    def __add__(self, other: 'Manifold1D'):
        return Manifold1D(self.value + other.value)

    def __repr__(self):
        return (f"Manifold1D(value={self.value}, min_value={self._min_value}, "
                f"max_value={self._max_value}, expected_shape={self._expected_shape})")

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def numpy_value(self):
        return self.value



if __name__ == '__main__':
    import numpy as np

    data1 = np.random.rand(1,1,3)*1
    data2 = np.random.rand(1,1,3)*2

    x = Manifold1D(data1)
    y = Manifold1D(data2)

    print(x + y)

    print("x = ", x.value)
    print("y = ", y.value)
    print(x == y)
    print("x_min, x_max = ", x.min, x.max)
    print("y_min, y_max = ", y.min, y.max)
    z = x + y
    print("z = ", z.value)

    print(Manifold1D._min)
    print(Manifold1D.min)
