import numpy as np
from robel_dclaw_task_space.manifold_2d.value_object import Manifold2D, ManifoldDiff2D


def run_tests():
    # ----
    data1 = np.random.randn(6)
    obj1  = Manifold2D(data1)
    # ----
    data2 = np.random.randn(6) * 0.1
    obj2  = ManifoldDiff2D(data2)
    # ----
    obj3  = (obj1 + obj2)
    # ----
    print(obj1)
    print(obj2)
    print(obj3)

if __name__ == "__main__":
    run_tests()
