import numpy as np
from robel_dclaw_task_space.manifold_2d.value_object import Manifold2D, ManifoldDiff2D
from robel_dclaw_task_space.manifold_2d.value_object import ManifoldDiff2DfromSingleFinger
from robel_dclaw_task_space.manifold_2d.value_object import Manifold2DfromSingleFinger

def run_tests():
    # ----
    data1    = np.random.randn(2)
    obj1     = ManifoldDiff2DfromSingleFinger(data1)
    full_obj = obj1.to_all_fingers_object()
    # ----
    print(obj1)
    print(full_obj)

    # ----
    data1    = np.random.randn(2)
    obj1     = Manifold2DfromSingleFinger(data1)
    full_obj = obj1.to_all_fingers_object()
    # ----
    print(obj1)
    print(full_obj)

if __name__ == "__main__":
    run_tests()
