import numpy as np
from robel_dclaw_task_space.manifold_2d.interface import Manifold2DTaskSpaceInterface
from robel_dclaw_task_space.manifold_2d.value_object import Manifold2D


def run_tests():
    # ----
    task_space                = Manifold2DTaskSpaceInterface()
    task_space_position_obj   = Manifold2D(value=np.random.randn(6))
    end_effector_position_obj = task_space.task2end(task_space_position_obj)

    task_space_position_obj2 = task_space.end2task(end_effector_position_obj)

    print("---------------------------------------------------------------")
    print(task_space_position_obj)
    # print(end_effector_position_obj)
    print(task_space_position_obj2)
    print("----------------------------------------------------------------")

if __name__ == "__main__":
    run_tests()
