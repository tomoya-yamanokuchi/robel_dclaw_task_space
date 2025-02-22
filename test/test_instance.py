import numpy as np
from robel_dclaw_kinematics import ForwardKinematics
from robel_dclaw_task_space.manifold_1d.interface import Manifold1DTaskSpaceInterface



def run_tests():
    # ----
    forward_kinematics = ForwardKinematics()
    task_space = Manifold1DTaskSpaceInterface()
    # ----
    num_joint             = 9
    robot_position        = np.zeros(num_joint)
    end_effector_position = forward_kinematics.calc(robot_position).squeeze()
    task_space_positioin  = task_space.end2task(end_effector_position).squeeze()

    print("---------------------------------------------------------------")
    print("        robot_position = ", robot_position)
    print(" end_effector_position = ", end_effector_position)
    print("  task_space_positioin = ", task_space_positioin)
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    run_tests()
