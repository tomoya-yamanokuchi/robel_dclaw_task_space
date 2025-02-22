import numpy as np
from ..value_object import Manifold2D
from ..value_object import BiasedEndEffectorPosition_2D_Plane
from value_object import EndEffectorPosition
from service import normalize, denormalize


class Manifold2DTaskSpaceInterface:
    def __init__(self):
        self.num_claw = 3

    def _task2end_1claw(self, task_space_position):
        z_task         = task_space_position[:, :, 0] # 順番間違えないように！
        y_task         = task_space_position[:, :, 1] # 順番間違えないように！
        y_end_effector = denormalize(y_task, x_min=BiasedEndEffectorPosition_2D_Plane.y_lb, x_max=BiasedEndEffectorPosition_2D_Plane.y_ub, m=Manifold2D._min, M=Manifold2D._max)
        z_end_effector = denormalize(z_task, x_min=BiasedEndEffectorPosition_2D_Plane.z_lb, x_max=BiasedEndEffectorPosition_2D_Plane.z_ub, m=Manifold2D._min, M=Manifold2D._max)
        x_end_effector = np.zeros(y_end_effector.shape) + BiasedEndEffectorPosition_2D_Plane.x_base
        return np.stack([x_end_effector, y_end_effector, z_end_effector], axis=-1)

    # @abstractmethod
    def task2end(self, task_space_position: Manifold2D):
        end_effector_position = [self._task2end_1claw(x) for x in np.split(task_space_position.value, self.num_claw, axis=-1)]
        return EndEffectorPosition(np.concatenate(end_effector_position, axis=-1))

    # @abstractmethod
    def end2task(self, end_effector_position: EndEffectorPosition):
        task_space_position = [self._end2task_1claw(x) for x in np.split(end_effector_position.value, self.num_claw, axis=-1)]
        return Manifold2D(np.concatenate(task_space_position, axis=-1))

    def _end2task_1claw(self, end_effector_position_1claw):
        y      = end_effector_position_1claw[:, :, 1]
        z      = end_effector_position_1claw[:, :, 2]
        # import ipdb; ipdb.set_trace()
        y_task = normalize(x=y, x_min=BiasedEndEffectorPosition_2D_Plane.y_lb, x_max=BiasedEndEffectorPosition_2D_Plane.y_ub, m=Manifold2D._min, M=Manifold2D._max)
        z_task = normalize(x=z, x_min=BiasedEndEffectorPosition_2D_Plane.z_lb, x_max=BiasedEndEffectorPosition_2D_Plane.z_ub, m=Manifold2D._min, M=Manifold2D._max)
        return np.stack([z_task, y_task], axis=-1) # np.c_[z, y] # 順番間違えないように！

