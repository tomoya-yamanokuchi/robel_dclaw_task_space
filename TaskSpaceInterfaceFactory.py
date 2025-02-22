from .manifold_1d.interface import Manifold1DTaskSpaceInterface
from .manifold_2d.interface import Manifold2DTaskSpaceInterface


class TaskSpaceInterfaceFactory:
    @staticmethod
    def create(task_space_name: str):
        if "manifold_1d" == task_space_name : return Manifold1DTaskSpaceInterface()
        if "manifold_2d" == task_space_name : return Manifold2DTaskSpaceInterface()
        raise NotImplementedError()
