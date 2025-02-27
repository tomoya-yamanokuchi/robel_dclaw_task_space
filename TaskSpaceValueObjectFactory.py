from .manifold_1d.value_object import Manifold1D
from .manifold_2d.value_object import Manifold2D


class TaskSpaceValueObjectFactory:
    @staticmethod
    def create(task_space_name: str):
        if "manifold_1d" == task_space_name : return Manifold1D
        if "manifold_2d" == task_space_name : return Manifold2D
        raise NotImplementedError()
