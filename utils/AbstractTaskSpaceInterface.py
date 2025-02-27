from abc import ABC, abstractmethod
from value_object import EndEffectorPosition
from ..utils import AbstractTaskSpaceObject

class AbstractTaskSpaceInterface(ABC):
    @abstractmethod
    def task2end(self, task_space_position_obj: AbstractTaskSpaceObject) -> EndEffectorPosition:
        pass

    @abstractmethod
    def end2task(self, end_effector_position_obj: EndEffectorPosition) -> AbstractTaskSpaceObject:
        pass

