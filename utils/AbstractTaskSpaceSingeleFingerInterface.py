from abc import ABC, abstractmethod
from ..utils import AbstractTaskSpaceObject

class AbstractTaskSpaceSingeleFingerInterface(ABC):
    @abstractmethod
    def to_all_fingers_object(self) -> AbstractTaskSpaceObject:
        pass

