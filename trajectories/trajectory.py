from .waypoint import Waypoint

class Trajectory:

    def __init__(self) -> None:
        pass

    def get_waypoint(self, time: float) -> Waypoint:
        """
        Needs to be implemented by subclass. time \in [0,1]
        """
        raise NotImplementedError()


    