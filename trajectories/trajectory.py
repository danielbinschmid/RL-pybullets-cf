from .waypoint import Waypoint


class Trajectory:

    def __init__(self) -> None:
        pass

    def get_waypoint(self, time: float) -> Waypoint:
        """
        Needs to be implemented by subclass. time \in [0,1]
        """
        raise NotImplementedError()

    def get_next_waypoint(self, time: float) -> Waypoint:
        """
        Needs to be implemented by subclass.
        """
        raise NotImplementedError()
    
    def is_done():
        """
        Needs to be implemented by subclass.
        """
        raise NotImplementedError()


    def reset(self):
        """
        Needs to be implemented by subclass.
        """
        raise NotImplementedError()
