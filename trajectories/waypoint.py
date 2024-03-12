import numpy as np
from enum import Enum, auto


class Waypoint:
    """
    3D coordinate with a timestamp.
    """

    coordinate: np.ndarray
    """R^3 coordinate"""
    timestamp: float
    """Time as float"""

    def __init__(self, coordinate: np.ndarray, timestamp: float) -> None:
        self.coordinate = coordinate
        self.timestamp = timestamp
