import numpy as np


class BBox2D:
    def __init__(self, y0, x0, y1, x1):
        self._x0 = x0
        self._y0 = y0
        self._x1 = x1
        self._y1 = y1


class Object2D:
    def __init__(self, score: float, label: str, bbox: BBox2D):
        self._score = score
        self._label = label
        self._bbox = bbox


class ObjectDetectorInterface:
    def detect_objects(self, image: np.ndarray) -> list[Object2D]: ...