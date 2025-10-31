import numpy as np


class BBox2D:
    def __init__(self, y0, x0, y1, x1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class Object2D:
    def __init__(self, score: float, bbox: BBox2D, label: str):
        self.score = score
        self.label = label
        self.bbox = bbox


class ObjectDetectorInterface:
    def detect_objects(self, image: np.ndarray) -> list[Object2D]: ...