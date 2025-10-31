import numpy as np

class BBox2D:
    def __init__(self, y0, x0, y1, x1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        
    def __str__(self):
        return f"BBox2D(y0={round(self.y0, 3)}, x0={round(self.x0, 3)}, y1={round(self.y1, 3)}, x1={round(self.x1, 3)})"


class Object2D:
    def __init__(self, score: float, bbox: BBox2D, label: str):
        self.score = score
        self.label = label
        self.bbox = bbox
        
    def __str__(self):
        return f"Object2D(score={round(self.score, 3)}, label={self.label} bbox={self.bbox}"


class ObjectDetectorInterface:
    def detect_objects(self, image: np.ndarray) -> list[Object2D]: ...