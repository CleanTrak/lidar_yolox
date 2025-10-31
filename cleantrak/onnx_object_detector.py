import numpy as np
import onnxruntime as ort
import json

from cleantrak.object_detection_interface import BBox2D, Object2D, ObjectDetectorInterface
# from yolox.data.data_augment import preproc
from cleantrak.yolox_utils import yolox_preprocess, yolox_postprocess


class OnnxObjectDetector(ObjectDetectorInterface):
    def __init__(self, onnx_file: str):
        self._onnx_file = onnx_file
        self._session = ort.InferenceSession(onnx_file)
        self._input_shape_nhwc = self._session._inputs_meta[0].shape
        labels_str = self._session._model_meta.custom_metadata_map["labels"]
        self._labels = json.loads(labels_str)

    def detect_objects(self, image: np.ndarray) -> list[Object2D]:
        image, scale = yolox_preprocess(image, self._input_shape_nhwc[2:])
        ort_inputs = {self._session.get_inputs()[0].name: image[np.newaxis, :, :, :]}
        output = self._session.run(None, ort_inputs)
        scores, boxes, cls_inds = yolox_postprocess(output, self._input_shape_nhwc[2:], scale, score_thr=0.01)
        objects = to_list_of_objects2d(scores, boxes, cls_inds)
        for o in objects:
            o.label = self._labels[o.label]
        return objects


def to_list_of_objects2d(scores: np.ndarray, bboxes: np.ndarray, class_ids: np.ndarray) -> list[Object2D]:
    objects = []
    for score, bbox, class_id in zip(scores, bboxes, class_ids, strict=True):
        box = BBox2D(y0=bbox[1], x0=bbox[0], y1=bbox[2], x1=bbox[3])
        obj = Object2D(score, box, int(class_id))
        objects.append(obj)
    return objects
