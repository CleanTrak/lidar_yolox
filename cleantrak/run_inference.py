#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}/..")

import cv2
from cleantrak.onnx_object_detector import OnnxObjectDetector
from cleantrak.draw_utils import draw_objects


def parse_args():
    p = argparse.ArgumentParser(description="Run ONNX detector on dataset or an image")
    p.add_argument("-m", "--model", required=True, help="Path to ONNX model")
    p.add_argument("-i", "--input", required=True, help="Path to dataset or input image")
    p.add_argument("-o", "--output", required=True, help="Path to save output coco annotations or image")
    return p.parse_args()


def main():
    args = parse_args()

    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"Failed to read image: {args.input}", file=sys.stderr)
        return 1

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    detector = OnnxObjectDetector(args.model)
    objects = detector.detect_objects(img_rgb)

    draw_objects(img_rgb, objects)

    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)

    print(f"Saved result to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())