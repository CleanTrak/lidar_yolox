import cv2
import numpy as np
from cleantrak.object_detection_interface import Object2D


def draw_objects(image: np.ndarray, objects2d: list[Object2D]):
    """
    Draws outline rectangles and text labels for each Object2D on an RGB image.
    - Input image is RGB; no color conversions are performed.
    - Each unique label gets a distinct RGB color from an 8-color palette.
    - Color assignment is stable for the same label *set* regardless of order
      (mapping uses sorted unique labels).
    - Rectangles are outlines only (no fills).

    Args:
        image: RGB uint8 array (H, W, 3).
        objects2d: sequence of Object2D(score, bbox, label).
    """

    assert image.ndim == 3 and image.shape[2] == 3
    assert image.dtype == np.uint8

    # 8 well-distinguishable colors (RGB)
    palette_rgb = [
        (255,   0,   0),  # red
        (  0, 255,   0),  # green
        (  0,   0, 255),  # blue
        (255, 165,   0),  # orange
        (255,   0, 255),  # magenta
        (  0, 255, 255),  # cyan
        (255, 255,   0),  # yellow
        (148,   0, 211),  # violet
    ]

    labels = [o.label for o in objects2d]
    uniq_sorted = sorted(set(labels))
    label2idx = {lab: i for i, lab in enumerate(uniq_sorted)}

    h, w = image.shape[:2]
    # scale thickness and font for consistent look across sizes
    thickness = max(1, (h + w) // 800)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(1.2, (h + w) / 2000.0))
    text_thickness = max(1, thickness)  # outline thickness for text

    for obj in objects2d:
        bb = obj.bbox
        x1 = int(round(bb.x0)); y1 = int(round(bb.y0))
        x2 = int(round(bb.x1)); y2 = int(round(bb.y1))

        # clamp to image bounds and skip invalid boxes
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        color_rgb = palette_rgb[label2idx[obj.label] % len(palette_rgb)]

        # Rectangle (outline only)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_rgb, thickness=thickness, lineType=cv2.LINE_AA)

        # Text label (scaled) — include score as percent with two decimals
        # score is always present
        text = f"{obj.label} {int(obj.score * 100)}%"
        (tw, th), base = cv2.getTextSize(text, font, font_scale, text_thickness)

        # Place text above the box if possible, else just inside the top-left corner
        tx = x1 + 2
        ty_above = y1 - 4
        if ty_above - th - base >= 0:
            ty = ty_above
        else:
            ty = y1 + th + 2  # inside the box

        cv2.putText(image, text, (tx, ty), font, font_scale, color_rgb, text_thickness, cv2.LINE_AA)
