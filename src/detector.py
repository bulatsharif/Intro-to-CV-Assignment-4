import json
import logging
from typing import Dict, List

import numpy as np


def detect_objects(frames: List[np.ndarray]) -> List[Dict]:
    try:
        import cv2
    except ImportError:
        logging.warning("OpenCV not installed, skipping detection")
        return []

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    results: List[Dict] = []
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
        frame_boxes = []
        for (x, y, w, h) in boxes:
            frame_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        if frame_boxes:
            results.append({"frame": idx, "boxes": frame_boxes})
    logging.info("Detection completed with %d frames containing objects", len(results))
    return results


def save_detections(results: List[Dict], path: str) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info("Saved detections to %s", path)
