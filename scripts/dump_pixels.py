#!/usr/bin/env python3
"""Dump grayscale pixel values from OpenCV for comparison with Go."""
import sys
import json
import cv2
import numpy as np

img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR | 128)  # no EXIF rotation
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_f = gray.astype(np.float32)

h, w = gray_f.shape
# Dump first 20x20 block and stats
block = gray_f[:20, :20].tolist()
print(json.dumps({
    "width": w, "height": h,
    "mean": float(gray_f.mean()),
    "std": float(gray_f.std()),
    "min": float(gray_f.min()),
    "max": float(gray_f.max()),
    "top_left_20x20": block,
}))
