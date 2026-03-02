#!/usr/bin/env python3
"""
Compare Python BRISQUE vs Go BRISQUE implementation.

Loads images WITHOUT EXIF rotation (matching Go's image/jpeg decoder)
and computes BRISQUE using OpenCV's built-in QualityBRISQUE.

Usage:
    pip install opencv-contrib-python numpy
    python3 scripts/compare_brisque.py testdata/golden/sample.jpg
    python3 scripts/compare_brisque.py --json testdata/golden/sample.jpg
"""

import argparse
import json
import sys

import cv2
import numpy as np

# IMREAD_IGNORE_ORIENTATION = 128 — skip EXIF rotation to match Go's decoder
_IMREAD_FLAGS = cv2.IMREAD_COLOR | 128


def compute_brisque_opencv(image_path, model_path, range_path):
    """Compute BRISQUE score and features using OpenCV (no EXIF rotation)."""
    img = cv2.imread(image_path, _IMREAD_FLAGS)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    score = None
    features = None
    shape = img.shape  # (rows, cols, channels)

    try:
        result = cv2.quality.QualityBRISQUE_compute(
            img, model_path, range_path
        )
        score = float(result[0]) if isinstance(result, tuple) else float(result[0][0])
    except Exception:
        pass

    try:
        feat_mat = np.zeros((1, 36), dtype=np.float32)
        cv2.quality.QualityBRISQUE_computeFeatures(img, feat_mat)
        features = feat_mat.flatten().tolist()
    except Exception:
        pass

    return score, features, (shape[1], shape[0])  # (width, height)


def main():
    parser = argparse.ArgumentParser(description="Compare BRISQUE implementations")
    parser.add_argument("images", nargs="+", help="Image file paths to score")
    parser.add_argument(
        "--model",
        default="scripts/brisque_model_live.yml",
        help="Path to OpenCV BRISQUE model YAML",
    )
    parser.add_argument(
        "--range",
        default="scripts/brisque_range_live.yml",
        help="Path to OpenCV BRISQUE range YAML",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = []

    for image_path in args.images:
        result = {"image": image_path}

        try:
            score, features, (w, h) = compute_brisque_opencv(
                image_path, args.model, args.range
            )
            result["image_size"] = {"width": w, "height": h}
            if score is not None:
                result["opencv_score"] = score
            if features is not None:
                result["opencv_features"] = features
        except Exception as e:
            result["opencv_error"] = str(e)

        results.append(result)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"\n=== {r['image']} ===")
            if "image_size" in r:
                sz = r["image_size"]
                print(f"  Size: {sz['width']}x{sz['height']}")
            if "opencv_score" in r:
                print(f"  OpenCV score: {r['opencv_score']:.6f}")
            elif "opencv_error" in r:
                print(f"  OpenCV error: {r['opencv_error']}")
            if "opencv_features" in r:
                feats = r["opencv_features"]
                print(f"  OpenCV features ({len(feats)}):")
                for i, f in enumerate(feats):
                    print(f"    [{i:2d}] {f:.10f}")


if __name__ == "__main__":
    main()
