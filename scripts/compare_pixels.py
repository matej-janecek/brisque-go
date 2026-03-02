#!/usr/bin/env python3
"""Compare pixel values between Go and OpenCV for a JPEG image."""
import subprocess, json, sys
import cv2
import numpy as np

path = sys.argv[1]

# OpenCV pixels
img = cv2.imread(path, cv2.IMREAD_COLOR | 128)
gray_ocv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32).flatten()

# Go pixels (run Go script, parse JSON)
result = subprocess.run(
    ["go", "run", "scripts/dump_pixels_go.go", path],
    capture_output=True, text=True
)
go_data = json.loads(result.stdout)
w, h = go_data["width"], go_data["height"]

# Get full Go pixel dump - need to modify script
# For now, compare stats and sample
print(f"Image: {path} ({w}x{h})")
print(f"OpenCV mean: {gray_ocv.mean():.6f}")
print(f"Go mean:     {go_data['mean']:.6f}")
print(f"Mean diff:   {abs(gray_ocv.mean() - go_data['mean']):.6f}")
print()

# Compare 20x20 block pixel by pixel
block_ocv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[:20, :20]
block_go = np.array(go_data["top_left_20x20"], dtype=np.float32)

diff = np.abs(block_go - block_ocv)
print(f"20x20 block pixel diffs:")
print(f"  Max diff:  {diff.max():.1f}")
print(f"  Mean diff: {diff.mean():.4f}")
print(f"  Pixels with diff>0: {(diff > 0).sum()}/{diff.size}")
print(f"  Pixels with diff>1: {(diff > 1).sum()}/{diff.size}")
print(f"  Pixels with diff>2: {(diff > 2).sum()}/{diff.size}")
print()

# Show locations of differences
if diff.max() > 0:
    print("Pixel differences (y, x, Go, OpenCV, diff):")
    for y in range(20):
        for x in range(20):
            d = abs(block_go[y][x] - block_ocv[y][x])
            if d > 0:
                print(f"  ({y:2d},{x:2d}): Go={block_go[y][x]:3.0f} OCV={block_ocv[y][x]:3.0f} diff={d:.1f}")
