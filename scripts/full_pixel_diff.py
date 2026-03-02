#!/usr/bin/env python3
"""Full pixel-level diff between Go (Y channel) and OpenCV grayscale."""
import subprocess, json, sys, struct
import cv2
import numpy as np

path = sys.argv[1]

# OpenCV: imread -> BGR -> cvtColor -> Gray (uint8)
img = cv2.imread(path, cv2.IMREAD_COLOR | 128)  # no EXIF rotation
gray_ocv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

# Go: dump all Y channel values as raw binary
result = subprocess.run(
    ["go", "run", "scripts/dump_all_pixels_go.go", path],
    capture_output=True
)
if result.returncode != 0:
    print("Go error:", result.stderr.decode(), file=sys.stderr)
    sys.exit(1)

# Parse: first 8 bytes = width(u32) + height(u32), rest = float32 pixels
header = result.stdout[:8]
w, h = struct.unpack('<II', header)
pixels = np.frombuffer(result.stdout[8:], dtype=np.float32).reshape(h, w)

print(f"Image: {path} ({w}x{h})")
print(f"OpenCV shape: {gray_ocv.shape}, Go shape: {pixels.shape}")
print()

diff = pixels - gray_ocv
absdiff = np.abs(diff)

print(f"Pixel diff stats:")
print(f"  Mean absolute diff: {absdiff.mean():.4f}")
print(f"  Max absolute diff:  {absdiff.max():.1f}")
print(f"  Std of diff:        {diff.std():.4f}")
print(f"  Mean diff (signed): {diff.mean():.4f}")
print()

# Histogram of absolute diffs
for threshold in [0, 0.5, 1, 2, 3, 5, 10, 20]:
    count = (absdiff > threshold).sum()
    pct = 100.0 * count / absdiff.size
    print(f"  Pixels with |diff| > {threshold:5.1f}: {count:8d} ({pct:.2f}%)")
print()

# Show where the largest diffs are
if absdiff.max() > 0:
    # Find top-10 largest diffs
    flat_idx = np.argsort(absdiff.ravel())[::-1][:20]
    print("Top 20 largest pixel diffs (y, x, Go, OpenCV, diff):")
    for idx in flat_idx:
        y, x = divmod(idx, w)
        print(f"  ({y:3d},{x:3d}): Go={pixels[y,x]:6.1f} OCV={gray_ocv[y,x]:6.1f} diff={diff[y,x]:+.1f}")
