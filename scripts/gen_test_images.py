#!/usr/bin/env python3
"""Generate additional test images for BRISQUE integration tests.

Creates variants from existing Kodak images to cover:
- Different JPEG quality levels (q=10, q=30, q=70, q=90)
- Grayscale JPEG
- PNG (no decoder difference)
- Wide panoramic crop
- Near-minimum size (32x32)
- Large image (1920x1080 upscale)
"""
import cv2
import numpy as np
import os

GOLDEN = "testdata/golden"
# Use landscape_high.jpg as the source for variants
src = cv2.imread(os.path.join(GOLDEN, "landscape_high.jpg"), cv2.IMREAD_COLOR | 128)
h, w = src.shape[:2]

# 1. JPEG quality variants from the same source
for q in [10, 30, 70, 90]:
    path = os.path.join(GOLDEN, f"landscape_q{q}.jpg")
    cv2.imwrite(path, src, [cv2.IMWRITE_JPEG_QUALITY, q])
    sz = os.path.getsize(path)
    print(f"Created {path} ({w}x{h}, q={q}, {sz/1024:.1f}K)")

# 2. Grayscale JPEG
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
path = os.path.join(GOLDEN, "landscape_gray.jpg")
cv2.imwrite(path, gray, [cv2.IMWRITE_JPEG_QUALITY, 80])
sz = os.path.getsize(path)
print(f"Created {path} ({w}x{h}, grayscale q=80, {sz/1024:.1f}K)")

# 3. PNG (lossless, no JPEG decoder difference)
path = os.path.join(GOLDEN, "landscape.png")
cv2.imwrite(path, src)
sz = os.path.getsize(path)
print(f"Created {path} ({w}x{h}, PNG, {sz/1024:.1f}K)")

# 4. Wide panoramic crop (768x128)
pano = src[:128, :, :]
ph, pw = pano.shape[:2]
path = os.path.join(GOLDEN, "panoramic_wide.jpg")
cv2.imwrite(path, pano, [cv2.IMWRITE_JPEG_QUALITY, 80])
sz = os.path.getsize(path)
print(f"Created {path} ({pw}x{ph}, panoramic, {sz/1024:.1f}K)")

# 5. Near-minimum size (32x32 crop)
small = cv2.resize(src, (32, 32), interpolation=cv2.INTER_CUBIC)
sh, sw = small.shape[:2]
path = os.path.join(GOLDEN, "tiny_32x32.jpg")
cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 80])
sz = os.path.getsize(path)
print(f"Created {path} ({sw}x{sh}, tiny, {sz/1024:.1f}K)")

# 6. Large image (1920x1280 upscale)
large = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_CUBIC)
lh, lw = large.shape[:2]
path = os.path.join(GOLDEN, "large_1920.jpg")
cv2.imwrite(path, large, [cv2.IMWRITE_JPEG_QUALITY, 85])
sz = os.path.getsize(path)
print(f"Created {path} ({lw}x{lh}, large, {sz/1024:.1f}K)")

# 7. Synthetic gradient (tests uniform content / edge case for MSCN)
grad = np.zeros((256, 256), dtype=np.uint8)
for y in range(256):
    grad[y, :] = y
path = os.path.join(GOLDEN, "gradient_256.jpg")
cv2.imwrite(path, grad, [cv2.IMWRITE_JPEG_QUALITY, 95])
sz = os.path.getsize(path)
print(f"Created {path} (256x256, synthetic gradient, {sz/1024:.1f}K)")

# 8. Noisy image (high BRISQUE score expected)
noisy = np.random.RandomState(42).randint(0, 256, (256, 256), dtype=np.uint8)
path = os.path.join(GOLDEN, "noise_256.jpg")
cv2.imwrite(path, noisy, [cv2.IMWRITE_JPEG_QUALITY, 90])
sz = os.path.getsize(path)
print(f"Created {path} (256x256, random noise, {sz/1024:.1f}K)")

print("\nDone. Generated 8 new test images.")
