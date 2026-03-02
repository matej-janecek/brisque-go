# brisque-go

Pure Go implementation of the [BRISQUE](https://ieeexplore.ieee.org/document/6272356) (Blind/Referenceless Image Spatial Quality Evaluator) algorithm for no-reference image quality assessment.

Lower scores indicate better perceptual quality. Typical range is 0–100 for natural images.

## Features

- **Zero dependencies** — pure Go, no CGo, no OpenCV required
- **Zero-allocation scoring** via pre-allocated `Workspace`
- **Concurrency-safe** — immutable `Model`, safe for concurrent use
- **Batch processing** — `ScoreBatch` with bounded worker pool
- **Fast paths** for `image.Gray`, `image.RGBA`, `image.NRGBA`, and raw grayscale buffers
- **Embedded default model** — no external files needed at runtime
- **Context support** — cancellation propagated through all code paths

## Install

```
go get github.com/matej/brisque-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "image/jpeg"
    "os"

    brisque "github.com/matej/brisque-go"
)

func main() {
    model := brisque.DefaultModel()

    f, _ := os.Open("photo.jpg")
    defer f.Close()
    img, _ := jpeg.Decode(f)

    score, err := model.ScoreImage(context.Background(), img)
    if err != nil {
        fmt.Fprintln(os.Stderr, err)
        os.Exit(1)
    }
    fmt.Printf("BRISQUE score: %.2f\n", score)
}
```

## API

### Model Construction

```go
// Use the embedded default model (trained on LIVE-R2 database)
model := brisque.DefaultModel()

// Load a custom model from file
model, err := brisque.LoadModelFromFile("my_model.txt")

// Load from any io.Reader
model, err := brisque.NewModel(reader)

// With options
model := brisque.DefaultModel(
    brisque.WithParallelThreshold(500_000),
    brisque.WithWorkspacePool(myPool),
)
```

### Scoring

```go
ctx := context.Background()

// Score any image.Image
score, err := model.ScoreImage(ctx, img)

// Score raw grayscale bytes (fastest path, skips color conversion)
score, err := model.ScoreGray(ctx, pixelBytes, width, height)

// Zero-allocation path with pre-allocated workspace
ws := brisque.NewWorkspace(1920, 1080)
score, err := model.ScoreWithWorkspace(ctx, ws, img)

// Batch scoring (concurrent, bounded by GOMAXPROCS)
scores, err := model.ScoreBatch(ctx, images)
```

### Error Handling

All errors are structured types with actionable metadata:

```go
score, err := model.ScoreImage(ctx, img)
if err != nil {
    switch e := err.(type) {
    case *brisque.ErrImageTooSmall:
        fmt.Printf("need at least %dx%d, got %dx%d\n",
            e.MinWidth, e.MinHeight, e.Width, e.Height)
    case *brisque.ErrDegenerateDistribution:
        fmt.Printf("fitting failed at scale %d\n", e.Scale)
    }
}
```

## CLI

```
go install github.com/matej/brisque-go/cmd/brisque@latest
```

```bash
# Single image
brisque photo.jpg

# Multiple images
brisque *.jpg

# Directory
brisque ./images/

# JSON output
brisque --json photo.jpg

# Control concurrency
brisque --workers 4 ./images/
```

## Performance

Benchmarked on an Intel i9-13900KS:

| Benchmark | Time | Allocs |
|-----------|------|--------|
| ScoreImage 1080p | ~128 ms | 2 |
| ScoreWithWorkspace 1080p | ~108 ms | **0** |

Minimum image size: 16x16 pixels.

## How It Works

BRISQUE computes 36 features from natural scene statistics:

1. Convert to grayscale, compute local mean and variance via Gaussian convolution
2. Compute MSCN (Mean Subtracted Contrast Normalized) coefficients
3. Fit an Asymmetric Generalized Gaussian Distribution (AGGD) to MSCN coefficients (2 params)
4. Compute 4 pairwise products (H, V, D1, D2 shifts) and fit AGGD to each (4x4 = 16 params)
5. Repeat at half scale (18 features x 2 scales = 36 total)
6. Scale features to [-1, 1] and predict quality via an RBF-kernel SVR

The embedded model was trained on the [LIVE-R2](https://live.ece.utexas.edu/research/Quality/subjective.htm) image quality database using epsilon-SVR with 774 support vectors.

## Comparison with OpenCV

This implementation matches the algorithm in OpenCV's [`qualitybrisque.cpp`](https://github.com/opencv/opencv_contrib/blob/master/modules/quality/src/qualitybrisque.cpp) and uses the same embedded model. Both produce equivalent results on normal images, with small numerical differences due to floating-point precision.

### Verified behavior

The integration test (`go test -tags integration`) compares features and scores against OpenCV across 17 test images covering quality levels (q=5 through q=95), formats (JPEG, grayscale JPEG, PNG), sizes (32×32 to 1920×1280), aspect ratios (panoramic to portrait), and synthetic content (noise). All algorithmic steps match OpenCV:

- Gaussian convolution with `BORDER_REPLICATE`
- AGGD fitting for both MSCN coefficients and pairwise shift products
- Zero-padded shifted products (H, V, D1, D2)
- Bicubic 2x downsampling (`INTER_CUBIC` with `BORDER_REFLECT_101`)
- Linear-scan alpha solver (step = 0.001)
- Score clamped to [0, 100]

### Why scores differ slightly

The image processing pipeline (convolution, MSCN, shifts, downsampling) uses **float32** to match OpenCV's `CV_32F` precision. The AGGD fitting and SVR prediction use **float64** (OpenCV also uses `double` for these). For JPEG images, the Y (luminance) channel is extracted directly from the YCbCr data, avoiding an unnecessary RGB round-trip. Remaining differences come from the JPEG decoder (Go's `image/jpeg` vs OpenCV's libjpeg):

| Image | Go | OpenCV | Difference |
|---|---|---|---|
| Kodak landscape (q=95) | 19.60 | 19.64 | 0.04 |
| Kodak landscape (q=90) | 24.25 | 24.23 | 0.03 |
| Kodak landscape (q=70) | 15.39 | 15.37 | 0.01 |
| Kodak landscape (q=30) | 24.70 | 25.23 | 0.53 |
| Kodak landscape (q=10) | 55.47 | 55.71 | 0.25 |
| Kodak landscape grayscale | 15.89 | 15.84 | 0.04 |
| Kodak landscape PNG | 19.57 | 19.64 | 0.07 |
| Kodak lighthouse (q=50) | 17.69 | 17.81 | 0.12 |
| Lena (q=80) | 8.91 | 8.86 | 0.04 |
| iPhone 6 photo | 21.44 | 22.57 | 1.13 |
| Large 1920×1280 | 50.22 | 50.20 | 0.02 |
| Panoramic 768×128 | 14.88 | 14.65 | 0.22 |
| Random noise 256×256 | 100.00 | 100.00 | 0.00 |

On heavily compressed or very small images the gap widens:

| Image | Go | OpenCV | Difference |
|---|---|---|---|
| Kodak scene (q=5) | 75.98 | 71.82 | 4.16 |
| Tiny 32×32 | 57.29 | 56.41 | 0.88 |

The sole source of these differences is the JPEG decoder: Go's `image/jpeg` and OpenCV's libjpeg use different IDCT implementations. The JPEG standard (IEEE 1180-1990) allows this variance — both produce valid decodings. For heavily compressed images (~2% of pixels differ by 1–21 values), the small pixel differences get amplified through the BRISQUE pipeline:

1. **MSCN coefficients** near zero flip sign between decoders, changing which AGGD branch they fall into
2. **AGGD sigma² features** diverge because the left/right split changes
3. **Feature scaling** amplifies sigma² differences (narrow value ranges map to [-1, 1])
4. **SVR with 774 support vectors** compounds the error through the RBF kernel sum

For PNG and grayscale inputs (no JPEG decoding involved), differences are < 0.1 and come only from minor rounding in the grayscale conversion formula. The scores agree on quality ranking and are suitable for thresholding.

## Project Layout

```
brisque-go/
    api.go              # ScoreImage, ScoreGray, ScoreBatch, ScoreWithWorkspace
    model.go            # Model construction and parsing
    model_default.go    # Embedded default model (auto-generated)
    workspace.go        # Pre-allocated workspace
    options.go          # Functional options
    errors.go           # Structured error types
    example_test.go     # Runnable GoDoc examples
    internal/
        conv/           # Gaussian kernel, separable convolution
        stats/          # GGD and AGGD solvers
        features/       # Feature extraction, downsampling
        scaler/         # Min-max feature scaling
        svr/            # RBF kernel SVR prediction
        imageutil/      # Float image type and conversion
    cmd/brisque/        # CLI
    scripts/            # Model conversion tools
    testdata/golden/    # Reference test images
```

## References

- Mittal, A., Moorthy, A.K., Bovik, A.C. (2012). "No-Reference Image Quality Assessment in the Spatial Domain." *IEEE Transactions on Image Processing*, 21(12), 4695-4708.
- Default model from [OpenCV contrib](https://github.com/opencv/opencv_contrib/tree/master/modules/quality/samples)

## License

MIT
