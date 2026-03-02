package conv

import (
	"image"

	"github.com/matej/brisque-go/internal/imageutil"
)

// Convolve performs a separable 2D convolution of src with the given
// symmetric 1D kernel. The result is a valid convolution (output shrinks
// by kernel_size-1 in each dimension). dst and tmp must be pre-allocated
// with sufficient size. dst will be resized to the valid output region.
//
// tmp is used as intermediate storage for the horizontal pass.
// Computation is done in float32 to match OpenCV's CV_32F filter engine.
func Convolve(dst, src *imageutil.FloatImage, kernel []float64, tmp []float32) {
	ksize := len(kernel)
	half := ksize / 2
	srcW := src.Width()
	srcH := src.Height()

	// Convert kernel to float32 (matches OpenCV's RowVec_32f)
	kf := make([]float32, ksize)
	for i, v := range kernel {
		kf[i] = float32(v)
	}

	// Valid output dimensions
	outW := srcW - ksize + 1
	outH := srcH - ksize + 1
	if outW <= 0 || outH <= 0 {
		dst.Reset(image.Rect(0, 0, 0, 0))
		return
	}

	// Horizontal pass: src (srcW x srcH) -> tmp (outW x srcH)
	tmpStride := outW
	_ = tmp[:tmpStride*srcH] // bounds check hint

	for y := 0; y < srcH; y++ {
		srcRow := src.Pix[y*src.Stride : y*src.Stride+srcW]
		tmpRow := tmp[y*tmpStride : y*tmpStride+outW]
		for x := 0; x < outW; x++ {
			var sum float32
			for k := 0; k < ksize; k++ {
				sum += srcRow[x+k] * kf[k]
			}
			tmpRow[x] = sum
		}
	}

	// Vertical pass: tmp (outW x srcH) -> dst (outW x outH)
	dst.Reset(image.Rect(
		src.Rect.Min.X+half,
		src.Rect.Min.Y+half,
		src.Rect.Min.X+half+outW,
		src.Rect.Min.Y+half+outH,
	))

	for y := 0; y < outH; y++ {
		dstRow := dst.Pix[y*dst.Stride : y*dst.Stride+outW]
		for x := 0; x < outW; x++ {
			var sum float32
			for k := 0; k < ksize; k++ {
				sum += tmp[(y+k)*tmpStride+x] * kf[k]
			}
			dstRow[x] = sum
		}
	}
}

// ConvolveReplicate performs a separable 2D convolution with BORDER_REPLICATE
// padding, producing a same-size output. This matches OpenCV's GaussianBlur
// with cv::BORDER_REPLICATE using float32 throughout (CV_32F filter engine).
func ConvolveReplicate(dst, src *imageutil.FloatImage, kernel []float64, tmp []float32) {
	ksize := len(kernel)
	half := ksize / 2
	srcW := src.Width()
	srcH := src.Height()

	// Convert kernel to float32 (matches OpenCV's RowVec_32f)
	kf := make([]float32, ksize)
	for i, v := range kernel {
		kf[i] = float32(v)
	}

	dst.Reset(image.Rect(0, 0, srcW, srcH))

	tmpStride := srcW
	_ = tmp[:tmpStride*srcH] // bounds check hint

	// Horizontal pass with BORDER_REPLICATE
	for y := 0; y < srcH; y++ {
		srcRow := src.Pix[y*src.Stride : y*src.Stride+srcW]
		tmpRow := tmp[y*tmpStride : y*tmpStride+srcW]

		// Left border
		for x := 0; x < half; x++ {
			var sum float32
			for k := 0; k < ksize; k++ {
				sx := x + k - half
				if sx < 0 {
					sx = 0
				}
				sum += srcRow[sx] * kf[k]
			}
			tmpRow[x] = sum
		}

		// Interior (no clamping)
		for x := half; x < srcW-half; x++ {
			var sum float32
			for k := 0; k < ksize; k++ {
				sum += srcRow[x-half+k] * kf[k]
			}
			tmpRow[x] = sum
		}

		// Right border
		for x := srcW - half; x < srcW; x++ {
			var sum float32
			for k := 0; k < ksize; k++ {
				sx := x + k - half
				if sx >= srcW {
					sx = srcW - 1
				}
				sum += srcRow[sx] * kf[k]
			}
			tmpRow[x] = sum
		}
	}

	// Vertical pass with BORDER_REPLICATE
	for y := 0; y < srcH; y++ {
		dstRow := dst.Pix[y*dst.Stride : y*dst.Stride+srcW]

		if y >= half && y < srcH-half {
			// Interior rows (no clamping)
			for x := 0; x < srcW; x++ {
				var sum float32
				for k := 0; k < ksize; k++ {
					sum += tmp[(y-half+k)*tmpStride+x] * kf[k]
				}
				dstRow[x] = sum
			}
		} else {
			// Border rows
			for x := 0; x < srcW; x++ {
				var sum float32
				for k := 0; k < ksize; k++ {
					sy := y + k - half
					if sy < 0 {
						sy = 0
					} else if sy >= srcH {
						sy = srcH - 1
					}
					sum += tmp[sy*tmpStride+x] * kf[k]
				}
				dstRow[x] = sum
			}
		}
	}
}
