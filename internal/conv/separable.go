package conv

import (
	"github.com/matej/brisque-go/internal/imageutil"
	"image"
)

// Convolve performs a separable 2D convolution of src with the given
// symmetric 1D kernel. The result is a valid convolution (output shrinks
// by kernel_size-1 in each dimension). dst and tmp must be pre-allocated
// with sufficient size. dst will be resized to the valid output region.
//
// tmp is used as intermediate storage for the horizontal pass.
func Convolve(dst, src *imageutil.FloatImage, kernel []float64, tmp []float64) {
	ksize := len(kernel)
	half := ksize / 2
	srcW := src.Width()
	srcH := src.Height()

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
			sum := 0.0
			for k := 0; k < ksize; k++ {
				sum += srcRow[x+k] * kernel[k]
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
			sum := 0.0
			for k := 0; k < ksize; k++ {
				sum += tmp[(y+k)*tmpStride+x] * kernel[k]
			}
			dstRow[x] = sum
		}
	}
}
