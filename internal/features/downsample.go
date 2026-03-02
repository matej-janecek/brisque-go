package features

import (
	"image"

	"github.com/matej/brisque-go/internal/imageutil"
)

// cubicWeight holds the precomputed 1D bicubic interpolation weights
// for exactly 2x downsampling (fractional offset = 0.5, a = -0.75).
// Source pixels: center-1, center, center+1, center+2
var cubicWeight = [4]float64{-0.09375, 0.59375, 0.59375, -0.09375}

// ResizeCubicHalf performs a 2x bicubic downsampling of src into dst,
// matching OpenCV's cv::resize with INTER_CUBIC and BORDER_REFLECT_101.
func ResizeCubicHalf(dst, src *imageutil.FloatImage) {
	srcW := src.Width()
	srcH := src.Height()
	dstW := srcW / 2
	dstH := srcH / 2

	dst.Reset(image.Rect(0, 0, dstW, dstH))

	for dy := 0; dy < dstH; dy++ {
		sy := 2 * dy // center source y
		dstRow := dst.Pix[dy*dst.Stride : dy*dst.Stride+dstW]
		for dx := 0; dx < dstW; dx++ {
			sx := 2 * dx // center source x
			sum := 0.0
			for ky := 0; ky < 4; ky++ {
				py := reflectBorder101(sy-1+ky, srcH)
				srcRow := src.Pix[py*src.Stride:]
				wy := cubicWeight[ky]
				for kx := 0; kx < 4; kx++ {
					px := reflectBorder101(sx-1+kx, srcW)
					sum += wy * cubicWeight[kx] * float64(srcRow[px])
				}
			}
			dstRow[dx] = float32(sum)
		}
	}
}

// reflectBorder101 implements BORDER_REFLECT_101 (reflect around edge pixel).
// For image [0, size-1]: position -1 maps to 1, position size maps to size-2.
func reflectBorder101(p, size int) int {
	if p < 0 {
		return -p
	}
	if p >= size {
		return 2*size - 2 - p
	}
	return p
}
