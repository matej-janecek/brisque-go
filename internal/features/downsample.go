package features

import (
	"image"

	"github.com/matej/brisque-go/internal/imageutil"
)

// Downsample2x performs a 2x box downsampling of src into dst.
// dst must be pre-allocated or will be reset to the correct size.
func Downsample2x(dst, src *imageutil.FloatImage) {
	srcW := src.Width()
	srcH := src.Height()
	dstW := srcW / 2
	dstH := srcH / 2

	dst.Reset(image.Rect(0, 0, dstW, dstH))

	for y := 0; y < dstH; y++ {
		sy := y * 2
		srcRow0 := src.Pix[sy*src.Stride : sy*src.Stride+srcW]
		srcRow1 := src.Pix[(sy+1)*src.Stride : (sy+1)*src.Stride+srcW]
		dstRow := dst.Pix[y*dst.Stride : y*dst.Stride+dstW]
		for x := 0; x < dstW; x++ {
			sx := x * 2
			dstRow[x] = (srcRow0[sx] + srcRow0[sx+1] + srcRow1[sx] + srcRow1[sx+1]) * 0.25
		}
	}
}
